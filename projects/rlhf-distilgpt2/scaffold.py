"""RLHF from scratch on distilgpt2 — end-to-end demo.

Run once you've solved (enough of) the steps:

    python projects.py rlhf-distilgpt2 --scaffold

Walks the whole pipeline on the real distilgpt2 (downloaded once): decoding ->
SFT -> LoRA -> reward modeling -> PPO ingredients -> DPO -> evaluation + chat.
Everything is intentionally tiny so it runs in well under a minute on a GPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from solution import *  # noqa: E402,F401,F403


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = set_pad_token_to_eos(load_distilgpt2_tokenizer())
    model = load_distilgpt2_model().to(dev)
    print(f"loaded distilgpt2 on {dev}")

    print("\n=== 1) decoding ===")
    print("greedy:", repr(generate_and_decode(model, tok, "The capital of France is", 8)))
    logits = model(tok("Hello", return_tensors="pt").input_ids.to(dev)).logits[0, -1]
    print("top-k(5) keeps", int(torch.isfinite(top_k_filter(logits, 5)).sum()), "tokens;",
          "top-p(0.9) keeps", int(torch.isfinite(top_p_filter(logits, 0.9)).sum()))

    print("\n=== 2) SFT (loss should drop) ===")
    data = build_synthetic_instruction_dataset(16)
    tokenized = [tokenize_example(apply_template(format_example(e)), tok) for e in data]
    batches = [collate_lm_batch(b, tok.pad_token_id)
               for b in iterate_minibatches(tokenized, 4)]
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    losses = [sft_train_step(model, batches[i % len(batches)], opt) for i in range(20)]
    print(f"  SFT loss: {losses[0]:.3f} -> {losses[-1]:.3f}")

    print("\n=== 3) LoRA ===")
    base = load_distilgpt2_model()
    before = count_trainable_params(base)
    freeze_base_params(base)
    print(f"  trainable params: {before:,} -> {count_trainable_params(base):,} after freezing base")

    print("\n=== 4) reward model (pairwise accuracy should rise) ===")
    prefs = build_synthetic_preference_dataset(16)
    formatted = [format_preference(p) for p in prefs]
    chosen = tok([f["chosen_text"] for f in formatted], return_tensors="pt", padding=True).input_ids
    rejected = tok([f["rejected_text"] for f in formatted], return_tensors="pt", padding=True).input_ids

    class RewardModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(tok.vocab_size, 32)
            self.head = torch.nn.Linear(32, 1)

        def forward(self, ids):
            return self.head(self.emb(ids).mean(1)).squeeze(-1)

    rm = RewardModel()
    ropt = torch.optim.Adam(rm.parameters(), lr=1e-2)
    acc0 = pairwise_accuracy(rm(chosen), rm(rejected)).item()
    for _ in range(60):
        reward_train_step(rm, {"chosen": chosen, "rejected": rejected}, ropt)
    acc1 = pairwise_accuracy(rm(chosen), rm(rejected)).item()
    print(f"  pairwise accuracy: {acc0:.2f} -> {acc1:.2f}")

    print("\n=== 5) PPO ingredients on a real rollout ===")
    ids = tok("The weather today is", return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        lp = batch_sequence_logprob(model, ids).cpu()
        last_logits = model(ids).logits[:, -1, :].cpu()
        kl = per_token_kl(lp, lp - 0.1)
        adv = gae_advantages(torch.tensor([0.5, 0.5]), torch.tensor([0.0, 0.0, 0.0]), 0.99, 0.95)
        surr = clipped_surrogate(policy_ratio(lp, lp - 0.05), torch.tensor([1.0]), 0.2)
        total = ppo_loss(surr, value_function_loss(torch.zeros(1), torch.ones(1)),
                         entropy_bonus(last_logits), 0.5, 0.01)
    print(f"  seq logprob={lp.item():.2f}  KL={kl.item():.3f}  adv={adv.tolist()}  ppo_loss={total.item():.3f}")

    print("\n=== 6) DPO (loss should drop) ===")
    policy = load_distilgpt2_model().to(dev)
    ref = load_distilgpt2_model().to(dev)
    for p in ref.parameters():
        p.requires_grad = False
    c = chosen.to(dev)[:8]
    r = rejected.to(dev)[:8]
    dopt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    with torch.no_grad():
        rc, rr = batch_sequence_logprob(ref, c), batch_sequence_logprob(ref, r)
    dpo_losses = []
    for _ in range(15):
        loss = dpo_loss(batch_sequence_logprob(policy, c), batch_sequence_logprob(policy, r), rc, rr, 0.1)
        dopt.zero_grad(); loss.backward(); dopt.step()
        dpo_losses.append(loss.item())
    print(f"  DPO loss: {dpo_losses[0]:.3f} -> {dpo_losses[-1]:.3f}")

    print("\n=== 7) evaluation + chat ===")
    prompts = build_eval_prompt_set()
    comps = generate_completions(model, tok, prompts, 6)
    rew = score_with_reward(lambda x: rm(x.to(next(rm.parameters()).device)), tok, comps)
    base_rew = score_with_reward(lambda x: rm(x.to(next(rm.parameters()).device)), tok,
                                 generate_completions(base.to(dev), tok, prompts, 6))
    print(f"  win rate (sft vs base, by reward model): {win_rate(rew, base_rew).item():.2f}")
    print("  chat('Say hello'):", repr(apply_stop_tokens(chat(model, tok, "Say hello", 12), ["###", "\n\n"])))


if __name__ == "__main__":
    main()
