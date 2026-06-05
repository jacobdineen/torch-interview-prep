"""Hidden tests for mini-vllm-inference-engine. One test_<id>_<name>(ns) per step; deterministic, independent oracles."""
import math  # noqa: F401

import numpy as np

def test_0001_token_qkv(ns):
    """token_qkv projects the embedding through Wq/Wk/Wv."""
    import numpy as np
    make_model = ns["make_model"]; token_qkv = ns["token_qkv"]
    model = make_model(vocab=16, d=8, seed=0)
    for tid in [0, 3, 7, 15]:
        q, k, v = token_qkv(model, tid)
        x = model['E'][tid]
        assert q.shape == (8,) and k.shape == (8,) and v.shape == (8,)
        assert np.allclose(q, x @ model['Wq'])
        assert np.allclose(k, x @ model['Wk'])
        assert np.allclose(v, x @ model['Wv'])


def test_0002_kv_attention(ns):
    """kv_attention matches a manual stable-softmax weighted average of V."""
    import numpy as np
    make_model = ns["make_model"]; kv_attention = ns["kv_attention"]
    rng = np.random.default_rng(1)
    d = 8; scale = 1.0/np.sqrt(d)
    for t in [1, 2, 5]:
        q = rng.standard_normal(d)
        K = rng.standard_normal((t, d))
        V = rng.standard_normal((t, d))
        o = kv_attention(q, K, V, scale)
        assert o.shape == (d,)
        scores = scale * (K @ q)
        m = np.max(scores)
        e = np.exp(scores - m)
        p = e / np.sum(e)
        expected = p @ V
        assert np.allclose(o, expected)
    # single token => attention returns that V exactly
    q = rng.standard_normal(d); K = rng.standard_normal((1, d)); V = rng.standard_normal((1, d))
    assert np.allclose(kv_attention(q, K, V, scale), V[0])


def test_0003_decode_step(ns):
    """decode_step appends k,v before attending and returns logits plus the new k,v."""
    import numpy as np
    make_model = ns["make_model"]; decode_step = ns["decode_step"]
    token_qkv = ns["token_qkv"]; kv_attention = ns["kv_attention"]
    model = make_model(vocab=16, d=8, seed=0)
    scale = 1.0/np.sqrt(model['d'])
    rng = np.random.default_rng(2)
    t = 3; d = model['d']
    K_cache = rng.standard_normal((t, d))
    V_cache = rng.standard_normal((t, d))
    tok = 5
    logits, k, v = decode_step(model, tok, K_cache, V_cache, scale)
    eq, ek, ev = token_qkv(model, tok)
    assert np.allclose(k, ek) and np.allclose(v, ev)
    Kf = np.vstack([K_cache, ek]); Vf = np.vstack([V_cache, ev])
    o = kv_attention(eq, Kf, Vf, scale)
    assert np.allclose(logits, o @ model['Wout'])
    assert logits.shape == (model['vocab'],)


def test_0004_greedy_generate(ns):
    """greedy_generate equals an independent hand-written canonical decode loop."""
    import numpy as np
    make_model = ns["make_model"]; greedy_generate = ns["greedy_generate"]
    token_qkv = ns["token_qkv"]
    model = make_model(vocab=16, d=8, seed=0)
    scale = 1.0/np.sqrt(model['d'])
    prompt = [1, 4, 2]; max_new = 5; d = model['d']

    # Independent oracle: full softmax attention from scratch, no decode helpers.
    Ks = []; Vs = []
    cur = None
    def process(tok):
        nonlocal cur
        q, k, v = token_qkv(model, tok)
        Ks.append(k); Vs.append(v)
        K = np.stack(Ks); V = np.stack(Vs)
        s = scale * (K @ q)
        s = s - s.max()
        p = np.exp(s); p = p / p.sum()
        o = p @ V
        cur = o @ model['Wout']
    for tok in prompt:
        process(tok)
    expected = []
    for _ in range(max_new):
        tok = int(np.argmax(cur))
        expected.append(tok)
        process(tok)

    out = greedy_generate(model, prompt, max_new, scale)
    assert out == expected
    assert len(out) == max_new
    assert all(isinstance(x, int) for x in out)
    # determinism
    assert greedy_generate(model, prompt, max_new, scale) == out

import math


def test_0005_new_block_manager(ns):
    """new_block_manager builds the expected free-list FIFO and metadata."""
    new_block_manager = ns["new_block_manager"]
    mgr = new_block_manager(5, 4)
    assert mgr['free'] == [0, 1, 2, 3, 4]
    assert mgr['block_size'] == 4
    assert mgr['num_blocks'] == 5
    # zero blocks
    mgr0 = new_block_manager(0, 4)
    assert mgr0['free'] == []
    assert mgr0['num_blocks'] == 0


def test_0006_allocate(ns):
    """allocate pops ceil(n/block_size) ids from the front and returns None on OOM."""
    new_block_manager = ns["new_block_manager"]
    allocate = ns["allocate"]

    mgr = new_block_manager(5, 4)
    # 0 tokens -> 0 blocks, empty table
    assert allocate(mgr, 0) == []
    assert mgr['free'] == [0, 1, 2, 3, 4]
    # 4 tokens -> ceil(4/4)=1 block, pop lowest from front
    assert allocate(mgr, 4) == [0]
    assert mgr['free'] == [1, 2, 3, 4]
    # 5 tokens -> ceil(5/4)=2 blocks
    assert allocate(mgr, 5) == [1, 2]
    assert mgr['free'] == [3, 4]
    # ceil math check generically
    mgr2 = new_block_manager(10, 4)
    n = 7
    got = allocate(mgr2, n)
    assert len(got) == math.ceil(n / 4) == 2
    assert got == [0, 1]
    # OOM: need more than available -> None, free-list untouched
    mgr3 = new_block_manager(2, 4)
    before = list(mgr3['free'])
    assert allocate(mgr3, 100) is None
    assert mgr3['free'] == before


def test_0007_free(ns):
    """free returns ids to the back of the list, enabling FIFO reuse."""
    new_block_manager = ns["new_block_manager"]
    allocate = ns["allocate"]
    free = ns["free"]

    mgr = new_block_manager(4, 4)
    t1 = allocate(mgr, 4)   # [0]
    t2 = allocate(mgr, 4)   # [1]
    assert mgr['free'] == [2, 3]
    free(mgr, t1)           # 0 to back
    assert mgr['free'] == [2, 3, 0]
    free(mgr, t2)           # 1 to back
    assert mgr['free'] == [2, 3, 0, 1]
    # FIFO reuse: next allocate takes from front (2 first, the oldest free)
    assert allocate(mgr, 4) == [2]
    # multi-id free preserves order
    mgr2 = new_block_manager(3, 4)
    tb = allocate(mgr2, 12)  # [0,1,2]
    assert mgr2['free'] == []
    free(mgr2, tb)
    assert mgr2['free'] == [0, 1, 2]


def test_0008_slot_for_token(ns):
    """slot_for_token maps logical index to (block id, offset) by hand-computed math."""
    slot_for_token = ns["slot_for_token"]
    bt = [7, 3, 9]  # arbitrary physical block ids
    bs = 4
    expected = {
        0: (7, 0), 1: (7, 1), 2: (7, 2), 3: (7, 3),
        4: (3, 0), 5: (3, 1), 6: (3, 2), 7: (3, 3),
        8: (9, 0), 11: (9, 3),
    }
    for idx, exp in expected.items():
        assert slot_for_token(bt, idx, bs) == exp
    # direct formula cross-check
    for idx in range(12):
        bid, off = slot_for_token(bt, idx, bs)
        assert bid == bt[idx // bs]
        assert off == idx % bs

def test_0009_write_kv(ns):
    """write_kv places k,v at the correct physical block/offset in-place."""
    import numpy as np
    write_kv = ns["write_kv"]
    num_blocks, block_size, d = 5, 4, 8
    k_pool = np.zeros((num_blocks, block_size, d))
    v_pool = np.zeros((num_blocks, block_size, d))
    block_table = [2, 0, 3]
    # token_index 5 -> block_table[5//4]=block_table[1]=0, off=5%4=1
    k = np.arange(d, dtype=float) + 1.0
    v = np.arange(d, dtype=float) + 100.0
    write_kv(k_pool, v_pool, block_table, 5, block_size, k, v)
    assert np.allclose(k_pool[0, 1], k)
    assert np.allclose(v_pool[0, 1], v)
    # everything else still zero
    k_pool[0, 1] = 0.0
    v_pool[0, 1] = 0.0
    assert np.allclose(k_pool, 0.0) and np.allclose(v_pool, 0.0)


def test_0010_gather_kv(ns):
    """gather_kv round-trips writes back into logically-contiguous order."""
    import numpy as np
    write_kv = ns["write_kv"]
    gather_kv = ns["gather_kv"]
    num_blocks, block_size, d = 6, 4, 8
    k_pool = np.zeros((num_blocks, block_size, d))
    v_pool = np.zeros((num_blocks, block_size, d))
    block_table = [3, 1]  # supports up to 8 tokens
    rng = np.random.default_rng(0)
    num_tokens = 7
    Kref = rng.standard_normal((num_tokens, d))
    Vref = rng.standard_normal((num_tokens, d))
    for t in range(num_tokens):
        write_kv(k_pool, v_pool, block_table, t, block_size, Kref[t], Vref[t])
    K, V = gather_kv(k_pool, v_pool, block_table, num_tokens, block_size)
    assert K.shape == (num_tokens, d) and V.shape == (num_tokens, d)
    assert np.allclose(K, Kref)
    assert np.allclose(V, Vref)


def test_0011_paged_attention(ns):
    """paged_attention equals a manual stable-softmax attention over the same tokens."""
    import numpy as np
    write_kv = ns["write_kv"]
    paged_attention = ns["paged_attention"]
    num_blocks, block_size, d = 6, 4, 8
    k_pool = np.zeros((num_blocks, block_size, d))
    v_pool = np.zeros((num_blocks, block_size, d))
    block_table = [2, 5]
    rng = np.random.default_rng(1)
    num_tokens = 6
    K = rng.standard_normal((num_tokens, d))
    V = rng.standard_normal((num_tokens, d))
    for t in range(num_tokens):
        write_kv(k_pool, v_pool, block_table, t, block_size, K[t], V[t])
    q = rng.standard_normal(d)
    scale = 1.0 / np.sqrt(d)
    o = paged_attention(q, k_pool, v_pool, block_table, num_tokens, block_size, scale)
    # manual stable softmax oracle
    scores = scale * (K @ q)
    scores = scores - np.max(scores)
    p = np.exp(scores)
    p = p / np.sum(p)
    o_ref = p @ V
    assert o.shape == (d,)
    assert np.allclose(o, o_ref)


def test_0012_paged_generate(ns):
    """paged_generate reproduces a hand-written canonical sequential decode exactly."""
    import numpy as np
    make_model = ns["make_model"]
    token_qkv = ns["token_qkv"]
    kv_attention = ns["kv_attention"]
    paged_generate = ns["paged_generate"]
    model = make_model(vocab=16, d=8, seed=0)
    d = model['d']
    scale = 1.0 / np.sqrt(d)
    prompt = [3, 1, 4]
    max_new = 5
    # independent canonical decode with contiguous cache
    K = np.zeros((0, d))
    V = np.zeros((0, d))
    cur = None
    for tok in prompt:
        q, k, v = token_qkv(model, tok)
        K = np.vstack([K, k])
        V = np.vstack([V, v])
        o = kv_attention(q, K, V, scale)
        cur = o @ model['Wout']
    expected = []
    for _ in range(max_new):
        t = int(np.argmax(cur))
        expected.append(t)
        q, k, v = token_qkv(model, t)
        K = np.vstack([K, k])
        V = np.vstack([V, v])
        o = kv_attention(q, K, V, scale)
        cur = o @ model['Wout']
    out = paged_generate(model, prompt, max_new, scale, num_blocks=8, block_size=4)
    assert out == expected
    assert all(isinstance(x, int) for x in out)
    assert len(out) == max_new

def test_0013_can_admit(ns):
    """can_admit reflects ceil(n_tokens/block_size) vs free-block count."""
    import math
    new_block_manager = ns["new_block_manager"]
    can_admit = ns["can_admit"]
    mgr = new_block_manager(num_blocks=3, block_size=4)
    # 3 blocks * 4 = 12 token capacity
    assert can_admit(mgr, 0) is True
    assert can_admit(mgr, 1) is True
    assert can_admit(mgr, 12) is True            # exactly ceil(12/4)=3 == free
    assert can_admit(mgr, 13) is False           # ceil(13/4)=4 > 3
    assert can_admit(mgr, 8) == (len(mgr['free']) >= math.ceil(8 / 4))
    # after draining, capacity shrinks
    mgr['free'] = [0]
    assert can_admit(mgr, 4) is True
    assert can_admit(mgr, 5) is False


def test_0014_prefill_request(ns):
    """prefill_request returns last-token logits matching a contiguous canonical decode."""
    import math
    import numpy as np
    make_model = ns["make_model"]
    new_block_manager = ns["new_block_manager"]
    allocate = ns["allocate"]
    decode_step = ns["decode_step"]
    prefill_request = ns["prefill_request"]

    model = make_model(vocab=16, d=8, seed=0)
    d = model['d']
    scale = 1.0 / math.sqrt(d)
    block_size = 4
    num_blocks = 10
    prompt = [2, 5, 1]
    max_new = 3

    mgr = new_block_manager(num_blocks, block_size)
    bt = allocate(mgr, len(prompt) + max_new)
    k_pool = np.zeros((num_blocks, block_size, d))
    v_pool = np.zeros((num_blocks, block_size, d))

    req = prefill_request(model, 'r0', prompt, bt, k_pool, v_pool, block_size, scale, max_new)

    # independent oracle: plain contiguous decode over the prompt
    K = np.zeros((0, d)); V = np.zeros((0, d)); cur = None
    for t in prompt:
        cur, k, v = decode_step(model, t, K, V, scale)
        K = np.vstack([K, k]); V = np.vstack([V, v])

    assert req['id'] == 'r0'
    assert req['block_table'] == bt
    assert req['num_tokens'] == len(prompt)
    assert req['steps_left'] == max_new
    assert req['generated'] == []
    assert np.allclose(req['cur_logits'], cur)
    assert req['cur_logits'].shape == (model['vocab'],)


def test_0015_batch_decode_step(ns):
    """batch_decode_step appends one greedy token per request and flags finishers at steps_left==0."""
    import math
    import numpy as np
    make_model = ns["make_model"]
    new_block_manager = ns["new_block_manager"]
    allocate = ns["allocate"]
    prefill_request = ns["prefill_request"]
    batch_decode_step = ns["batch_decode_step"]
    greedy_generate = ns["greedy_generate"]

    model = make_model(vocab=16, d=8, seed=0)
    d = model['d']
    scale = 1.0 / math.sqrt(d)
    block_size = 4
    num_blocks = 10
    prompt = [2, 5, 1]
    max_new = 3

    mgr = new_block_manager(num_blocks, block_size)
    bt = allocate(mgr, len(prompt) + max_new)
    k_pool = np.zeros((num_blocks, block_size, d))
    v_pool = np.zeros((num_blocks, block_size, d))
    req = prefill_request(model, 'r0', prompt, bt, k_pool, v_pool, block_size, scale, max_new)
    running = [req]

    # step 1 and 2: nobody finishes
    fin = batch_decode_step(model, running, k_pool, v_pool, block_size, scale)
    assert fin == []
    assert req['steps_left'] == 2 and len(req['generated']) == 1
    fin = batch_decode_step(model, running, k_pool, v_pool, block_size, scale)
    assert fin == []
    assert req['steps_left'] == 1 and len(req['generated']) == 2
    # final step: request finishes and is returned
    fin = batch_decode_step(model, running, k_pool, v_pool, block_size, scale)
    assert fin == [req]
    assert req['steps_left'] == 0

    # the produced tokens must equal the canonical greedy decode
    assert req['generated'] == greedy_generate(model, prompt, max_new, scale)


def test_0016_run_batched(ns):
    """run_batched output for every request equals an independent greedy_generate decode."""
    import math
    make_model = ns["make_model"]
    run_batched = ns["run_batched"]
    greedy_generate = ns["greedy_generate"]

    model = make_model(vocab=16, d=8, seed=0)
    scale = 1.0 / math.sqrt(model['d'])
    block_size = 4
    max_new = 5
    requests = [('a', [1, 2, 3]), ('b', [5]), ('c', [0, 7, 7, 3, 9])]

    # roomy pool: all admitted at once
    res = run_batched(model, requests, num_blocks=20, block_size=block_size,
                      scale=scale, max_new=max_new)
    assert set(res.keys()) == {'a', 'b', 'c'}
    for rid, prompt in requests:
        assert res[rid] == greedy_generate(model, prompt, max_new, scale)

    # tight pool: forces serialized admission + block reuse via free()
    res2 = run_batched(model, requests, num_blocks=2, block_size=block_size,
                       scale=scale, max_new=2)
    assert set(res2.keys()) == {'a', 'b', 'c'}
    for rid, prompt in requests:
        assert res2[rid] == greedy_generate(model, prompt, 2, scale)

import numpy as np


def test_0017_radix_insert(ns):
    radix_insert = ns["radix_insert"]
    root = {'children': {}}
    radix_insert(root, [1, 2, 3])
    # path 1->2->3 exists
    assert 1 in root['children']
    n1 = root['children'][1]
    assert 2 in n1['children']
    n2 = n1['children'][2]
    assert 3 in n2['children']
    n3 = n2['children'][3]
    assert n3['children'] == {}
    # insert overlapping path shares prefix, branches at last token
    radix_insert(root, [1, 2, 4])
    assert set(n2['children'].keys()) == {3, 4}
    # re-inserting an existing path adds nothing new
    radix_insert(root, [1, 2, 3])
    assert set(n2['children'].keys()) == {3, 4}
    assert set(root['children'].keys()) == {1}


def test_0018_radix_longest_prefix(ns):
    radix_insert = ns["radix_insert"]
    radix_longest_prefix = ns["radix_longest_prefix"]
    root = {'children': {}}
    radix_insert(root, [1, 2, 3])
    radix_insert(root, [1, 2, 4])
    # query [1,2,5] -> 2 (1,2 cached, 5 not)
    assert radix_longest_prefix(root, [1, 2, 5]) == 2
    # query [1,2,3,9] -> 3 (1,2,3 cached, 9 not)
    assert radix_longest_prefix(root, [1, 2, 3, 9]) == 3
    # query [7] -> 0 (first token not a child)
    assert radix_longest_prefix(root, [7]) == 0
    # empty query -> 0
    assert radix_longest_prefix(root, []) == 0
    # full match of an inserted path
    assert radix_longest_prefix(root, [1, 2, 4]) == 3


def test_0019_greedy_sample(ns):
    greedy_sample = ns["greedy_sample"]
    rng = np.random.default_rng(0)
    for _ in range(10):
        logits = rng.standard_normal(16)
        assert greedy_sample(logits) == int(np.argmax(logits))
        assert isinstance(greedy_sample(logits), int)
    # tie: argmax returns first occurrence
    logits = np.array([1.0, 3.0, 3.0, 0.0])
    assert greedy_sample(logits) == 1


def test_0020_sample_token(ns):
    sample_token = ns["sample_token"]
    greedy_sample = ns["greedy_sample"]
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(16)
    am = int(np.argmax(logits))

    # temperature <= 0 -> argmax
    assert sample_token(logits, 0.0, 0, 1.0, np.random.default_rng(1)) == am
    assert sample_token(logits, -1.0, 0, 1.0, np.random.default_rng(1)) == am

    # top_k == 1 forces argmax (only one nonzero prob)
    for _ in range(5):
        assert sample_token(logits, 1.0, 1, 1.0, np.random.default_rng(7)) == am

    # very small temperature concentrates mass on argmax -> argmax with prob ~1
    assert sample_token(logits, 1e-6, 0, 1.0, np.random.default_rng(3)) == am

    # determinism: same seed -> same draw
    a = sample_token(logits, 1.0, 0, 1.0, np.random.default_rng(42))
    b = sample_token(logits, 1.0, 0, 1.0, np.random.default_rng(42))
    assert a == b

    # probs match a manual stable softmax draw (temperature=1, no filtering)
    z = logits / 1.0
    e = np.exp(z - np.max(z))
    probs = e / np.sum(e)
    expected = int(np.random.default_rng(123).choice(len(logits), p=probs))
    got = sample_token(logits, 1.0, 0, 1.0, np.random.default_rng(123))
    assert got == expected

    # top_p nucleus: with a dominant logit, top_p<1 keeps just the top token
    sharp = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
    # softmax of sharp puts >0.5 on index 0, so top_p=0.5 nucleus = {0}
    assert sample_token(sharp, 1.0, 0, 0.5, np.random.default_rng(9)) == 0

    # return type is python int
    assert isinstance(sample_token(logits, 1.0, 0, 1.0, np.random.default_rng(0)), int)
