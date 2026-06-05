"""Hidden reference implementations for mini-vllm-inference-engine. Steps call earlier references
(shared namespace). One def per step with a one-line docstring."""
import math  # noqa: F401

import numpy as np

def make_model(vocab=16, d=8, seed=0):
    """Build a tiny LM as a dict of weight arrays."""
    rng = np.random.default_rng(seed); s = 1.0/np.sqrt(d)
    return {'vocab':vocab, 'd':d,
            'E':  rng.standard_normal((vocab,d))*s,
            'Wq': rng.standard_normal((d,d))*s, 'Wk': rng.standard_normal((d,d))*s,
            'Wv': rng.standard_normal((d,d))*s, 'Wout': rng.standard_normal((d,vocab))*s}


def token_qkv(model, token_id):
    """Embed a token and project it to its query, key, and value vectors."""
    x = model['E'][token_id]
    q = x @ model['Wq']
    k = x @ model['Wk']
    v = x @ model['Wv']
    return q, k, v


def kv_attention(q, K, V, scale):
    """Compute scaled-dot-product attention of one query over cached keys/values."""
    scores = scale * (K @ q)
    scores = scores - np.max(scores)
    p = np.exp(scores)
    p = p / np.sum(p)
    o = p @ V
    return o


def decode_step(model, token_id, K_cache, V_cache, scale):
    """Process one token: append its k/v, attend over the cache, return logits and the new k,v."""
    q, k, v = token_qkv(model, token_id)
    Kf = np.vstack([K_cache, k])
    Vf = np.vstack([V_cache, v])
    o = kv_attention(q, Kf, Vf, scale)
    logits = o @ model['Wout']
    return logits, k, v


def greedy_generate(model, prompt_ids, max_new, scale):
    """Greedily decode max_new tokens with a contiguous KV cache (the canonical ground truth)."""
    d = model['d']
    K_cache = np.zeros((0, d))
    V_cache = np.zeros((0, d))
    cur = None
    for tok in prompt_ids:
        cur, k, v = decode_step(model, tok, K_cache, V_cache, scale)
        K_cache = np.vstack([K_cache, k])
        V_cache = np.vstack([V_cache, v])
    out = []
    for _ in range(max_new):
        tok = int(np.argmax(cur))
        out.append(tok)
        cur, k, v = decode_step(model, tok, K_cache, V_cache, scale)
        K_cache = np.vstack([K_cache, k])
        V_cache = np.vstack([V_cache, v])
    return out

import math


def new_block_manager(num_blocks, block_size):
    """Create a paging manager with a FIFO free-list of physical block ids."""
    return {'free': list(range(num_blocks)), 'block_size': block_size, 'num_blocks': num_blocks}


def allocate(mgr, n_tokens):
    """Pop ceil(n_tokens/block_size) blocks from the front of the free-list, or None if OOM."""
    block_size = mgr['block_size']
    need = 0 if n_tokens == 0 else math.ceil(n_tokens / block_size)
    if len(mgr['free']) < need:
        return None
    table = mgr['free'][:need]
    del mgr['free'][:need]
    return table


def free(mgr, block_table):
    """Return every block id in the table to the back of the free-list in order."""
    for bid in block_table:
        mgr['free'].append(bid)


def slot_for_token(block_table, token_index, block_size):
    """Map a logical token index to its (physical block id, in-block offset)."""
    block_id = block_table[token_index // block_size]
    offset = token_index % block_size
    return block_id, offset

def write_kv(k_pool, v_pool, block_table, token_index, block_size, k, v):
    """Write key/value vectors into the paged pool at the logical token's physical slot."""
    bid, off = slot_for_token(block_table, token_index, block_size)
    k_pool[bid, off] = k
    v_pool[bid, off] = v


def gather_kv(k_pool, v_pool, block_table, num_tokens, block_size):
    """Read the first num_tokens KV slots from the pools into logically-contiguous K,V arrays."""
    d = k_pool.shape[2]
    K = np.zeros((num_tokens, d))
    V = np.zeros((num_tokens, d))
    for t in range(num_tokens):
        bid, off = slot_for_token(block_table, t, block_size)
        K[t] = k_pool[bid, off]
        V[t] = v_pool[bid, off]
    return K, V


def paged_attention(q, k_pool, v_pool, block_table, num_tokens, block_size, scale):
    """Compute attention of q over the paged KV pool, equivalent to contiguous kv_attention."""
    K, V = gather_kv(k_pool, v_pool, block_table, num_tokens, block_size)
    return kv_attention(q, K, V, scale)


def paged_generate(model, prompt_ids, max_new, scale, num_blocks, block_size):
    """Greedy decode storing KV in a paged pool; output matches greedy_generate exactly."""
    d = model['d']
    mgr = new_block_manager(num_blocks, block_size)
    k_pool = np.zeros((num_blocks, block_size, d))
    v_pool = np.zeros((num_blocks, block_size, d))
    bt = allocate(mgr, len(prompt_ids) + max_new)
    num = 0
    cur = None
    for tok in prompt_ids:
        q, k, v = token_qkv(model, tok)
        write_kv(k_pool, v_pool, bt, num, block_size, k, v)
        num += 1
        o = paged_attention(q, k_pool, v_pool, bt, num, block_size, scale)
        cur = o @ model['Wout']
    generated = []
    for _ in range(max_new):
        tok = int(np.argmax(cur))
        generated.append(tok)
        q, k, v = token_qkv(model, tok)
        write_kv(k_pool, v_pool, bt, num, block_size, k, v)
        num += 1
        o = paged_attention(q, k_pool, v_pool, bt, num, block_size, scale)
        cur = o @ model['Wout']
    return generated

def can_admit(mgr, n_tokens):
    """Return True if the manager has enough free blocks to hold n_tokens."""
    need = 0 if n_tokens == 0 else math.ceil(n_tokens / mgr['block_size'])
    return len(mgr['free']) >= need


def prefill_request(model, req_id, prompt_ids, block_table, k_pool, v_pool, block_size, scale, max_new):
    """Process all prompt tokens into the paged pool and return the running-request dict."""
    num_tokens = 0
    cur_logits = None
    for tok in prompt_ids:
        q, k, v = token_qkv(model, tok)
        write_kv(k_pool, v_pool, block_table, num_tokens, block_size, k, v)
        num_tokens += 1
        o = paged_attention(q, k_pool, v_pool, block_table, num_tokens, block_size, scale)
        cur_logits = o @ model['Wout']
    return {'id': req_id, 'block_table': block_table, 'num_tokens': len(prompt_ids),
            'cur_logits': cur_logits, 'generated': [], 'steps_left': max_new}


def batch_decode_step(model, running, k_pool, v_pool, block_size, scale):
    """Advance every running request by one greedy token; return those that just finished."""
    finished = []
    for r in running:
        tok = int(np.argmax(r['cur_logits']))
        r['generated'].append(tok)
        r['steps_left'] -= 1
        q, k, v = token_qkv(model, tok)
        write_kv(k_pool, v_pool, r['block_table'], r['num_tokens'], block_size, k, v)
        r['num_tokens'] += 1
        o = paged_attention(q, k_pool, v_pool, r['block_table'], r['num_tokens'], block_size, scale)
        r['cur_logits'] = o @ model['Wout']
        if r['steps_left'] == 0:
            finished.append(r)
    return finished


def run_batched(model, requests, num_blocks, block_size, scale, max_new):
    """Continuously batch (admit/prefill/decode) all requests; return {req_id: generated ids}."""
    d = model['d']
    mgr = new_block_manager(num_blocks, block_size)
    k_pool = np.zeros((num_blocks, block_size, d))
    v_pool = np.zeros((num_blocks, block_size, d))
    waiting = list(requests)
    running = []
    results = {}
    while waiting or running:
        still_waiting = []
        for req_id, prompt_ids in waiting:
            n = len(prompt_ids) + max_new
            if can_admit(mgr, n):
                bt = allocate(mgr, n)
                running.append(prefill_request(model, req_id, prompt_ids, bt,
                                                k_pool, v_pool, block_size, scale, max_new))
            else:
                still_waiting.append((req_id, prompt_ids))
        waiting = still_waiting
        if running:
            finished = batch_decode_step(model, running, k_pool, v_pool, block_size, scale)
            for r in finished:
                results[r['id']] = r['generated']
                free(mgr, r['block_table'])
            running = [r for r in running if r['steps_left'] > 0]
    return results

def radix_insert(root, token_ids):
    """Insert a token sequence as a path of child nodes under the radix root."""
    node = root
    for t in token_ids:
        t = int(t)
        if t not in node['children']:
            node['children'][t] = {'children': {}}
        node = node['children'][t]


def radix_longest_prefix(root, token_ids):
    """Return the length of the longest leading run of tokens already cached as a path."""
    node = root
    count = 0
    for t in token_ids:
        t = int(t)
        if t in node['children']:
            node = node['children'][t]
            count += 1
        else:
            break
    return count


def greedy_sample(logits):
    """Return the index of the maximum logit."""
    return int(np.argmax(logits))


def sample_token(logits, temperature, top_k, top_p, rng):
    """Sample a token id from logits with temperature, top-k, and nucleus (top-p) filtering."""
    if temperature <= 0:
        return greedy_sample(logits)
    z = np.asarray(logits, dtype=np.float64) / temperature
    if top_k:
        k = int(top_k)
        if k < z.shape[0]:
            thresh = np.sort(z)[-k]
            z = np.where(z >= thresh, z, -np.inf)
    m = np.max(z)
    e = np.exp(z - m)
    probs = e / np.sum(e)
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cum = np.cumsum(sorted_probs)
        keep = cum < top_p
        keep[0] = True
        cutoff = int(np.argmax(cum >= top_p)) + 1
        mask = np.zeros_like(probs, dtype=bool)
        mask[order[:cutoff]] = True
        probs = np.where(mask, probs, 0.0)
        probs = probs / np.sum(probs)
    return int(rng.choice(len(logits), p=probs))
