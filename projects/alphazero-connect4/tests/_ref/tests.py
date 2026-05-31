"""Hidden step tests for alphazero-connect4.

Each ``test_<id>_<name>(ns)`` grades one step; ``ns`` holds the reference
implementations with the user's function swapped in. Engine/MCTS steps are graded
by value/behavior; network and loss steps by shape/value.
"""
import numpy as np
import torch
from contextlib import contextmanager


@contextmanager
def step(label):
    try:
        yield
    except AssertionError as e:
        raise AssertionError(f"step {label!r}: {e}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def expect_eq(got, want):
    if got != want:
        raise AssertionError(f"got {got!r} but expected {want!r}")


def expect_true(cond, msg="expected True"):
    if not cond:
        raise AssertionError(msg)


def expect_shape(got, shape):
    g = tuple(got.shape)
    if g != tuple(shape):
        raise AssertionError(f"shape mismatch: got {g} vs want {tuple(shape)}")


def expect_allclose(got, want, atol=1e-5, rtol=1e-4):
    g, w = _np(got).astype(float), _np(want).astype(float)
    if g.shape != w.shape:
        raise AssertionError(f"shape mismatch: got {g.shape} vs want {w.shape}")
    if np.allclose(g, w, atol=atol, rtol=rtol, equal_nan=True):
        return
    d = np.abs(g - w)
    idx = np.unravel_index(int(np.nanargmax(d)), d.shape) if d.size else ()
    raise AssertionError(f"values differ: max abs diff={np.nanmax(d):.3e} at {idx}; "
                         f"got={g[idx]:.6g} want={w[idx]:.6g}")


def _board(rows):
    return np.array(rows, dtype=int)


# --------------------- Part 1 — Connect-4 Game Engine ---------------------

def test_0001_make_empty_board(ns):
    b = ns["make_empty_board"]()
    with step("6x7 of zeros"):
        expect_shape(b, (6, 7))
        expect_allclose(b, np.zeros((6, 7)))


def test_0002_column_top_row(ns):
    b = ns["make_empty_board"]()
    with step("empty column lands at the bottom row (5)"):
        expect_eq(ns["column_top_row"](b, 3), 5)
    b[5, 3] = 1
    with step("next piece stacks above"):
        expect_eq(ns["column_top_row"](b, 3), 4)
    b[:, 2] = 1
    with step("full column -> -1"):
        expect_eq(ns["column_top_row"](b, 2), -1)


def test_0003_column_full(ns):
    b = ns["make_empty_board"]()
    with step("empty not full; filled column is full"):
        expect_true(not ns["column_full"](b, 0))
        b[:, 0] = 1
        expect_true(ns["column_full"](b, 0))


def test_0004_valid_moves(ns):
    b = ns["make_empty_board"]()
    b[:, 1] = 1  # fill column 1
    with step("all columns except the full one"):
        expect_eq(list(ns["valid_moves"](b)), [0, 2, 3, 4, 5, 6])


def test_0005_drop_piece(ns):
    b = ns["make_empty_board"]()
    nb = ns["drop_piece"](b, 3, 1)
    with step("lands at the bottom; original unchanged"):
        expect_eq(int(nb[5, 3]), 1)
        expect_allclose(b, np.zeros((6, 7)))
        nb2 = ns["drop_piece"](nb, 3, -1)
        expect_eq(int(nb2[4, 3]), -1)


def test_0006_four_in_a_row_horizontal(ns):
    b = ns["make_empty_board"]()
    b[5, 1:5] = 1
    with step("detects a horizontal four"):
        expect_true(ns["four_in_a_row_horizontal"](b, 1))
        expect_true(not ns["four_in_a_row_horizontal"](b, -1))


def test_0007_four_in_a_row_vertical(ns):
    b = ns["make_empty_board"]()
    b[2:6, 0] = 1
    with step("detects a vertical four"):
        expect_true(ns["four_in_a_row_vertical"](b, 1))


def test_0008_four_in_a_row_diagonal_down_right(ns):
    b = ns["make_empty_board"]()
    for i in range(4):
        b[i, i] = 1
    with step("detects a down-right diagonal"):
        expect_true(ns["four_in_a_row_diagonal_down_right"](b, 1))


def test_0009_four_in_a_row_diagonal_up_right(ns):
    b = ns["make_empty_board"]()
    for i in range(4):
        b[5 - i, i] = 1
    with step("detects an up-right diagonal"):
        expect_true(ns["four_in_a_row_diagonal_up_right"](b, 1))


def test_0010_check_winner(ns):
    b = ns["make_empty_board"]()
    with step("no winner on empty board"):
        expect_eq(ns["check_winner"](b), 0)
    b[5, 0:4] = -1
    with step("returns the winning player"):
        expect_eq(ns["check_winner"](b), -1)


def test_0011_board_is_full(ns):
    with step("full vs not"):
        expect_true(ns["board_is_full"](np.ones((6, 7), dtype=int)))
        expect_true(not ns["board_is_full"](ns["make_empty_board"]()))


def test_0012_is_terminal(ns):
    b = ns["make_empty_board"]()
    with step("empty not terminal; a win is terminal"):
        expect_true(not ns["is_terminal"](b))
        b[5, 0:4] = 1
        expect_true(ns["is_terminal"](b))


def test_0013_other_player(ns):
    with step("toggles"):
        expect_eq(ns["other_player"](1), -1)
        expect_eq(ns["other_player"](-1), 1)


def test_0014_step_env(ns):
    b = ns["make_empty_board"]()
    b[5, 0:3] = 1  # three in a row; dropping col 3 wins
    nb, reward, done = ns["step_env"](b, 3, 1)
    with step("winning move -> reward 1, done"):
        expect_allclose(reward, 1.0)
        expect_true(done)
    nb2, r2, d2 = ns["step_env"](ns["make_empty_board"](), 0, 1)
    with step("ordinary move -> reward 0, not done"):
        expect_allclose(r2, 0.0)
        expect_true(not d2)


# --------------------- Part 2 — Encoding and Policy-Value Network ---------------------

def test_0015_encode_board(ns):
    b = ns["make_empty_board"]()
    b[5, 0] = 1
    b[5, 1] = -1
    enc = ns["encode_board"](b, 1)
    with step("2 planes (mine, opponent) from player's view"):
        expect_shape(enc, (2, 6, 7))
        expect_allclose(enc[0, 5, 0], 1.0)  # my piece
        expect_allclose(enc[1, 5, 1], 1.0)  # opponent piece
        expect_allclose(enc[0, 5, 1], 0.0)


def test_0016_board_to_torch_tensor(ns):
    enc = ns["encode_board"](ns["make_empty_board"](), 1)
    t = ns["board_to_torch_tensor"](enc)
    with step("batched float tensor (1, C, 6, 7)"):
        expect_shape(t, (1, 2, 6, 7))
        expect_true(t.dtype == torch.float32, "should be float")


def test_0017_init_conv_backbone(ns):
    bb = ns["init_conv_backbone"](2, 8)
    out = bb(torch.zeros(1, 2, 6, 7))
    with step("preserves spatial size, outputs hidden channels"):
        expect_shape(out, (1, 8, 6, 7))


def test_0018_init_policy_head(ns):
    head = ns["init_policy_head"](8 * 6 * 7, 7)
    with step("maps flattened features to 7 logits"):
        expect_shape(head(torch.zeros(1, 8 * 6 * 7)), (1, 7))


def test_0019_init_value_head(ns):
    head = ns["init_value_head"](8 * 6 * 7)
    out = head(torch.randn(4, 8 * 6 * 7))
    with step("scalar value in [-1, 1]"):
        expect_shape(out, (4, 1))
        expect_true(bool((out.abs() <= 1.0 + 1e-5).all()), "tanh range")


def test_0020_build_policy_value_net(ns):
    net = ns["build_policy_value_net"](2, 16, 7)
    logits, value = net(torch.zeros(1, 2, 6, 7))
    with step("forward returns (logits (B,7), value (B,))"):
        expect_shape(logits, (1, 7))
        expect_shape(value, (1,))


def test_0021_policy_value_forward(ns):
    net = ns["build_policy_value_net"](2, 16, 7)
    x = ns["board_to_torch_tensor"](ns["encode_board"](ns["make_empty_board"](), 1))
    logits, value = ns["policy_value_forward"](net, x)
    with step("(logits, value) with value in range"):
        expect_shape(logits, (1, 7))
        expect_true(bool((value.abs() <= 1.0 + 1e-5).all()), "value in [-1,1]")


# --------------------- Part 3 — Action Masking and Sampling ---------------------

def test_0022_action_mask(ns):
    b = ns["make_empty_board"]()
    b[:, 2] = 1  # column 2 full
    m = ns["action_mask"](b)
    with step("1 for legal columns, 0 for full"):
        expect_shape(m, (7,))
        expect_allclose(m, [1, 1, 0, 1, 1, 1, 1])


def test_0023_masked_policy_logits(ns):
    logits = torch.zeros(7)
    mask = torch.tensor([1.0, 0, 1, 1, 1, 1, 1])
    out = ns["masked_policy_logits"](logits, mask)
    with step("illegal -> -inf"):
        expect_true(torch.isinf(out[1]) and out[1] < 0, "masked col should be -inf")
        expect_allclose(out[0], 0.0)


def test_0024_masked_log_softmax(ns):
    logits = torch.zeros(7)
    mask = torch.tensor([1.0, 0, 1, 0, 0, 0, 0])  # only cols 0,2 legal
    lp = ns["masked_log_softmax"](logits, mask)
    with step("normalizes over legal actions; illegal -> -inf"):
        expect_true(torch.isinf(lp[1]), "illegal -inf")
        expect_allclose(torch.exp(lp[torch.tensor([0, 2])]).sum(), 1.0)


def test_0025_sample_action_from_policy(ns):
    logits = torch.tensor([0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # peak on col1
    mask = torch.tensor([1.0, 0, 1, 1, 1, 1, 1])  # but col1 illegal
    g = torch.Generator().manual_seed(0)
    with step("never samples an illegal column"):
        for _ in range(20):
            a = ns["sample_action_from_policy"](logits, mask, g)
            expect_true(a != 1, "sampled an illegal column")


def test_0026_greedy_action_from_policy(ns):
    logits = torch.tensor([0.0, 5.0, 9.0, 0.0, 0.0, 0.0, 0.0])  # col2 best, but illegal
    mask = torch.tensor([1.0, 1, 0, 1, 1, 1, 1])
    with step("picks the best legal column"):
        expect_eq(ns["greedy_action_from_policy"](logits, mask), 1)


# --------------------- Part 4 — PUCT Monte Carlo Tree Search ---------------------

def test_0027_make_mcts_node(ns):
    node = ns["make_mcts_node"](ns["make_empty_board"](), 1, 0.3)
    with step("fresh node fields"):
        expect_eq(node["N"], 0)
        expect_allclose(node["W"], 0.0)
        expect_allclose(node["prior"], 0.3)
        expect_eq(node["expanded"], False)


def test_0028_node_q_value(ns):
    node = ns["make_mcts_node"](None, 1, 0.0)
    with step("0 when unvisited; W/N otherwise"):
        expect_allclose(ns["node_q_value"](node), 0.0)
        node["N"], node["W"] = 4, 2.0
        expect_allclose(ns["node_q_value"](node), 0.5)


def test_0029_ucb_score(ns):
    child = ns["make_mcts_node"](None, 1, 0.5)
    with step("pure exploration when unvisited"):
        expect_allclose(ns["ucb_score"](4, child, 1.0), 1.0 * 0.5 * (4 ** 0.5) / 1)
    child["N"], child["W"] = 2, 2.0  # Q = 1
    with step("subtracts the child's Q (opponent perspective)"):
        expect_allclose(ns["ucb_score"](4, child, 1.0), -1.0 + 1.0 * 0.5 * (4 ** 0.5) / 3)


def test_0030_select_best_child(ns):
    node = ns["make_mcts_node"](None, 1, 0.0)
    node["N"] = 4
    node["children"] = {0: ns["make_mcts_node"](None, -1, 0.1),
                        3: ns["make_mcts_node"](None, -1, 0.9)}
    a, child = ns["select_best_child"](node, 1.0)
    with step("higher-prior unvisited child wins"):
        expect_eq(a, 3)


def test_0031_select_leaf(ns):
    root = ns["make_mcts_node"](ns["make_empty_board"](), 1, 0.0)
    with step("unexpanded root is its own leaf"):
        leaf, path = ns["select_leaf"](root, 1.0)
        expect_true(leaf is root and path == [root], "leaf should be root")
    # expand root one level, then it should descend
    root["children"] = {2: ns["make_mcts_node"](ns["drop_piece"](root["board"], 2, 1), -1, 1.0)}
    root["expanded"] = True
    root["N"] = 1
    leaf, path = ns["select_leaf"](root, 1.0)
    with step("descends into the (unexpanded) child"):
        expect_eq(len(path), 2)
        expect_true(leaf is root["children"][2], "leaf is the child")


def test_0032_evaluate_with_network(ns):
    torch.manual_seed(0)
    net = ns["build_policy_value_net"](2, 16, 7)
    priors, value = ns["evaluate_with_network"](net, ns["make_empty_board"](), 1)
    with step("priors over legal moves sum to 1; value in [-1,1]"):
        expect_allclose(float(np.sum(priors)), 1.0, atol=1e-4)
        expect_true(-1.0 <= value <= 1.0, "value range")


def test_0033_expand_node(ns):
    node = ns["make_mcts_node"](ns["make_empty_board"](), 1, 0.0)
    ns["expand_node"](node, np.full(7, 1 / 7))
    with step("a child per legal move; expanded flag set"):
        expect_eq(len(node["children"]), 7)
        expect_true(node["expanded"], "should be expanded")
        # each child has one of our pieces placed and the opponent to move
        expect_eq(node["children"][0]["player"], -1)


def test_0034_backup_value(ns):
    a = ns["make_mcts_node"](None, 1, 0.0)
    b = ns["make_mcts_node"](None, -1, 0.0)
    ns["backup_value"]([a, b], 1.0)
    with step("leaf gets +value, parent gets -value, both visited once"):
        expect_eq((a["N"], b["N"]), (1, 1))
        expect_allclose(b["W"], 1.0)
        expect_allclose(a["W"], -1.0)


def test_0035_run_one_simulation(ns):
    torch.manual_seed(0)
    net = ns["build_policy_value_net"](2, 16, 7)
    root = ns["make_mcts_node"](ns["make_empty_board"](), 1, 0.0)
    ns["run_one_simulation"](root, net, 1.5)
    with step("root gets visited and expanded"):
        expect_eq(root["N"], 1)
        expect_true(root["expanded"], "root expanded after a sim")


def test_0036_run_mcts(ns):
    torch.manual_seed(0)
    net = ns["build_policy_value_net"](2, 16, 7)
    root = ns["make_mcts_node"](ns["make_empty_board"](), 1, 0.0)
    ns["run_mcts"](root, net, 30, 1.5)
    with step("root visit count equals the number of simulations"):
        expect_eq(root["N"], 30)
    # tactical: player 1 has three in a row on the bottom; col 3 wins.
    b = ns["make_empty_board"]()
    b[5, 0:3] = 1
    win_root = ns["make_mcts_node"](b, 1, 0.0)
    ns["run_mcts"](win_root, net, 120, 1.5)
    with step("MCTS finds the immediate winning move (col 3)"):
        expect_eq(ns["mcts_choose_action"](win_root, 0), 3)


def test_0037_visit_count_policy(ns):
    root = ns["make_mcts_node"](None, 1, 0.0)
    root["children"] = {0: ns["make_mcts_node"](None, -1, 0.0),
                        3: ns["make_mcts_node"](None, -1, 0.0)}
    root["children"][0]["N"] = 3
    root["children"][3]["N"] = 1
    with step("temperature 1 normalizes counts; temperature 0 is one-hot"):
        expect_allclose(ns["visit_count_policy"](root, 1.0)[[0, 3]], [0.75, 0.25])
        expect_allclose(ns["visit_count_policy"](root, 0)[0], 1.0)


def test_0038_mcts_choose_action(ns):
    root = ns["make_mcts_node"](None, 1, 0.0)
    root["children"] = {2: ns["make_mcts_node"](None, -1, 0.0),
                        5: ns["make_mcts_node"](None, -1, 0.0)}
    root["children"][5]["N"] = 10
    root["children"][2]["N"] = 1
    with step("temperature 0 picks the most-visited action"):
        expect_eq(ns["mcts_choose_action"](root, 0), 5)


# --------------------- Part 5 — Self-Play Data Generation ---------------------

def test_0039_record_self_play_step(ns):
    buf = []
    ns["record_self_play_step"](buf, "state", np.ones(7) / 7, 1)
    with step("appends a (state, policy, player) tuple"):
        expect_eq(len(buf), 1)
        expect_eq(buf[0][2], 1)


def test_0040_play_self_play_game(ns):
    torch.manual_seed(0)
    net = ns["build_policy_value_net"](2, 16, 7)
    records, winner = ns["play_self_play_game"](net, 5, 1.5, 1.0, np.random.default_rng(0))
    with step("records states/policies; winner is valid"):
        expect_true(len(records) > 0, "should record moves")
        expect_shape(records[0][0], (2, 6, 7))
        expect_allclose(float(np.sum(records[0][1])), 1.0, atol=1e-5)
        expect_true(winner in (1, -1, 0), "valid winner")


def test_0041_assign_value_targets(ns):
    records = [("s0", np.ones(7) / 7, 1), ("s1", np.ones(7) / 7, -1)]
    out = ns["assign_value_targets"](records, winner=1)
    with step("value is +1 for the winner, -1 for the loser"):
        expect_allclose(out[0][2], 1.0)
        expect_allclose(out[1][2], -1.0)


def test_0042_generate_self_play_batch(ns):
    torch.manual_seed(0)
    net = ns["build_policy_value_net"](2, 16, 7)
    data = ns["generate_self_play_batch"](net, 2, 5, 1.5, 1.0, np.random.default_rng(0))
    with step("returns (state, policy, value) tuples with valid values"):
        expect_true(len(data) > 0, "should produce data")
        expect_true(all(d[2] in (1.0, -1.0, 0.0) for d in data), "values in {-1,0,1}")


# --------------------- Part 6 — Losses and Training Loop ---------------------

def test_0043_value_loss_mse(ns):
    import torch.nn.functional as F
    pred = torch.tensor([0.5, -0.5])
    targ = torch.tensor([1.0, -1.0])
    with step("MSE"):
        expect_allclose(ns["value_loss_mse"](pred, targ), F.mse_loss(pred, targ))


def test_0044_policy_loss_cross_entropy(ns):
    import torch.nn.functional as F
    logits = torch.randn(3, 7)
    target = torch.softmax(torch.randn(3, 7), dim=-1)
    want = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    with step("cross-entropy against the target distribution"):
        expect_allclose(ns["policy_loss_cross_entropy"](logits, target), want)


def test_0045_l2_regularization_loss(ns):
    net = ns["build_policy_value_net"](2, 8, 7)
    want = 1e-4 * sum((p ** 2).sum() for p in net.parameters())
    with step("weight_decay * sum of squared params"):
        expect_allclose(ns["l2_regularization_loss"](net, 1e-4), want)


def test_0046_combined_loss(ns):
    with step("sum of the three terms"):
        expect_allclose(ns["combined_loss"](torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.5)), 3.5)


def test_0047_encode_batch_states(ns):
    states = [np.zeros((2, 6, 7), dtype=np.float32) for _ in range(4)]
    with step("stacks to (B,2,6,7) float tensor"):
        out = ns["encode_batch_states"](states)
        expect_shape(out, (4, 2, 6, 7))
        expect_true(out.dtype == torch.float32, "float tensor")


def test_0048_iterate_minibatches(ns):
    with step("batches of the given size"):
        expect_eq([len(b) for b in ns["iterate_minibatches"](list(range(10)), 4)], [4, 4, 2])


def _fake_dataset(ns, n=16, seed=0):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        board = ns["make_empty_board"]()
        for c in rng.choice(7, size=3, replace=False):
            board = ns["drop_piece"](board, int(c), 1)
        pi = np.asarray(torch.softmax(torch.randn(7), dim=-1).numpy(), dtype=np.float32)
        data.append((ns["encode_board"](board, 1), pi, float(rng.uniform(-1, 1))))
    return data


def test_0049_training_step(ns):
    torch.manual_seed(0)
    net = ns["build_policy_value_net"](2, 16, 7)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss = ns["training_step"](net, _fake_dataset(ns, 8), opt, 1e-4)
    with step("returns a finite scalar loss"):
        expect_true(np.isfinite(loss), "loss finite")


def test_0050_training_epoch(ns):
    torch.manual_seed(0)
    net = ns["build_policy_value_net"](2, 16, 7)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    data = _fake_dataset(ns, 16)
    first = ns["training_epoch"](net, data, opt, 8, 1e-5)
    last = first
    for _ in range(25):
        last = ns["training_epoch"](net, data, opt, 8, 1e-5)
    with step("loss decreases as the net fits the buffer"):
        expect_true(last < first, f"loss should drop: first={first:.3f} last={last:.3f}")


# --------------------- Part 7 — Iterated Training Loop ---------------------

def test_0051_self_play_iteration(ns):
    torch.manual_seed(0)
    net = ns["build_policy_value_net"](2, 16, 7)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss, data = ns["self_play_iteration"](net, opt, 1, 5, 1.5, 1.0, 8, 1e-4, np.random.default_rng(0))
    with step("generates data and trains; returns (loss, data)"):
        expect_true(np.isfinite(loss), "finite loss")
        expect_true(len(data) > 0, "produced self-play data")


def test_0052_train_loop(ns):
    torch.manual_seed(0)
    net = ns["build_policy_value_net"](2, 16, 7)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    out_net, losses = ns["train_loop"](net, opt, 2, 1, 5, 1.5, 1.0, 8, 1e-4, np.random.default_rng(0))
    with step("runs the requested iterations"):
        expect_eq(len(losses), 2)


# --------------------- Part 8 — Agents and Evaluation ---------------------

def test_0053_random_policy_action(ns):
    b = ns["make_empty_board"]()
    b[:, 0] = 1  # column 0 full
    rng = np.random.default_rng(0)
    with step("returns a legal column"):
        for _ in range(20):
            expect_true(ns["random_policy_action"](b, rng) in range(1, 7), "must be legal")


def test_0054_greedy_agent_action(ns):
    torch.manual_seed(0)
    net = ns["build_policy_value_net"](2, 16, 7)
    b = ns["make_empty_board"]()
    b[:, 3] = 1  # column 3 full
    with step("returns a legal column"):
        a = ns["greedy_agent_action"](net, b, 1)
        expect_true(a in [0, 1, 2, 4, 5, 6], "should be legal")


def test_0055_play_one_match(ns):
    rng = np.random.default_rng(0)
    winner = ns["play_one_match"](lambda b, p: ns["random_policy_action"](b, rng),
                                  lambda b, p: ns["random_policy_action"](b, rng), rng)
    with step("returns a valid winner"):
        expect_true(winner in (1, -1, 0), "valid winner")


def test_0056_match_win_rate(ns):
    with step("fraction won by perspective"):
        expect_allclose(ns["match_win_rate"]([1, 1, -1, 0], 1), 0.5)


def test_0057_evaluate_against_random(ns):
    torch.manual_seed(0)
    net = ns["build_policy_value_net"](2, 16, 7)
    rate = ns["evaluate_against_random"](net, 6, np.random.default_rng(0))
    with step("returns a win rate in [0,1]"):
        expect_true(0.0 <= rate <= 1.0, f"rate out of range: {rate}")
