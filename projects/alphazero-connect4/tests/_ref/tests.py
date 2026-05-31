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
