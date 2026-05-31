"""Hidden step tests for tic-tac-toe-rl.

Each ``test_<id>_<name>(ns)`` grades one step; ``ns`` holds the reference
implementations with the user's function swapped in. Assertion messages carry
markers ("shape mismatch", "values differ", "gradient check failed") that the
runner's likely-cause heuristic keys on.
"""
import numpy as np
from contextlib import contextmanager


# --------------------------- harness ---------------------------

@contextmanager
def step(label):
    try:
        yield
    except AssertionError as e:
        raise AssertionError(f"step {label!r}: {e}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e


def expect_eq(got, want):
    if got != want:
        raise AssertionError(f"got {got!r} but expected {want!r}")


def expect_true(cond, msg="expected True"):
    if not cond:
        raise AssertionError(msg)


def expect_shape(got, shape):
    got = np.asarray(got)
    if tuple(got.shape) != tuple(shape):
        raise AssertionError(f"shape mismatch: got {tuple(got.shape)} vs want {tuple(shape)}")


def expect_allclose(got, want, atol=1e-6, rtol=1e-5):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    if got.shape != want.shape:
        raise AssertionError(f"shape mismatch: got {got.shape} vs want {want.shape}")
    if np.allclose(got, want, atol=atol, rtol=rtol, equal_nan=True):
        return
    d = np.abs(got - want)
    idx = np.unravel_index(int(np.nanargmax(d)), d.shape) if d.size else ()
    raise AssertionError(f"values differ: max abs diff={np.nanmax(d):.3e} at {idx}; "
                         f"got={got[idx]:.6g} want={want[idx]:.6g}")


def grad_check(f, x, dout, analytic_dx, eps=1e-5, atol=2e-4, name="dx"):
    """Finite-difference check: numeric d/dx of (f(x)*dout).sum() vs analytic_dx."""
    x = np.asarray(x, dtype=float)
    analytic_dx = np.asarray(analytic_dx, dtype=float)
    if analytic_dx.shape != x.shape:
        raise AssertionError(f"{name} shape mismatch: got {analytic_dx.shape} vs want {x.shape}")
    num = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = np.sum(np.asarray(f(x)) * dout)
        x[i] = old - eps
        fm = np.sum(np.asarray(f(x)) * dout)
        x[i] = old
        num[i] = (fp - fm) / (2 * eps)
        it.iternext()
    d = np.abs(num - analytic_dx)
    denom = np.maximum(1.0, np.abs(num) + np.abs(analytic_dx))
    if np.max(d / denom) > atol:
        idx = np.unravel_index(int(np.argmax(d / denom)), d.shape)
        raise AssertionError(f"gradient check failed for {name}: max rel diff="
                             f"{np.max(d / denom):.3e} at {idx}; "
                             f"analytic={analytic_dx[idx]:.6g} numeric={num[idx]:.6g}")


def _board(rows):
    return np.array(rows, dtype=int)


# --------------------- Part 1 — Board & Game Engine ---------------------

def test_0001_create_empty_board(ns):
    b = ns["create_empty_board"]()
    with step("3x3 of zeros"):
        expect_shape(b, (3, 3))
        expect_allclose(b, np.zeros((3, 3)))


def test_0002_encode_player(ns):
    with step("X -> +1, O -> -1"):
        expect_eq(ns["encode_player"]("X"), 1)
        expect_eq(ns["encode_player"]("O"), -1)


def test_0003_print_board(ns):
    b = _board([[1, 0, -1], [0, 1, 0], [-1, 0, 0]])
    s = ns["print_board"](b)
    with step("renders X/O/."):
        expect_true("X" in s and "O" in s and "." in s, "should use X/O/.")
        expect_eq(s.split("\n")[0].replace(" ", ""), "X.O")


def test_0004_is_cell_empty(ns):
    b = _board([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    with step("empty vs occupied"):
        expect_true(not ns["is_cell_empty"](b, 0), "cell 0 is occupied")
        expect_true(ns["is_cell_empty"](b, 1), "cell 1 is empty")


def test_0005_place_move(ns):
    b = ns["create_empty_board"]()
    nb = ns["place_move"](b, 4, 1)
    with step("places mark at action 4, no mutation of input"):
        expect_eq(int(nb[1, 1]), 1)
        expect_allclose(b, np.zeros((3, 3)))  # original unchanged


def test_0006_get_legal_moves(ns):
    b = _board([[1, -1, 0], [0, 0, 0], [0, 0, 0]])
    with step("empty action indices"):
        expect_eq(list(ns["get_legal_moves"](b)), [2, 3, 4, 5, 6, 7, 8])


def test_0007_check_row_win(ns):
    with step("full row of X wins; mixed does not"):
        expect_true(ns["check_row_win"](_board([[1, 1, 1], [0, 0, 0], [0, 0, 0]]), 1))
        expect_true(not ns["check_row_win"](_board([[1, 1, -1], [0, 0, 0], [0, 0, 0]]), 1))


def test_0008_check_column_win(ns):
    with step("full column wins"):
        expect_true(ns["check_column_win"](_board([[1, 0, 0], [1, 0, 0], [1, 0, 0]]), 1))
        expect_true(not ns["check_column_win"](ns["create_empty_board"](), 1))


def test_0009_check_main_diagonal_win(ns):
    with step("main diagonal"):
        expect_true(ns["check_main_diagonal_win"](_board([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]), -1))
        expect_true(not ns["check_main_diagonal_win"](ns["create_empty_board"](), 1))


def test_0010_check_anti_diagonal_win(ns):
    with step("anti diagonal"):
        expect_true(ns["check_anti_diagonal_win"](_board([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), 1))
        expect_true(not ns["check_anti_diagonal_win"](ns["create_empty_board"](), 1))


def test_0011_is_winner(ns):
    with step("any line wins"):
        expect_true(ns["is_winner"](_board([[1, 1, 1], [0, 0, 0], [0, 0, 0]]), 1))
        expect_true(not ns["is_winner"](_board([[1, -1, 1], [0, 0, 0], [0, 0, 0]]), 1))


def test_0012_is_draw(ns):
    full = _board([[1, -1, 1], [1, -1, -1], [-1, 1, 1]])  # full, no 3-in-a-row
    with step("full + no winner -> draw"):
        expect_true(ns["is_draw"](full))
        expect_true(not ns["is_draw"](ns["create_empty_board"]()))
        expect_true(not ns["is_draw"](_board([[1, 1, 1], [0, 0, 0], [0, 0, 0]])))


def test_0013_get_game_status(ns):
    with step("X win=+1, O win=-1, draw=0, ongoing=None"):
        expect_eq(ns["get_game_status"](_board([[1, 1, 1], [0, 0, 0], [0, 0, 0]])), 1)
        expect_eq(ns["get_game_status"](_board([[-1, -1, -1], [0, 0, 0], [0, 0, 0]])), -1)
        expect_eq(ns["get_game_status"](_board([[1, -1, 1], [1, -1, -1], [-1, 1, 1]])), 0)
        expect_eq(ns["get_game_status"](ns["create_empty_board"]()), None)


def test_0014_get_current_player(ns):
    with step("X first; after one X move it's O's turn"):
        expect_eq(ns["get_current_player"](ns["create_empty_board"]()), 1)
        expect_eq(ns["get_current_player"](_board([[1, 0, 0], [0, 0, 0], [0, 0, 0]])), -1)


def test_0015_switch_player(ns):
    with step("toggles"):
        expect_eq(ns["switch_player"](1), -1)
        expect_eq(ns["switch_player"](-1), 1)


def test_0016_play_hardcoded_game(ns):
    board = ns["play_hardcoded_game"]([0, 3, 1, 4, 2])  # X: 0,1,2 (top row); O: 3,4
    with step("plays the sequence; X wins the top row"):
        expect_true(ns["is_winner"](board, 1), "X should have won")
        expect_eq(ns["get_game_status"](board), 1)


def test_0017_play_interactive_game(ns):
    moves = iter([0, 3, 1, 4, 2])
    status = ns["play_interactive_game"](lambda b, p: next(moves))
    with step("drives play via callback to a terminal status"):
        expect_eq(status, 1)


def test_0018_TicTacToeGame(ns):
    g = ns["TicTacToeGame"]()
    with step("fresh game: 9 legal moves, X to move"):
        expect_eq(len(g.legal_moves()), 9)
        expect_eq(g.current_player, 1)
    with step("stepping plays X then O and reports status"):
        g.step(0)
        expect_eq(g.current_player, -1)
        g.step(3); g.step(1); g.step(4)
        expect_eq(g.step(2), 1)  # X completes the top row
    with step("reset clears the board"):
        g.reset()
        expect_eq(len(g.legal_moves()), 9)
