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


# --------------------- Part 2 — Random and Minimax Baselines ---------------------

def test_0019_random_move_agent(ns):
    b = _board([[1, -1, 0], [0, 0, 0], [0, 0, 0]])
    rng = np.random.default_rng(0)
    with step("returns a legal action"):
        for _ in range(20):
            expect_true(ns["random_move_agent"](b, rng) in [2, 3, 4, 5, 6, 7, 8], "must be legal")


def test_0020_play_random_vs_random_game(ns):
    with step("returns a terminal status"):
        s = ns["play_random_vs_random_game"](np.random.default_rng(1))
        expect_true(s in (1, -1, 0), "status must be +1/-1/0")


def test_0021_play_random_vs_random_matches(ns):
    res = ns["play_random_vs_random_matches"](10, np.random.default_rng(2))
    with step("returns one status per game"):
        expect_eq(len(res), 10)
        expect_true(all(s in (1, -1, 0) for s in res), "all valid statuses")


def test_0022_compute_outcome_rates(ns):
    r = ns["compute_outcome_rates"]([1, 1, -1, 0])
    with step("fractions for x/o/draw"):
        expect_allclose(r["x_win"], 0.5)
        expect_allclose(r["o_win"], 0.25)
        expect_allclose(r["draw"], 0.25)


def test_0023_minimax_terminal_score(ns):
    with step("X win=+1, O win=-1, draw=0"):
        expect_eq(ns["minimax_terminal_score"](_board([[1, 1, 1], [0, 0, 0], [0, 0, 0]])), 1)
        expect_eq(ns["minimax_terminal_score"](_board([[-1, -1, -1], [0, 0, 0], [0, 0, 0]])), -1)
        expect_eq(ns["minimax_terminal_score"](_board([[1, -1, 1], [1, -1, -1], [-1, 1, 1]])), 0)


def test_0024_minimax_recursive(ns):
    # X to move, can win at action 2 (top row). Optimal value = +1.
    b = _board([[1, 1, 0], [-1, -1, 0], [0, 0, 0]])
    with step("optimal value with X to move"):
        expect_eq(ns["minimax_recursive"](b, 1), 1)
    # O to move and must block; with best play it's a draw on this shallow board.
    b2 = _board([[1, 1, -1], [-1, -1, 1], [1, 0, 0]])  # nearly full
    with step("value is in {-1,0,1}"):
        expect_true(ns["minimax_recursive"](b2, -1) in (-1, 0, 1), "value range")


def test_0025_minimax_max_min_step(ns):
    with step("X maximizes, O minimizes"):
        expect_eq(ns["minimax_max_min_step"](1, [-1, 0, 1]), 1)
        expect_eq(ns["minimax_max_min_step"](-1, [-1, 0, 1]), -1)


def test_0026_minimax_best_move(ns):
    # X to move; the winning move is action 2.
    b = _board([[1, 1, 0], [-1, -1, 0], [0, 0, 0]])
    with step("picks the winning action"):
        expect_eq(ns["minimax_best_move"](b, 1), 2)
    # O to move; must block X's top row at action 2.
    b2 = _board([[1, 1, 0], [-1, 0, 0], [0, 0, 0]])
    with step("blocks the opponent's win"):
        expect_eq(ns["minimax_best_move"](b2, -1), 2)


def test_0027_minimax_alpha_beta(ns):
    boards = [
        _board([[1, 1, 0], [-1, -1, 0], [0, 0, 0]]),
        _board([[1, 1, -1], [-1, -1, 1], [1, 0, 0]]),
        _board([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
    ]
    with step("matches minimax_recursive on several boards"):
        for b, p in [(boards[0], 1), (boards[1], -1), (boards[2], 1)]:
            expect_eq(ns["minimax_alpha_beta"](b, p, -np.inf, np.inf), ns["minimax_recursive"](b, p))


def test_0028_play_minimax_vs_random_matches(ns):
    res = ns["play_minimax_vs_random_matches"](6, np.random.default_rng(3))
    with step("optimal X never loses to random O"):
        expect_eq(len(res), 6)
        expect_true(all(s in (1, 0) for s in res), f"X should never lose, got {res}")


def test_0029_play_minimax_vs_minimax_matches(ns):
    res = ns["play_minimax_vs_minimax_matches"](3)
    with step("optimal vs optimal is always a draw"):
        expect_true(all(s == 0 for s in res), f"all draws expected, got {res}")


# --------------------- Part 3 — Tabular Q-Learning Foundations ---------------------

def test_0030_encode_board_state_key(ns):
    b = _board([[1, 0, -1], [0, 1, 0], [0, 0, -1]])
    with step("hashable 9-tuple of cells"):
        expect_eq(ns["encode_board_state_key"](b), (1, 0, -1, 0, 1, 0, 0, 0, -1))


def test_0031_canonical_board_key(ns):
    b = _board([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    rot = np.rot90(b)  # a symmetric variant
    with step("symmetric boards share a canonical key"):
        expect_eq(ns["canonical_board_key"](b), ns["canonical_board_key"](rot))


def test_0032_initialize_q_table(ns):
    with step("empty table"):
        expect_eq(ns["initialize_q_table"](), {})


def test_0033_get_q_value(ns):
    q = ns["initialize_q_table"]()
    with step("defaults to 0 for unseen states"):
        expect_allclose(ns["get_q_value"](q, (0,) * 9, 4), 0.0)


def test_0034_set_q_value(ns):
    q = ns["initialize_q_table"]()
    ns["set_q_value"](q, (0,) * 9, 4, 2.5)
    with step("stores and reads back"):
        expect_allclose(ns["get_q_value"](q, (0,) * 9, 4), 2.5)
        expect_allclose(ns["get_q_value"](q, (0,) * 9, 0), 0.0)  # other actions untouched


def test_0035_choose_learning_rate_alpha(ns):
    with step("in (0, 1]"):
        a = ns["choose_learning_rate_alpha"]()
        expect_true(0 < a <= 1, f"alpha out of range: {a}")


def test_0036_choose_discount_factor_gamma(ns):
    with step("in (0, 1]"):
        g = ns["choose_discount_factor_gamma"]()
        expect_true(0 < g <= 1, f"gamma out of range: {g}")


def test_0037_choose_initial_epsilon(ns):
    with step("in [0, 1]"):
        e = ns["choose_initial_epsilon"]()
        expect_true(0 <= e <= 1, f"epsilon out of range: {e}")


def test_0038_epsilon_decay_schedule(ns):
    with step("decays multiplicatively, floored"):
        expect_allclose(ns["epsilon_decay_schedule"](1.0, 0.99, 0.1), 0.99)
        expect_allclose(ns["epsilon_decay_schedule"](0.1, 0.5, 0.1), 0.1)  # floor holds


def test_0039_epsilon_greedy_explore_move(ns):
    b = _board([[1, -1, 0], [0, 0, 0], [0, 0, 0]])
    with step("returns a legal move"):
        expect_true(ns["epsilon_greedy_explore_move"](b, np.random.default_rng(0)) in [2, 3, 4, 5, 6, 7, 8])


def test_0040_epsilon_greedy_select_action(ns):
    q = ns["initialize_q_table"]()
    b = ns["create_empty_board"]()
    ns["set_q_value"](q, ns["encode_board_state_key"](b), 4, 5.0)  # action 4 is best
    with step("epsilon=0 acts greedily"):
        expect_eq(ns["epsilon_greedy_select_action"](q, b, 0.0, np.random.default_rng(0)), 4)
    with step("epsilon=1 explores a legal move"):
        expect_true(ns["epsilon_greedy_select_action"](q, b, 1.0, np.random.default_rng(0)) in range(9))


def test_0041_greedy_argmax_over_legal_actions(ns):
    q = ns["initialize_q_table"]()
    b = _board([[1, 0, 0], [0, 0, 0], [0, 0, 0]])  # action 0 illegal
    ns["set_q_value"](q, ns["encode_board_state_key"](b), 0, 9.0)  # illegal but high
    ns["set_q_value"](q, ns["encode_board_state_key"](b), 5, 3.0)  # best legal
    with step("ignores illegal actions"):
        expect_eq(ns["greedy_argmax_over_legal_actions"](q, b), 5)


def test_0042_random_tie_break_argmax(ns):
    with step("breaks ties among the maxima"):
        out = {ns["random_tie_break_argmax"]([1, 3, 3, 0], np.random.default_rng(s)) for s in range(20)}
        expect_true(out <= {1, 2} and len(out) >= 1, f"should pick among ties, got {out}")


def test_0043_tic_tac_toe_reward(ns):
    with step("+1 win, -1 loss, 0 draw/none from player's view"):
        expect_allclose(ns["tic_tac_toe_reward"](1, 1), 1.0)
        expect_allclose(ns["tic_tac_toe_reward"](-1, 1), -1.0)
        expect_allclose(ns["tic_tac_toe_reward"](0, 1), 0.0)
        expect_allclose(ns["tic_tac_toe_reward"](1, -1), -1.0)


def test_0044_q_learning_nonterminal_target(ns):
    with step("r + gamma * next_max"):
        expect_allclose(ns["q_learning_nonterminal_target"](0.5, 0.9, 2.0), 0.5 + 0.9 * 2.0)


def test_0045_q_learning_terminal_target(ns):
    with step("just the reward"):
        expect_allclose(ns["q_learning_terminal_target"](1.0), 1.0)


def test_0046_q_learning_update(ns):
    with step("Q + alpha*(target - Q)"):
        expect_allclose(ns["q_learning_update"](1.0, 0.1, 2.0), 1.0 + 0.1 * (2.0 - 1.0))


def test_0047_episode_reset_game(ns):
    with step("fresh empty board"):
        expect_allclose(ns["episode_reset_game"](), np.zeros((3, 3)))


def test_0048_episode_agent_pick_action(ns):
    q = ns["initialize_q_table"]()
    b = ns["create_empty_board"]()
    with step("returns an action (greedy when epsilon=0)"):
        expect_true(ns["episode_agent_pick_action"](q, b, 0.0, np.random.default_rng(0)) in range(9))


def test_0049_episode_apply_action(ns):
    b = ns["create_empty_board"]()
    nb, status = ns["episode_apply_action"](b, 0, 1)
    with step("returns next board and status"):
        expect_eq(int(nb[0, 0]), 1)
        expect_eq(status, None)


def test_0050_episode_apply_q_update(ns):
    q = ns["initialize_q_table"]()
    key = (0,) * 9
    ns["episode_apply_q_update"](q, key, 4, 1.0, 0.5)  # target 1, alpha .5, from 0 -> 0.5
    with step("moves Q toward the target"):
        expect_allclose(ns["get_q_value"](q, key, 4), 0.5)


def test_0051_episode_check_terminate(ns):
    with step("terminal iff status is not None"):
        expect_true(ns["episode_check_terminate"](1))
        expect_true(not ns["episode_check_terminate"](None))


def test_0052_train_q_learning_agent(ns):
    rng = np.random.default_rng(0)
    q, rewards = ns["train_q_learning_agent"](4000, 0.2, 0.99, 0.2, rng)
    # Greedy evaluation vs a random opponent: the trained agent should rarely lose.
    losses = 0
    for _ in range(200):
        board = ns["create_empty_board"]()
        while ns["get_game_status"](board) is None:
            board = ns["place_move"](board, ns["greedy_argmax_over_legal_actions"](q, board), 1)
            if ns["get_game_status"](board) is not None:
                break
            board = ns["place_move"](board, ns["random_move_agent"](board, rng), -1)
        if ns["get_game_status"](board) == -1:
            losses += 1
    with step("trained agent loses to random < 10% of the time"):
        expect_true(losses / 200 < 0.1, f"too many losses: {losses}/200")


def test_0053_compute_batched_outcome_stats(ns):
    r = ns["compute_batched_outcome_stats"]([1, 1, -1, 0], 1)
    with step("win/loss/draw from perspective"):
        expect_allclose(r["win"], 0.5)
        expect_allclose(r["loss"], 0.25)
        expect_allclose(r["draw"], 0.25)


# --------------------- Part 4 — Self-Play, Evaluation & Persistence ---------------------

def test_0054_self_play_episode(ns):
    q = ns["initialize_q_table"]()
    q, status = ns["self_play_episode"](q, 0.2, 0.99, 0.5, np.random.default_rng(0))
    with step("returns a terminal status and populates the Q-table"):
        expect_true(status in (1, -1, 0), "valid status")
        expect_true(len(q) > 0, "Q-table should have entries")


def test_0055_flip_board_perspective(ns):
    b = _board([[1, -1, 0], [0, 1, 0], [0, 0, -1]])
    with step("player 1 identity; player -1 negates"):
        expect_allclose(ns["flip_board_perspective"](b, 1), b)
        expect_allclose(ns["flip_board_perspective"](b, -1), -b)


def test_0056_perspective_reward_sign(ns):
    with step("sign per player"):
        expect_eq(ns["perspective_reward_sign"](1), 1)
        expect_eq(ns["perspective_reward_sign"](-1), -1)


def test_0057_train_q_agent_self_play(ns):
    rng = np.random.default_rng(0)
    q, statuses = ns["train_q_agent_self_play"](6000, 0.2, 0.99, 0.2, rng)
    losses = 0
    for _ in range(200):
        board = ns["create_empty_board"]()
        while ns["get_game_status"](board) is None:
            board = ns["place_move"](board, ns["greedy_argmax_over_legal_actions"](q, board), 1)
            if ns["get_game_status"](board) is not None:
                break
            board = ns["place_move"](board, ns["random_move_agent"](board, rng), -1)
        if ns["get_game_status"](board) == -1:
            losses += 1
    with step("self-play agent rarely loses to random as X"):
        expect_true(losses / 200 < 0.15, f"too many losses: {losses}/200")


def test_0058_evaluate_q_agent_vs_random(ns):
    rng = np.random.default_rng(1)
    q, _ = ns["train_q_learning_agent"](3000, 0.2, 0.99, 0.2, rng)
    r = ns["evaluate_q_agent_vs_random"](q, 100, rng)
    with step("returns stats that sum to 1; trained agent wins most"):
        expect_allclose(r["win"] + r["loss"] + r["draw"], 1.0)
        expect_true(r["win"] > 0.5, f"expected a winning record, got {r}")


def test_0059_evaluate_q_agent_vs_minimax(ns):
    rng = np.random.default_rng(2)
    q, _ = ns["train_q_learning_agent"](2000, 0.2, 0.99, 0.2, rng)
    r = ns["evaluate_q_agent_vs_minimax"](q, 20)
    with step("cannot beat optimal play; stats valid"):
        expect_allclose(r["win"] + r["loss"] + r["draw"], 1.0)
        expect_allclose(r["win"], 0.0)


def test_0060_inspect_q_values_for_state(ns):
    q = ns["initialize_q_table"]()
    b = ns["create_empty_board"]()
    ns["set_q_value"](q, ns["encode_board_state_key"](b), 4, 1.5)
    out = ns["inspect_q_values_for_state"](q, b)
    with step("length-9 vector with the set value"):
        expect_shape(out, (9,))
        expect_allclose(out[4], 1.5)


def test_0061_serialize_q_table_to_dict(ns):
    q = ns["initialize_q_table"]()
    ns["set_q_value"](q, (0,) * 9, 4, 1.5)
    d = ns["serialize_q_table_to_dict"](q)
    import json
    with step("JSON-serializable"):
        json.dumps(d)  # must not raise


def test_0062_deserialize_q_table_from_dict(ns):
    q = ns["initialize_q_table"]()
    ns["set_q_value"](q, (1, 0, -1, 0, 1, 0, 0, 0, -1), 4, 1.5)
    d = ns["serialize_q_table_to_dict"](q)
    q2 = ns["deserialize_q_table_from_dict"](d)
    with step("round-trips the Q-table"):
        expect_allclose(ns["get_q_value"](q2, (1, 0, -1, 0, 1, 0, 0, 0, -1), 4), 1.5)


# --------------------- Part 5 — Deep Q-Network Agent ---------------------

def test_0063_encode_board_flat_length_nine(ns):
    b = _board([[1, 0, -1], [0, 1, 0], [0, 0, -1]])
    with step("length-9 flat float vector"):
        expect_shape(ns["encode_board_flat_length_nine"](b), (9,))
        expect_allclose(ns["encode_board_flat_length_nine"](b), [1, 0, -1, 0, 1, 0, 0, 0, -1])


def test_0064_encode_board_one_hot_length_eighteen(ns):
    b = _board([[1, -1, 0], [0, 0, 0], [0, 0, 0]])
    out = ns["encode_board_one_hot_length_eighteen"](b)
    with step("length 18: [is_X, is_O] per cell"):
        expect_shape(out, (18,))
        expect_allclose(out[0:2], [1, 0])   # cell 0 = X
        expect_allclose(out[2:4], [0, 1])   # cell 1 = O
        expect_allclose(out[4:6], [0, 0])   # cell 2 = empty


def test_0065_build_mlp_architecture(ns):
    with step("layer sizes"):
        expect_eq(list(ns["build_mlp_architecture"](18, 64, 9)), [18, 64, 9])


def test_0066_initialize_mlp_parameters(ns):
    p = ns["initialize_mlp_parameters"]([18, 64, 9], np.random.default_rng(0))
    with step("shapes W1/b1/W2/b2"):
        expect_shape(p["W1"], (18, 64))
        expect_shape(p["b1"], (64,))
        expect_shape(p["W2"], (64, 9))
        expect_shape(p["b2"], (9,))


def test_0067_mlp_forward_pass(ns):
    p = ns["initialize_mlp_parameters"]([18, 8, 9], np.random.default_rng(1))
    x = np.random.default_rng(2).standard_normal((4, 18))
    with step("(batch, 9) output matching the manual computation"):
        out = ns["mlp_forward_pass"](p, x)
        expect_shape(out, (4, 9))
        manual = np.maximum(x @ p["W1"] + p["b1"], 0) @ p["W2"] + p["b2"]
        expect_allclose(out, manual)


def test_0068_mask_illegal_actions_neg_inf(ns):
    b = _board([[1, 0, 0], [0, 0, 0], [0, 0, 0]])  # cell 0 occupied
    q = np.ones(9)
    masked = ns["mask_illegal_actions_neg_inf"](q, b)
    with step("occupied cells become -inf"):
        expect_true(masked[0] == -np.inf, "illegal action should be -inf")
        expect_true(masked[1] == 1.0, "legal action unchanged")


def test_0069_argmax_action_from_q_values(ns):
    with step("argmax"):
        expect_eq(ns["argmax_action_from_q_values"](np.array([0.0, 5.0, 2.0, -np.inf])), 1)


def test_0070_mse_loss_on_chosen_action(ns):
    q = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    actions = np.array([0, 2])
    targets = np.array([2.0, 1.0])
    with step("MSE on the chosen action"):
        expect_allclose(ns["mse_loss_on_chosen_action"](q, actions, targets),
                        np.mean([(1.0 - 2.0) ** 2, (0.0 - 1.0) ** 2]))


def test_0071_mlp_backward_pass(ns):
    rng = np.random.default_rng(71)
    params = ns["initialize_mlp_parameters"]([18, 8, 9], rng)
    x = rng.standard_normal((5, 18))
    actions = rng.integers(0, 9, size=5)
    targets = rng.standard_normal(5)
    grads = ns["mlp_backward_pass"](params, x, actions, targets)

    def loss_with(key, val):
        p = {**params, key: val}
        return ns["mse_loss_on_chosen_action"](ns["mlp_forward_pass"](p, x), actions, targets)

    with step("all parameter gradients pass a finite-difference check"):
        for k in ("W1", "b1", "W2", "b2"):
            grad_check(lambda v, _k=k: loss_with(_k, v), params[k], 1.0, grads[k], name=f"d{k}")


def test_0072_adam_update_step(ns):
    # Minimize (x - 3)^2 with Adam; x should approach 3.
    params = {"x": np.array([0.0])}
    state = None
    for _ in range(400):
        grads = {"x": 2.0 * (params["x"] - 3.0)}
        params, state = ns["adam_update_step"](params, grads, state, lr=0.1)
    with step("Adam minimizes a simple quadratic"):
        expect_true(abs(params["x"][0] - 3.0) < 0.1, f"x={params['x'][0]}")


def test_0073_create_replay_buffer(ns):
    with step("empty"):
        expect_eq(len(ns["create_replay_buffer"]()), 0)


def test_0074_append_transition_to_buffer(ns):
    buf = ns["create_replay_buffer"]()
    ns["append_transition_to_buffer"](buf, ("s", 0, 1.0, "s2", 1.0))
    with step("grows by one"):
        expect_eq(len(buf), 1)


def test_0075_cap_buffer_size_drop_oldest(ns):
    buf = [i for i in range(10)]
    ns["cap_buffer_size_drop_oldest"](buf, 4)
    with step("keeps the newest max_size"):
        expect_eq(list(buf), [6, 7, 8, 9])


def test_0076_sample_minibatch_from_buffer(ns):
    buf = [(i,) for i in range(20)]
    mb = ns["sample_minibatch_from_buffer"](buf, 8, np.random.default_rng(0))
    with step("returns batch_size transitions from the buffer"):
        expect_eq(len(mb), 8)
        expect_true(all(t in buf for t in mb), "samples must come from the buffer")


def test_0077_build_target_network_copy(ns):
    p = ns["initialize_mlp_parameters"]([18, 8, 9], np.random.default_rng(0))
    t = ns["build_target_network_copy"](p)
    with step("equal values but a distinct array (detached)"):
        expect_allclose(t["W1"], p["W1"])
        t["W1"][0, 0] += 1.0
        expect_true(t["W1"][0, 0] != p["W1"][0, 0], "should be a copy, not a view")


def test_0078_compute_target_q_with_target_network(ns):
    tp = ns["initialize_mlp_parameters"]([18, 8, 9], np.random.default_rng(0))
    ns_states = np.random.default_rng(1).standard_normal((3, 18))
    rewards = np.array([1.0, 0.0, -1.0])
    dones = np.array([1.0, 0.0, 0.0])
    out = ns["compute_target_q_with_target_network"](tp, ns_states, rewards, dones, 0.99)
    qn = ns["mlp_forward_pass"](tp, ns_states)
    with step("r + gamma*max*(1-done); terminal uses reward only"):
        expect_allclose(out[0], 1.0)  # done -> just reward
        expect_allclose(out[1], 0.0 + 0.99 * qn[1].max())
        expect_allclose(out[2], -1.0 + 0.99 * qn[2].max())


def test_0079_sync_target_network_periodically(ns):
    p = ns["initialize_mlp_parameters"]([18, 8, 9], np.random.default_rng(0))
    old = ns["build_target_network_copy"](p)
    with step("copies on sync step, otherwise unchanged"):
        synced = ns["sync_target_network_periodically"](p, old, 250, 250)
        expect_allclose(synced["W1"], p["W1"])
        same = ns["sync_target_network_periodically"](p, old, 251, 250)
        expect_true(same is old, "should return the existing target between syncs")


def test_0080_train_dqn_agent(ns):
    rng = np.random.default_rng(0)
    params = ns["train_dqn_agent"](4000, rng)
    wins = losses = 0
    for _ in range(200):
        board = ns["create_empty_board"]()
        while ns["get_game_status"](board) is None:
            q = ns["mlp_forward_pass"](params, ns["encode_board_one_hot_length_eighteen"](board)[None, :])[0]
            a = ns["argmax_action_from_q_values"](ns["mask_illegal_actions_neg_inf"](q, board))
            board = ns["place_move"](board, a, 1)
            if ns["get_game_status"](board) is not None:
                break
            board = ns["place_move"](board, ns["random_move_agent"](board, rng), -1)
        s = ns["get_game_status"](board)
        wins += s == 1
        losses += s == -1
    with step("trained DQN beats random more than it loses"):
        expect_true(wins > losses, f"DQN should outperform random: wins={wins} losses={losses}")


def test_0081_compare_dqn_tabular_random_minimax(ns):
    rng = np.random.default_rng(1)
    dqn = ns["train_dqn_agent"](1500, rng)
    q, _ = ns["train_q_learning_agent"](1500, 0.2, 0.99, 0.2, rng)
    out = ns["compare_dqn_tabular_random_minimax"](dqn, q, 50, rng)
    with step("reports stats per agent; tabular beats random"):
        for key in ("dqn", "tabular", "random"):
            expect_allclose(out[key]["win"] + out[key]["loss"] + out[key]["draw"], 1.0)
        expect_true(out["tabular"]["win"] > out["random"]["win"], "tabular should beat random more")


# --------------------- Part 6 — Policy Gradients & Extensions ---------------------

def test_0082_sarsa_on_policy_update(ns):
    with step("Q + alpha*(r + gamma*next_q - Q)"):
        expect_allclose(ns["sarsa_on_policy_update"](1.0, 0.1, 0.5, 0.9, 2.0),
                        1.0 + 0.1 * (0.5 + 0.9 * 2.0 - 1.0))


def test_0083_reinforce_log_prob_of_action(ns):
    logits = np.array([1.0, 2.0, 0.5])
    with step("log softmax probability"):
        z = logits - logits.max()
        expect_allclose(ns["reinforce_log_prob_of_action"](logits, 1),
                        z[1] - np.log(np.exp(z).sum()))
    with step("gradient w.r.t. logits is onehot - softmax"):
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        onehot = np.array([0.0, 1.0, 0.0])
        grad_check(lambda L: ns["reinforce_log_prob_of_action"](L, 1), logits, 1.0,
                   onehot - probs, name="dlogits")


def test_0084_reinforce_collect_episode_returns(ns):
    out = ns["reinforce_collect_episode_returns"]([0.0, 0.0, 1.0], 0.9)
    with step("discounted returns-to-go"):
        expect_allclose(out, [0.9 ** 2, 0.9, 1.0])


def test_0085_reinforce_policy_gradient_update(ns):
    rng = np.random.default_rng(85)
    logits = rng.standard_normal((4, 5))
    actions = rng.integers(0, 5, size=4)
    returns = rng.standard_normal(4)
    grads = ns["reinforce_policy_gradient_update"](logits, actions, returns)

    def loss(L):
        return -sum(returns[t] * ns["reinforce_log_prob_of_action"](L[t], actions[t])
                    for t in range(len(actions)))

    with step("gradient of the REINFORCE loss, finite-difference checked"):
        grad_check(loss, logits, 1.0, grads, name="dlogits")


def test_0086_compare_value_vs_policy_learners(ns):
    out = ns["compare_value_vs_policy_learners"](3000, np.random.default_rng(0))
    with step("both learners reported; both beat random"):
        for key in ("value", "policy"):
            expect_allclose(out[key]["win"] + out[key]["loss"] + out[key]["draw"], 1.0)
        expect_true(out["value"]["win"] > out["value"]["loss"], "value learner should win more")
        expect_true(out["policy"]["win"] > out["policy"]["loss"], "policy learner should win more")


def test_0087_symmetry_augmented_training(ns):
    rng = np.random.default_rng(0)
    q = ns["symmetry_augmented_training"](1500, 0.2, 0.99, 0.2, rng)
    losses = 0
    for _ in range(200):
        board = ns["create_empty_board"]()
        while ns["get_game_status"](board) is None:
            board = ns["place_move"](board, ns["greedy_argmax_over_legal_actions"](q, board), 1)
            if ns["get_game_status"](board) is not None:
                break
            board = ns["place_move"](board, ns["random_move_agent"](board, rng), -1)
        if ns["get_game_status"](board) == -1:
            losses += 1
    with step("symmetry augmentation learns fast; rarely loses to random"):
        expect_true(losses / 200 < 0.15, f"too many losses: {losses}/200")
