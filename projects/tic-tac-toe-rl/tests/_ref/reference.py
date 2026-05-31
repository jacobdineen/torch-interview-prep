"""Reference implementations for tic-tac-toe-rl (HIDDEN).

Conventions used throughout:
  * Board: a 3x3 int numpy array. 0 = empty, +1 = player X, -1 = player O.
  * Players are +1 (X, moves first) and -1 (O).
  * Actions are integers 0..8 in row-major order (action = row*3 + col).

Each step's test grades the user's function by swapping it in over this reference
namespace, so steps are checked in isolation.
"""
import numpy as np


# ============== Part 1 — Board Representation & Game Engine ==============

def create_empty_board():
    """A fresh 3x3 board of zeros (all cells empty)."""
    return np.zeros((3, 3), dtype=int)


def encode_player(symbol):
    """Map a player symbol to its board mark: 'X' -> +1, 'O' -> -1."""
    return 1 if symbol == "X" else -1


def print_board(board):
    """Render the board as a string using X / O / . , rows on separate lines."""
    sym = {0: ".", 1: "X", -1: "O"}
    return "\n".join(" ".join(sym[int(c)] for c in row) for row in board)


def is_cell_empty(board, action):
    """True if the cell at action 0..8 is empty."""
    return bool(board.reshape(-1)[action] == 0)


def place_move(board, action, player):
    """Return a NEW board with ``player``'s mark placed at ``action`` (no mutation)."""
    out = board.copy()
    out[action // 3, action % 3] = player
    return out


def get_legal_moves(board):
    """List of empty action indices (0..8)."""
    return [i for i in range(9) if board.reshape(-1)[i] == 0]


def check_row_win(board, player):
    """True if ``player`` occupies a full row."""
    return bool(np.any(np.all(board == player, axis=1)))


def check_column_win(board, player):
    """True if ``player`` occupies a full column."""
    return bool(np.any(np.all(board == player, axis=0)))


def check_main_diagonal_win(board, player):
    """True if ``player`` occupies the main (top-left to bottom-right) diagonal."""
    return bool(np.all(np.diag(board) == player))


def check_anti_diagonal_win(board, player):
    """True if ``player`` occupies the anti (top-right to bottom-left) diagonal."""
    return bool(np.all(np.diag(np.fliplr(board)) == player))


def is_winner(board, player):
    """True if ``player`` has any winning line (row, column, or diagonal)."""
    return (check_row_win(board, player) or check_column_win(board, player)
            or check_main_diagonal_win(board, player) or check_anti_diagonal_win(board, player))


def is_draw(board):
    """True if the board is full and neither player has won."""
    return len(get_legal_moves(board)) == 0 and not is_winner(board, 1) and not is_winner(board, -1)


def get_game_status(board):
    """Game outcome: +1 (X wins), -1 (O wins), 0 (draw), or None (still in play)."""
    if is_winner(board, 1):
        return 1
    if is_winner(board, -1):
        return -1
    if is_draw(board):
        return 0
    return None


def get_current_player(board):
    """Whose turn it is: +1 if X and O have played equally (X starts), else -1."""
    return 1 if int(np.sum(board == 1)) == int(np.sum(board == -1)) else -1


def switch_player(player):
    """The other player: +1 <-> -1."""
    return -player


def play_hardcoded_game(moves):
    """Play a fixed list of actions, alternating from X, and return the final board."""
    board = create_empty_board()
    player = 1
    for action in moves:
        board = place_move(board, action, player)
        player = switch_player(player)
    return board


def play_interactive_game(get_move):
    """Play to termination, asking ``get_move(board, player)`` for each action.
    Returns the final game status (+1 / -1 / 0). ``get_move`` makes this testable
    without real stdin (pass a scripted callback)."""
    board = create_empty_board()
    player = 1
    while True:
        status = get_game_status(board)
        if status is not None:
            return status
        board = place_move(board, get_move(board, player), player)
        player = switch_player(player)


class TicTacToeGame:
    """A reusable game object wrapping the functional engine.

    Attributes: ``board`` (3x3) and ``current_player`` (+1/-1).
    Methods: reset(), legal_moves(), step(action) -> status, status()."""

    def __init__(self):
        self.board = create_empty_board()
        self.current_player = 1

    def reset(self):
        self.board = create_empty_board()
        self.current_player = 1
        return self

    def legal_moves(self):
        return get_legal_moves(self.board)

    def step(self, action):
        self.board = place_move(self.board, action, self.current_player)
        self.current_player = switch_player(self.current_player)
        return get_game_status(self.board)

    def status(self):
        return get_game_status(self.board)


# ============== Part 2 — Random and Minimax Baselines ==============

def random_move_agent(board, rng):
    """Pick a uniformly random legal action using ``rng`` (a numpy Generator)."""
    return int(rng.choice(get_legal_moves(board)))


def play_random_vs_random_game(rng):
    """Play one game with both sides random; return the final status (+1/-1/0)."""
    board = create_empty_board()
    player = 1
    while get_game_status(board) is None:
        board = place_move(board, random_move_agent(board, rng), player)
        player = switch_player(player)
    return get_game_status(board)


def play_random_vs_random_matches(n_games, rng):
    """Play ``n_games`` random-vs-random games; return the list of statuses."""
    return [play_random_vs_random_game(rng) for _ in range(n_games)]


def compute_outcome_rates(statuses):
    """Fraction of X wins / O wins / draws from a list of statuses."""
    n = len(statuses)
    return {
        "x_win": sum(s == 1 for s in statuses) / n,
        "o_win": sum(s == -1 for s in statuses) / n,
        "draw": sum(s == 0 for s in statuses) / n,
    }


def minimax_terminal_score(board):
    """Score of a terminal board from X's perspective: +1 X win, -1 O win, 0 draw."""
    return get_game_status(board)


def minimax_max_min_step(player, values):
    """Combine child values: the maximizer (X, +1) takes max, the minimizer (O) min."""
    return max(values) if player == 1 else min(values)


def minimax_recursive(board, player):
    """Exact minimax value of ``board`` with ``player`` to move (X maximizes). Uses
    a transposition table (positions reachable many ways are scored once)."""
    cache = {}

    def go(b, p):
        key = (b.tobytes(), p)
        if key in cache:
            return cache[key]
        status = get_game_status(b)
        if status is not None:
            v = minimax_terminal_score(b)
        else:
            v = minimax_max_min_step(p, [go(place_move(b, a, p), switch_player(p))
                                         for a in get_legal_moves(b)])
        cache[key] = v
        return v

    return go(board, player)


def minimax_best_move(board, player):
    """The action with the optimal minimax value for ``player`` (first if tied).
    Scores all candidate moves with one shared transposition table."""
    cache = {}

    def go(b, p):
        key = (b.tobytes(), p)
        if key in cache:
            return cache[key]
        status = get_game_status(b)
        if status is not None:
            v = minimax_terminal_score(b)
        else:
            v = minimax_max_min_step(p, [go(place_move(b, a, p), switch_player(p))
                                         for a in get_legal_moves(b)])
        cache[key] = v
        return v

    legal = get_legal_moves(board)
    values = [go(place_move(board, a, player), switch_player(player)) for a in legal]
    best = minimax_max_min_step(player, values)
    return legal[values.index(best)]


def minimax_alpha_beta(board, player, alpha, beta):
    """Minimax value with alpha-beta pruning (same result as minimax_recursive)."""
    status = get_game_status(board)
    if status is not None:
        return minimax_terminal_score(board)
    if player == 1:
        value = -np.inf
        for a in get_legal_moves(board):
            value = max(value, minimax_alpha_beta(place_move(board, a, player),
                                                  switch_player(player), alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    value = np.inf
    for a in get_legal_moves(board):
        value = min(value, minimax_alpha_beta(place_move(board, a, player),
                                              switch_player(player), alpha, beta))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def play_minimax_vs_random_matches(n_games, rng):
    """Minimax (X) vs random (O). Minimax should never lose; returns statuses.
    Move selection uses alpha-beta (same as minimax, just faster)."""
    results = []
    for _ in range(n_games):
        board = create_empty_board()
        player = 1
        while get_game_status(board) is None:
            a = minimax_best_move(board, 1) if player == 1 else random_move_agent(board, rng)
            board = place_move(board, a, player)
            player = switch_player(player)
        results.append(get_game_status(board))
    return results


def play_minimax_vs_minimax_matches(n_games):
    """Minimax vs minimax — always a draw with optimal play; returns statuses."""
    results = []
    for _ in range(n_games):
        board = create_empty_board()
        player = 1
        while get_game_status(board) is None:
            board = place_move(board, minimax_best_move(board, player), player)
            player = switch_player(player)
        results.append(get_game_status(board))
    return results


# ============== Part 3 — Tabular Q-Learning Foundations ==============

def encode_board_state_key(board):
    """A hashable key for a board: the tuple of its 9 cells (row-major)."""
    return tuple(int(x) for x in board.reshape(-1))


def canonical_board_key(board):
    """Canonical key under the board's 8 symmetries (4 rotations x mirror): the
    lexicographically smallest cell-tuple over all transforms. Shrinks the state space."""
    keys = []
    b = board
    for _ in range(4):
        keys.append(tuple(int(x) for x in b.reshape(-1)))
        keys.append(tuple(int(x) for x in np.fliplr(b).reshape(-1)))
        b = np.rot90(b)
    return min(keys)


def initialize_q_table():
    """An empty Q-table (dict mapping state key -> length-9 action-value array)."""
    return {}


def get_q_value(q_table, state_key, action):
    """Q(state, action), defaulting to 0.0 for unseen states."""
    if state_key not in q_table:
        return 0.0
    return float(q_table[state_key][action])


def set_q_value(q_table, state_key, action, value):
    """Set Q(state, action) (creating a zero row for unseen states); return the table."""
    if state_key not in q_table:
        q_table[state_key] = np.zeros(9)
    q_table[state_key][action] = value
    return q_table


def choose_learning_rate_alpha():
    """A sensible tabular learning rate."""
    return 0.1


def choose_discount_factor_gamma():
    """A sensible discount factor."""
    return 0.99


def choose_initial_epsilon():
    """Start fully exploratory."""
    return 1.0


def epsilon_decay_schedule(epsilon, decay, min_epsilon):
    """Multiplicative epsilon decay with a floor."""
    return max(min_epsilon, epsilon * decay)


def epsilon_greedy_explore_move(board, rng):
    """The explore branch: a uniformly random legal move."""
    return random_move_agent(board, rng)


def greedy_argmax_over_legal_actions(q_table, board):
    """The legal action with the highest Q-value (first if tied)."""
    legal = get_legal_moves(board)
    key = encode_board_state_key(board)
    qs = [get_q_value(q_table, key, a) for a in legal]
    return legal[int(np.argmax(qs))]


def random_tie_break_argmax(values, rng):
    """Argmax index with ties broken uniformly at random."""
    values = np.asarray(values)
    ties = np.flatnonzero(values == values.max())
    return int(rng.choice(ties))


def epsilon_greedy_select_action(q_table, board, epsilon, rng):
    """With prob epsilon explore a random legal move, else act greedily."""
    if rng.random() < epsilon:
        return epsilon_greedy_explore_move(board, rng)
    return greedy_argmax_over_legal_actions(q_table, board)


def tic_tac_toe_reward(status, player):
    """Reward for ``player`` given a game status: +1 win, -1 loss, 0 otherwise."""
    if status == player:
        return 1.0
    if status == -player:
        return -1.0
    return 0.0


def q_learning_nonterminal_target(reward, gamma, next_max_q):
    """TD target with bootstrapping: r + gamma * max_a' Q(s', a')."""
    return reward + gamma * next_max_q


def q_learning_terminal_target(reward):
    """TD target at a terminal state: just the reward (no bootstrap)."""
    return reward


def q_learning_update(q_old, alpha, target):
    """Q-learning update: Q <- Q + alpha * (target - Q)."""
    return q_old + alpha * (target - q_old)


def episode_reset_game():
    """Start-of-episode board."""
    return create_empty_board()


def episode_agent_pick_action(q_table, board, epsilon, rng):
    """The agent's action for this step (epsilon-greedy)."""
    return epsilon_greedy_select_action(q_table, board, epsilon, rng)


def episode_apply_action(board, action, player):
    """Apply an action; return (next_board, status)."""
    nb = place_move(board, action, player)
    return nb, get_game_status(nb)


def episode_apply_q_update(q_table, state_key, action, target, alpha):
    """Apply one Q-learning update for (state_key, action) toward ``target``."""
    old = get_q_value(q_table, state_key, action)
    return set_q_value(q_table, state_key, action, q_learning_update(old, alpha, target))


def episode_check_terminate(status):
    """Whether the episode has ended."""
    return status is not None


def train_q_learning_agent(n_episodes, alpha, gamma, epsilon, rng):
    """Train a Q-learning agent (X) against a random opponent (O). Returns
    (q_table, episode_rewards) where rewards are the agent's terminal rewards."""
    q_table = initialize_q_table()
    rewards = []
    for _ in range(n_episodes):
        board = episode_reset_game()
        while True:
            state_key = encode_board_state_key(board)
            action = episode_agent_pick_action(q_table, board, epsilon, rng)
            board, status = episode_apply_action(board, action, 1)
            if episode_check_terminate(status):
                r = tic_tac_toe_reward(status, 1)
                q_table = episode_apply_q_update(q_table, state_key, action,
                                                 q_learning_terminal_target(r), alpha)
                rewards.append(r)
                break
            # opponent (random) replies
            board, status = episode_apply_action(board, random_move_agent(board, rng), -1)
            if episode_check_terminate(status):
                r = tic_tac_toe_reward(status, 1)
                q_table = episode_apply_q_update(q_table, state_key, action,
                                                 q_learning_terminal_target(r), alpha)
                rewards.append(r)
                break
            next_key = encode_board_state_key(board)
            next_max = max(get_q_value(q_table, next_key, a) for a in get_legal_moves(board))
            target = q_learning_nonterminal_target(0.0, gamma, next_max)
            q_table = episode_apply_q_update(q_table, state_key, action, target, alpha)
    return q_table, rewards


def compute_batched_outcome_stats(statuses, perspective):
    """Win/loss/draw rates for ``perspective`` (+1 or -1) over a list of statuses."""
    n = len(statuses)
    return {
        "win": sum(s == perspective for s in statuses) / n,
        "loss": sum(s == -perspective for s in statuses) / n,
        "draw": sum(s == 0 for s in statuses) / n,
    }
