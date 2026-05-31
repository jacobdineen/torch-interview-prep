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


# ============== Part 4 — Self-Play, Evaluation & Persistence ==============

def flip_board_perspective(board, player):
    """View the board from ``player``'s side: multiply by player so the mover's
    pieces are always +1. One Q-table can then serve both sides."""
    return board * player


def perspective_reward_sign(player):
    """Sign to convert an X-perspective reward to ``player``'s perspective (+1/-1)."""
    return player


def self_play_episode(q_table, alpha, gamma, epsilon, rng):
    """Run one self-play training episode (the agent plays both sides, learning
    from each via perspective flipping). Mutates and returns (q_table, status)."""
    board = create_empty_board()
    player = 1
    pending = {1: None, -1: None}  # each player's (state_key, action) awaiting a bootstrap
    while True:
        persp = flip_board_perspective(board, player)
        state_key = encode_board_state_key(persp)
        if pending[player] is not None:  # bootstrap this player's previous move
            psk, pa = pending[player]
            nmax = max(get_q_value(q_table, state_key, a) for a in get_legal_moves(persp))
            q_table = episode_apply_q_update(
                q_table, psk, pa, q_learning_nonterminal_target(0.0, gamma, nmax), alpha)
        action = epsilon_greedy_select_action(q_table, persp, epsilon, rng)
        board = place_move(board, action, player)
        pending[player] = (state_key, action)
        status = get_game_status(board)
        if status is not None:
            q_table = episode_apply_q_update(
                q_table, state_key, action,
                q_learning_terminal_target(tic_tac_toe_reward(status, player)), alpha)
            opp = switch_player(player)
            if pending[opp] is not None:
                osk, oa = pending[opp]
                q_table = episode_apply_q_update(
                    q_table, osk, oa,
                    q_learning_terminal_target(tic_tac_toe_reward(status, opp)), alpha)
            return q_table, status
        player = switch_player(player)


def train_q_agent_self_play(n_episodes, alpha, gamma, epsilon, rng):
    """Train via self-play for ``n_episodes``; return (q_table, statuses)."""
    q_table = initialize_q_table()
    statuses = []
    for _ in range(n_episodes):
        q_table, status = self_play_episode(q_table, alpha, gamma, epsilon, rng)
        statuses.append(status)
    return q_table, statuses


def evaluate_q_agent_vs_random(q_table, n_games, rng):
    """Greedy agent (X) vs random (O); return win/loss/draw stats from X's view."""
    statuses = []
    for _ in range(n_games):
        board = create_empty_board()
        while get_game_status(board) is None:
            board = place_move(board, greedy_argmax_over_legal_actions(q_table, board), 1)
            if get_game_status(board) is not None:
                break
            board = place_move(board, random_move_agent(board, rng), -1)
        statuses.append(get_game_status(board))
    return compute_batched_outcome_stats(statuses, 1)


def evaluate_q_agent_vs_minimax(q_table, n_games):
    """Greedy agent (X) vs optimal minimax (O); return stats from X's view.
    Against optimal play the agent can never win (best case is a draw)."""
    statuses = []
    for _ in range(n_games):
        board = create_empty_board()
        while get_game_status(board) is None:
            board = place_move(board, greedy_argmax_over_legal_actions(q_table, board), 1)
            if get_game_status(board) is not None:
                break
            board = place_move(board, minimax_best_move(board, -1), -1)
        statuses.append(get_game_status(board))
    return compute_batched_outcome_stats(statuses, 1)


def inspect_q_values_for_state(q_table, board):
    """The length-9 vector of Q-values for a board (zeros for unseen states)."""
    key = encode_board_state_key(board)
    return np.array([get_q_value(q_table, key, a) for a in range(9)])


def serialize_q_table_to_dict(q_table):
    """Convert a Q-table to a JSON-serializable {comma-joined key: list} dict."""
    return {",".join(str(int(x)) for x in key): list(map(float, vals))
            for key, vals in q_table.items()}


def deserialize_q_table_from_dict(d):
    """Inverse of serialize_q_table_to_dict."""
    return {tuple(int(x) for x in key.split(",")): np.array(vals, dtype=float)
            for key, vals in d.items()}


# ============== Part 5 — Deep Q-Network Agent ==============

def encode_board_flat_length_nine(board):
    """Flatten the board to a length-9 float vector."""
    return board.reshape(-1).astype(float)


def encode_board_one_hot_length_eighteen(board):
    """Two channels per cell: [is_X, is_O], flattened to length 18."""
    flat = board.reshape(-1)
    out = np.zeros(18)
    for i, c in enumerate(flat):
        if c == 1:
            out[2 * i] = 1.0
        elif c == -1:
            out[2 * i + 1] = 1.0
    return out


def build_mlp_architecture(input_dim, hidden_dim, output_dim):
    """Layer sizes for a 1-hidden-layer MLP: [input, hidden, output]."""
    return [input_dim, hidden_dim, output_dim]


def initialize_mlp_parameters(arch, rng):
    """He-initialized parameters {W1,b1,W2,b2} for the architecture."""
    i, h, o = arch
    return {
        "W1": rng.standard_normal((i, h)) * np.sqrt(2.0 / i),
        "b1": np.zeros(h),
        "W2": rng.standard_normal((h, o)) * np.sqrt(2.0 / h),
        "b2": np.zeros(o),
    }


def mlp_forward_pass(params, x):
    """Forward pass of the MLP: relu(x@W1+b1)@W2+b2. x is (batch, input_dim)."""
    h = np.maximum(x @ params["W1"] + params["b1"], 0.0)
    return h @ params["W2"] + params["b2"]


def mask_illegal_actions_neg_inf(q_values, board):
    """Set Q-values of illegal (occupied) cells to -inf so they're never chosen."""
    masked = np.array(q_values, dtype=float)
    legal = set(get_legal_moves(board))
    for a in range(9):
        if a not in legal:
            masked[a] = -np.inf
    return masked


def argmax_action_from_q_values(q_values):
    """The action with the highest Q-value."""
    return int(np.argmax(q_values))


def mse_loss_on_chosen_action(q_pred, actions, targets):
    """Mean squared error on the taken action only (DQN updates that action).
    q_pred (batch,9), actions (batch,), targets (batch,)."""
    idx = np.arange(len(actions))
    return float(np.mean((q_pred[idx, actions] - targets) ** 2))


def mlp_backward_pass(params, x, actions, targets):
    """Gradients of the chosen-action MSE loss. Returns {W1,b1,W2,b2}."""
    z1 = x @ params["W1"] + params["b1"]
    h = np.maximum(z1, 0.0)
    q = h @ params["W2"] + params["b2"]
    b = x.shape[0]
    idx = np.arange(b)
    dq = np.zeros_like(q)
    dq[idx, actions] = 2.0 * (q[idx, actions] - targets) / b
    dw2 = h.T @ dq
    db2 = dq.sum(axis=0)
    dh = dq @ params["W2"].T
    dz1 = dh * (z1 > 0)
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2}


def adam_update_step(params, grads, state, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
    """One Adam step over the parameter dict. ``state`` is None on the first call.
    Returns (new_params, state)."""
    b1, b2 = betas
    if state is None:
        state = {"m": {k: np.zeros_like(v) for k, v in params.items()},
                 "v": {k: np.zeros_like(v) for k, v in params.items()}, "t": 0}
    state["t"] += 1
    t = state["t"]
    new_params = {}
    for k in params:
        g = grads[k]
        state["m"][k] = b1 * state["m"][k] + (1 - b1) * g
        state["v"][k] = b2 * state["v"][k] + (1 - b2) * g ** 2
        mhat = state["m"][k] / (1 - b1 ** t)
        vhat = state["v"][k] / (1 - b2 ** t)
        new_params[k] = params[k] - lr * mhat / (np.sqrt(vhat) + eps)
    return new_params, state


def create_replay_buffer():
    """An empty replay buffer (list of transitions)."""
    return []


def append_transition_to_buffer(buffer, transition):
    """Append a (state, action, reward, next_state, done) transition; return buffer."""
    buffer.append(transition)
    return buffer


def cap_buffer_size_drop_oldest(buffer, max_size):
    """Keep the buffer at most ``max_size`` by dropping the oldest transitions."""
    while len(buffer) > max_size:
        buffer.pop(0)
    return buffer


def sample_minibatch_from_buffer(buffer, batch_size, rng):
    """Sample ``batch_size`` transitions uniformly (with replacement)."""
    idx = rng.integers(0, len(buffer), size=batch_size)
    return [buffer[i] for i in idx]


def build_target_network_copy(params):
    """A detached copy of the parameters for the target network."""
    return {k: v.copy() for k, v in params.items()}


def compute_target_q_with_target_network(target_params, next_states, rewards, dones, gamma):
    """DQN targets: r + gamma * max_a Q_target(s') * (1 - done). Batched."""
    qn = mlp_forward_pass(target_params, next_states)
    return rewards + gamma * qn.max(axis=1) * (1.0 - dones)


def sync_target_network_periodically(params, target_params, step, sync_every):
    """Every ``sync_every`` steps, copy the online params into the target net."""
    if step % sync_every == 0:
        return build_target_network_copy(params)
    return target_params


def train_dqn_agent(n_episodes, rng):
    """Train a DQN agent (X) vs a random opponent (O). Returns the online params."""
    params = initialize_mlp_parameters(build_mlp_architecture(18, 64, 9), rng)
    target = build_target_network_copy(params)
    buffer = create_replay_buffer()
    opt_state = None
    epsilon, gamma, batch, sync_every, max_buf = 1.0, 0.99, 32, 250, 2000
    stepc = 0
    for _ in range(n_episodes):
        board = create_empty_board()
        while True:
            s = encode_board_one_hot_length_eighteen(board)
            if rng.random() < epsilon:
                action = random_move_agent(board, rng)
            else:
                q = mlp_forward_pass(params, s[None, :])[0]
                action = argmax_action_from_q_values(mask_illegal_actions_neg_inf(q, board))
            board = place_move(board, action, 1)
            status = get_game_status(board)
            if status is not None:
                reward, done = tic_tac_toe_reward(status, 1), 1.0
            else:
                board = place_move(board, random_move_agent(board, rng), -1)
                status = get_game_status(board)
                reward = tic_tac_toe_reward(status, 1) if status is not None else 0.0
                done = 1.0 if status is not None else 0.0
            ns_vec = encode_board_one_hot_length_eighteen(board)
            buffer = append_transition_to_buffer(buffer, (s, action, reward, ns_vec, done))
            buffer = cap_buffer_size_drop_oldest(buffer, max_buf)
            stepc += 1
            if len(buffer) >= batch:
                mb = sample_minibatch_from_buffer(buffer, batch, rng)
                states = np.array([t[0] for t in mb])
                actions = np.array([t[1] for t in mb])
                rewards = np.array([t[2] for t in mb])
                next_states = np.array([t[3] for t in mb])
                dones = np.array([t[4] for t in mb])
                targets = compute_target_q_with_target_network(target, next_states, rewards, dones, gamma)
                grads = mlp_backward_pass(params, states, actions, targets)
                params, opt_state = adam_update_step(params, grads, opt_state, 1e-3)
                target = sync_target_network_periodically(params, target, stepc, sync_every)
            if done:
                break
        epsilon = max(0.1, epsilon * 0.9995)
    return params


def compare_dqn_tabular_random_minimax(dqn_params, q_table, n_games, rng):
    """Win/loss/draw vs a random opponent for each agent (DQN, tabular, random).
    Returns a dict keyed by agent name."""
    def dqn_move(board):
        q = mlp_forward_pass(dqn_params, encode_board_one_hot_length_eighteen(board)[None, :])[0]
        return argmax_action_from_q_values(mask_illegal_actions_neg_inf(q, board))

    def play_vs_random(move_fn):
        statuses = []
        for _ in range(n_games):
            board = create_empty_board()
            while get_game_status(board) is None:
                board = place_move(board, move_fn(board), 1)
                if get_game_status(board) is not None:
                    break
                board = place_move(board, random_move_agent(board, rng), -1)
            statuses.append(get_game_status(board))
        return compute_batched_outcome_stats(statuses, 1)

    return {
        "dqn": play_vs_random(dqn_move),
        "tabular": play_vs_random(lambda b: greedy_argmax_over_legal_actions(q_table, b)),
        "random": play_vs_random(lambda b: random_move_agent(b, rng)),
    }


# ============== Part 6 — Policy Gradients & Extensions ==============

def sarsa_on_policy_update(q_old, alpha, reward, gamma, next_q):
    """On-policy SARSA update: uses Q(s',a') for the action actually taken next,
    not the max. Q <- Q + alpha*(r + gamma*next_q - Q)."""
    return q_old + alpha * (reward + gamma * next_q - q_old)


def reinforce_log_prob_of_action(logits, action):
    """Log of the softmax probability assigned to ``action``."""
    z = logits - np.max(logits)
    return z[action] - np.log(np.sum(np.exp(z)))


def reinforce_collect_episode_returns(rewards, gamma):
    """Discounted returns-to-go: G_t = sum_{k>=t} gamma^(k-t) * r_k."""
    out = np.zeros(len(rewards))
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def reinforce_policy_gradient_update(logits_list, actions, returns):
    """Per-step gradient of the REINFORCE loss (-sum_t G_t log pi(a_t)) w.r.t. the
    logits: G_t * (softmax(logits_t) - onehot(a_t)). Shape (T, num_actions)."""
    logits_list = np.asarray(logits_list, dtype=float)
    t, a = logits_list.shape
    z = logits_list - logits_list.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    onehot = np.zeros((t, a))
    onehot[np.arange(t), actions] = 1.0
    return np.asarray(returns)[:, None] * (probs - onehot)


def compare_value_vs_policy_learners(n_episodes, rng):
    """Train a value-based (tabular Q-learning) and a policy-based (tabular
    REINFORCE) agent, evaluate both vs random, and return {'value','policy'} stats."""
    # value-based
    q_table, _ = train_q_learning_agent(n_episodes, 0.2, 0.99, 0.2, rng)
    value_stats = evaluate_q_agent_vs_random(q_table, 100, rng)

    # policy-based: tabular softmax policy trained with REINFORCE
    policy, lr, gamma = {}, 0.1, 0.99
    for _ in range(n_episodes):
        board = create_empty_board()
        traj, rewards = [], []
        while True:
            key = encode_board_state_key(board)
            masked = mask_illegal_actions_neg_inf(policy.get(key, np.zeros(9)), board)
            z = masked - masked.max()
            probs = np.exp(z) / np.exp(z).sum()
            action = int(rng.choice(9, p=probs))
            traj.append((key, masked, action))
            board = place_move(board, action, 1)
            status = get_game_status(board)
            if status is not None:
                rewards.append(tic_tac_toe_reward(status, 1))
                break
            board = place_move(board, random_move_agent(board, rng), -1)
            status = get_game_status(board)
            rewards.append(tic_tac_toe_reward(status, 1) if status is not None else 0.0)
            if status is not None:
                break
        returns = reinforce_collect_episode_returns(rewards, gamma)
        grads = reinforce_policy_gradient_update(
            np.array([m for _, m, _ in traj]), np.array([a for _, _, a in traj]), returns)
        for i, (key, _, _) in enumerate(traj):
            if key not in policy:
                policy[key] = np.zeros(9)
            policy[key] = policy[key] - lr * grads[i]

    def policy_move(board):
        key = encode_board_state_key(board)
        return argmax_action_from_q_values(
            mask_illegal_actions_neg_inf(policy.get(key, np.zeros(9)), board))

    statuses = []
    for _ in range(100):
        board = create_empty_board()
        while get_game_status(board) is None:
            board = place_move(board, policy_move(board), 1)
            if get_game_status(board) is not None:
                break
            board = place_move(board, random_move_agent(board, rng), -1)
        statuses.append(get_game_status(board))
    return {"value": value_stats, "policy": compute_batched_outcome_stats(statuses, 1)}


def symmetry_augmented_training(n_episodes, alpha, gamma, epsilon, rng):
    """Q-learning that applies each update to all 8 symmetric (state, action) pairs,
    learning ~8x faster per episode. Returns the Q-table."""
    def augment(board, action):
        marker = np.arange(9).reshape(3, 3)
        out = []
        b, m = board, marker
        for _ in range(4):
            for bb, mm in [(b, m), (np.fliplr(b), np.fliplr(m))]:
                key = tuple(int(x) for x in bb.reshape(-1))
                na = int(np.flatnonzero(mm.reshape(-1) == action)[0])
                out.append((key, na))
            b, m = np.rot90(b), np.rot90(m)
        return out

    q_table = initialize_q_table()
    for _ in range(n_episodes):
        board = create_empty_board()
        while True:
            action = epsilon_greedy_select_action(q_table, board, epsilon, rng)
            nb = place_move(board, action, 1)
            status = get_game_status(nb)
            if status is not None:
                target = q_learning_terminal_target(tic_tac_toe_reward(status, 1))
            else:
                nb = place_move(nb, random_move_agent(nb, rng), -1)
                status = get_game_status(nb)
                if status is not None:
                    target = q_learning_terminal_target(tic_tac_toe_reward(status, 1))
                else:
                    nmax = max(get_q_value(q_table, encode_board_state_key(nb), a)
                               for a in get_legal_moves(nb))
                    target = q_learning_nonterminal_target(0.0, gamma, nmax)
            for key, a in augment(board, action):
                set_q_value(q_table, key, a, q_learning_update(get_q_value(q_table, key, a), alpha, target))
            if status is not None:
                break
            board = nb
    return q_table
