"""Reference implementations for alphazero-connect4 (HIDDEN).

An AlphaZero-style agent for Connect-4 in PyTorch.
Conventions:
  * Board: a 6x7 int numpy array. 0 = empty, +1 = player one, -1 = player two.
    Row 0 is the top; pieces fall to the lowest empty row.
  * Actions are columns 0..6.

Each step's test grades the user's function by swapping it in over this reference
namespace, so steps are checked in isolation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============== Part 1 — Connect-4 Game Engine ==============

def make_empty_board():
    """A fresh 6x7 board of zeros."""
    return np.zeros((6, 7), dtype=int)


def column_top_row(board, col):
    """Row index a dropped piece would land in (lowest empty row), or -1 if full."""
    for r in range(board.shape[0] - 1, -1, -1):
        if board[r, col] == 0:
            return r
    return -1


def column_full(board, col):
    """True if a column has no empty cell."""
    return column_top_row(board, col) == -1


def valid_moves(board):
    """List of columns that are not full."""
    return [c for c in range(board.shape[1]) if not column_full(board, c)]


def drop_piece(board, col, player):
    """Return a NEW board with ``player``'s piece dropped into ``col``."""
    out = board.copy()
    out[column_top_row(board, col), col] = player
    return out


def four_in_a_row_horizontal(board, player):
    """True if ``player`` has four consecutive pieces in any row."""
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols - 3):
            if all(board[r, c + i] == player for i in range(4)):
                return True
    return False


def four_in_a_row_vertical(board, player):
    """True if ``player`` has four consecutive pieces in any column."""
    rows, cols = board.shape
    for c in range(cols):
        for r in range(rows - 3):
            if all(board[r + i, c] == player for i in range(4)):
                return True
    return False


def four_in_a_row_diagonal_down_right(board, player):
    """True if ``player`` has four in a row along a down-right (\\) diagonal."""
    rows, cols = board.shape
    for r in range(rows - 3):
        for c in range(cols - 3):
            if all(board[r + i, c + i] == player for i in range(4)):
                return True
    return False


def four_in_a_row_diagonal_up_right(board, player):
    """True if ``player`` has four in a row along an up-right (/) diagonal."""
    rows, cols = board.shape
    for r in range(3, rows):
        for c in range(cols - 3):
            if all(board[r - i, c + i] == player for i in range(4)):
                return True
    return False


def check_winner(board):
    """Return the winning player (+1 / -1), or 0 if there is no winner."""
    for player in (1, -1):
        if (four_in_a_row_horizontal(board, player) or four_in_a_row_vertical(board, player)
                or four_in_a_row_diagonal_down_right(board, player)
                or four_in_a_row_diagonal_up_right(board, player)):
            return player
    return 0


def board_is_full(board):
    """True if there are no empty cells."""
    return bool((board != 0).all())


def is_terminal(board):
    """True if the game is over (someone won or the board is full)."""
    return check_winner(board) != 0 or board_is_full(board)


def other_player(player):
    """The opponent of ``player``."""
    return -player


def step_env(board, col, player):
    """Apply ``player``'s move in ``col``. Returns (next_board, reward, done) where
    reward is +1 if this move wins for ``player``, else 0."""
    nb = drop_piece(board, col, player)
    reward = 1.0 if check_winner(nb) == player else 0.0
    return nb, reward, is_terminal(nb)


# ============== Part 2 — Board Encoding and Policy-Value Network ==============

def encode_board(board, player):
    """Two planes from ``player``'s perspective: [my pieces, opponent pieces].
    Shape (2, 6, 7), float."""
    return np.stack([(board == player), (board == -player)]).astype(np.float32)


def board_to_torch_tensor(encoded):
    """Turn an encoded board into a batched float tensor (1, C, 6, 7)."""
    return torch.from_numpy(np.asarray(encoded, dtype=np.float32)).unsqueeze(0)


def init_conv_backbone(in_channels, hidden):
    """A small conv backbone preserving the 6x7 spatial size."""
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU(),
        nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
    )


def init_policy_head(input_dim, n_actions):
    """Linear policy head over the flattened features -> action logits."""
    return nn.Linear(input_dim, n_actions)


def init_value_head(input_dim):
    """Value head -> a scalar in [-1, 1] via tanh."""
    return nn.Sequential(nn.Linear(input_dim, 1), nn.Tanh())


def build_policy_value_net(in_channels=2, hidden=32, n_actions=7):
    """Assemble the conv backbone + policy and value heads into one module whose
    forward(x) returns (policy_logits (B, n_actions), value (B,))."""
    flat = hidden * 6 * 7
    backbone = init_conv_backbone(in_channels, hidden)
    policy_head = init_policy_head(flat, n_actions)
    value_head = init_value_head(flat)

    class _PolicyValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.policy_head = policy_head
            self.value_head = value_head

        def forward(self, x):
            h = self.backbone(x).flatten(1)
            return self.policy_head(h), self.value_head(h).squeeze(-1)

    return _PolicyValueNet()


def policy_value_forward(net, x):
    """Run the network: returns (policy_logits, value)."""
    return net(x)


# ============== Part 3 — Action Masking and Policy Sampling ==============

def action_mask(board):
    """A length-7 float mask: 1 for legal (non-full) columns, 0 otherwise."""
    mask = torch.zeros(board.shape[1])
    for c in valid_moves(board):
        mask[c] = 1.0
    return mask


def masked_policy_logits(logits, mask):
    """Set logits of illegal actions (mask==0) to -inf."""
    return logits.masked_fill(mask == 0, float("-inf"))


def masked_log_softmax(logits, mask):
    """Log-softmax over legal actions only (illegal -> -inf)."""
    return F.log_softmax(masked_policy_logits(logits, mask), dim=-1)


def sample_action_from_policy(logits, mask, generator=None):
    """Sample a legal column from softmax(masked logits)."""
    probs = torch.softmax(masked_policy_logits(logits, mask), dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


def greedy_action_from_policy(logits, mask):
    """The legal column with the highest logit."""
    return int(torch.argmax(masked_policy_logits(logits, mask)).item())


# ============== Part 4 — PUCT Monte Carlo Tree Search ==============

def make_mcts_node(board, player, prior):
    """An MCTS node for a state: the board, the player to move, the prior prob of
    reaching it, visit count N, value sum W, children, and an expanded flag."""
    return {"board": board, "player": player, "prior": prior,
            "N": 0, "W": 0.0, "children": {}, "expanded": False}


def node_q_value(node):
    """Mean value of a node from its own player's perspective (0 if unvisited)."""
    return node["W"] / node["N"] if node["N"] > 0 else 0.0


def ucb_score(parent_visits, child, c_puct):
    """PUCT score of a child from the PARENT's perspective: -Q(child) (the child's
    value is from the opponent's view) + c_puct * P * sqrt(N_parent) / (1 + N_child)."""
    exploration = c_puct * child["prior"] * (parent_visits ** 0.5) / (1 + child["N"])
    return -node_q_value(child) + exploration


def select_best_child(node, c_puct):
    """The (action, child) with the highest PUCT score."""
    best_action, best_child, best_score = None, None, -float("inf")
    for action, child in node["children"].items():
        score = ucb_score(node["N"], child, c_puct)
        if score > best_score:
            best_action, best_child, best_score = action, child, score
    return best_action, best_child


def select_leaf(root, c_puct):
    """Walk down via PUCT until an unexpanded or terminal node. Returns (leaf, path)."""
    node = root
    path = [root]
    while node["expanded"] and not is_terminal(node["board"]):
        _, node = select_best_child(node, c_puct)
        path.append(node)
    return node, path


def evaluate_with_network(net, board, player):
    """Network evaluation of a position: (priors over 7 columns, value in [-1,1]),
    from ``player``'s perspective. Illegal columns get prior 0."""
    x = board_to_torch_tensor(encode_board(board, player))
    with torch.no_grad():
        logits, value = policy_value_forward(net, x)
        probs = torch.softmax(masked_policy_logits(logits[0], action_mask(board)), dim=-1)
    return probs.cpu().numpy(), float(value.item())


def expand_node(node, priors):
    """Create a child for every legal move, carrying its prior. Marks node expanded."""
    for a in valid_moves(node["board"]):
        child_board = drop_piece(node["board"], a, node["player"])
        node["children"][a] = make_mcts_node(child_board, other_player(node["player"]), float(priors[a]))
    node["expanded"] = True
    return node


def backup_value(path, value):
    """Propagate ``value`` (from the leaf player's perspective) up the path, flipping
    sign at each level since players alternate."""
    for node in reversed(path):
        node["N"] += 1
        node["W"] += value
        value = -value


def run_one_simulation(root, net, c_puct):
    """One MCTS simulation: select a leaf, evaluate/expand (or use the game result if
    terminal), and back the value up."""
    leaf, path = select_leaf(root, c_puct)
    if is_terminal(leaf["board"]):
        value = float(check_winner(leaf["board"]) * leaf["player"])
    else:
        priors, value = evaluate_with_network(net, leaf["board"], leaf["player"])
        expand_node(leaf, priors)
    backup_value(path, value)


def run_mcts(root, net, n_simulations, c_puct):
    """Run ``n_simulations`` simulations from ``root`` and return it."""
    for _ in range(n_simulations):
        run_one_simulation(root, net, c_puct)
    return root


def visit_count_policy(root, temperature):
    """Policy over 7 columns from child visit counts, sharpened by ``temperature``
    (temperature 0 -> one-hot on the most-visited action)."""
    counts = np.zeros(7)
    for a, child in root["children"].items():
        counts[a] = child["N"]
    if temperature == 0 or counts.sum() == 0:
        pi = np.zeros(7)
        pi[int(np.argmax(counts))] = 1.0
        return pi
    sharpened = counts ** (1.0 / temperature)
    return sharpened / sharpened.sum()


def mcts_choose_action(root, temperature, rng=None):
    """Pick a column: argmax visits at temperature 0, else sample the visit policy."""
    pi = visit_count_policy(root, temperature)
    if temperature == 0:
        return int(np.argmax(pi))
    rng = rng or np.random.default_rng()
    return int(rng.choice(len(pi), p=pi))


# ============== Part 5 — Self-Play Data Generation ==============

def record_self_play_step(buffer, state, policy, player):
    """Append a (state, MCTS policy, player) tuple; the value target is filled in
    later from the game outcome. Returns the buffer."""
    buffer.append((state, policy, player))
    return buffer


def play_self_play_game(net, n_simulations, c_puct, temperature, rng):
    """Play one MCTS self-play game. Returns (records, winner) where records are
    (encoded_state, mcts_policy, player) tuples."""
    board = make_empty_board()
    player = 1
    records = []
    while not is_terminal(board):
        root = make_mcts_node(board, player, 0.0)
        run_mcts(root, net, n_simulations, c_puct)
        record_self_play_step(records, encode_board(board, player),
                              visit_count_policy(root, temperature), player)
        action = mcts_choose_action(root, temperature, rng)
        board = drop_piece(board, action, player)
        player = other_player(player)
    return records, check_winner(board)


def assign_value_targets(records, winner):
    """Turn (state, policy, player) records into (state, policy, value) where value is
    +1 if that player won, -1 if they lost, 0 for a draw."""
    return [(state, policy, float(winner * player)) for state, policy, player in records]


def generate_self_play_batch(net, n_games, n_simulations, c_puct, temperature, rng):
    """Play ``n_games`` self-play games and return all (state, policy, value) tuples."""
    data = []
    for _ in range(n_games):
        records, winner = play_self_play_game(net, n_simulations, c_puct, temperature, rng)
        data.extend(assign_value_targets(records, winner))
    return data


# ============== Part 6 — Losses and Training Loop ==============

def value_loss_mse(pred_values, target_values):
    """Mean squared error between predicted and target values."""
    return F.mse_loss(pred_values, target_values)


def policy_loss_cross_entropy(policy_logits, target_policy):
    """Cross-entropy between the network policy and the MCTS target distribution."""
    return -(target_policy * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()


def l2_regularization_loss(net, weight_decay):
    """L2 penalty on all parameters."""
    return weight_decay * sum((p ** 2).sum() for p in net.parameters())


def combined_loss(value_loss, policy_loss, l2_loss):
    """AlphaZero loss: value MSE + policy cross-entropy + L2."""
    return value_loss + policy_loss + l2_loss


def encode_batch_states(states):
    """Stack a list of encoded (2,6,7) states into a (B,2,6,7) float tensor."""
    return torch.from_numpy(np.stack(states)).float()


def iterate_minibatches(data, batch_size, shuffle=False, generator=None):
    """Split ``data`` into minibatches (optionally shuffled). Returns a list of batches."""
    idx = (torch.randperm(len(data), generator=generator).tolist() if shuffle
           else list(range(len(data))))
    return [[data[j] for j in idx[i:i + batch_size]] for i in range(0, len(data), batch_size)]


def training_step(net, batch, optimizer, weight_decay):
    """One training step over a list of (state, policy, value) tuples. Returns loss."""
    states = encode_batch_states([b[0] for b in batch])
    target_pi = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    target_v = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    logits, value = policy_value_forward(net, states)
    loss = combined_loss(value_loss_mse(value, target_v),
                         policy_loss_cross_entropy(logits, target_pi),
                         l2_regularization_loss(net, weight_decay))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def training_epoch(net, data, optimizer, batch_size, weight_decay, generator=None):
    """One pass over ``data`` in minibatches. Returns the mean loss."""
    losses = [training_step(net, batch, optimizer, weight_decay)
              for batch in iterate_minibatches(data, batch_size, shuffle=True, generator=generator)]
    return sum(losses) / len(losses)


# ============== Part 7 — Iterated Training Loop ==============

def self_play_iteration(net, optimizer, n_games, n_simulations, c_puct, temperature,
                        batch_size, weight_decay, rng):
    """One AlphaZero iteration: generate self-play data, then train one epoch on it.
    Returns (mean_loss, data)."""
    data = generate_self_play_batch(net, n_games, n_simulations, c_puct, temperature, rng)
    loss = training_epoch(net, data, optimizer, batch_size, weight_decay)
    return loss, data


def train_loop(net, optimizer, n_iterations, n_games, n_simulations, c_puct, temperature,
               batch_size, weight_decay, rng):
    """Alternate self-play and training for ``n_iterations``. Returns (net, losses)."""
    losses = []
    for _ in range(n_iterations):
        loss, _ = self_play_iteration(net, optimizer, n_games, n_simulations, c_puct,
                                      temperature, batch_size, weight_decay, rng)
        losses.append(loss)
    return net, losses


# ============== Part 8 — Agents and Evaluation ==============

def random_policy_action(board, rng):
    """A random legal column."""
    return int(rng.choice(valid_moves(board)))


def greedy_agent_action(net, board, player):
    """Greedy column from the network's policy (no search)."""
    x = board_to_torch_tensor(encode_board(board, player))
    with torch.no_grad():
        logits, _ = policy_value_forward(net, x)
    return greedy_action_from_policy(logits[0], action_mask(board))


def play_one_match(agent_a, agent_b, rng):
    """Play a game: agent_a is player +1, agent_b is player -1. Each agent is a
    callable(board, player) -> action. Returns the winner (+1/-1/0)."""
    board = make_empty_board()
    player = 1
    while not is_terminal(board):
        action = agent_a(board, player) if player == 1 else agent_b(board, player)
        board = drop_piece(board, action, player)
        player = other_player(player)
    return check_winner(board)


def match_win_rate(results, perspective):
    """Fraction of games won by ``perspective`` (+1/-1)."""
    return sum(r == perspective for r in results) / len(results)


def evaluate_against_random(net, n_games, rng):
    """Greedy network agent (as +1) vs a random opponent; return its win rate."""
    results = []
    for _ in range(n_games):
        results.append(play_one_match(
            lambda b, p: greedy_agent_action(net, b, p),
            lambda b, p: random_policy_action(b, rng), rng))
    return match_win_rate(results, 1)
