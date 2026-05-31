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
