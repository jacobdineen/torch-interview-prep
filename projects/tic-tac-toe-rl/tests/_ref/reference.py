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
