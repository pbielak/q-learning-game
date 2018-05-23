"""
Tic-Tac-Toe game module
"""
import enum


class FieldStates(enum.Enum):
    EMPTY_FIELD = ''
    X_MARKER = 'X'
    O_MARKER = 'O'


class GameStatus(enum.Enum):
    PLAYING = 0
    DRAW = 1
    X_WIN = 2
    O_WIN = 3


class TicTacToe(object):
    BOARD_SIZE = 3

    """
    BOARD:
    -------------
    | 0 | 1 | 2 |
    | 3 | 4 | 5 |
    | 6 | 7 | 8 |
    -------------
    """

    def __init__(self):
        self.board = None
        self.round = None
        self._initialize_game()

    def reset(self):
        self._initialize_game()

    def _initialize_game(self):
        self.board = [FieldStates.EMPTY_FIELD, ] * (TicTacToe.BOARD_SIZE ** 2)
        self.round = 0

    def set_field(self, field_idx, field_type):
        self.board[field_idx] = field_type
        self.round += 1

    def is_field_empty(self, field_idx):
        return self.board[field_idx] == FieldStates.EMPTY_FIELD

    def is_terminal(self):
        return self.status != GameStatus.PLAYING

    def _check_all_equal(self, indexes):
        sublist = [self.board[idx] for idx in indexes]
        return len(set(sublist)) <= 1

    def _get_match(self):
        indexes = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # ROWS
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # COLUMNS
            (0, 4, 8), (2, 4, 6),  # DIAGONALS
        ]

        for idx_tuple in indexes:
            field_status = self.board[idx_tuple[0]]
            all_equal = self._check_all_equal(idx_tuple)
            if all_equal and field_status != FieldStates.EMPTY_FIELD:
                return field_status

        return None

    def _empty_fields_present(self):
        return self.board.count(FieldStates.EMPTY_FIELD) > 0

    @property
    def status(self):
        match = self._get_match()
        empty_fields_present = self._empty_fields_present()

        if match:
            if match.value == FieldStates.X_MARKER.value:
                return GameStatus.X_WIN

            if match.value == FieldStates.O_MARKER.value:
                return GameStatus.O_WIN

        else:
            if not empty_fields_present:
                return GameStatus.DRAW

            return GameStatus.PLAYING

    @property
    def current_player(self):
        if self.round % 2 == 0:
            return FieldStates.X_MARKER

        return FieldStates.O_MARKER
