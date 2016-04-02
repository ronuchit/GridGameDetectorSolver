"""
This file defines the solver for chess.
"""

from solver import Solver

class ChessSolver(Solver):
    WIDTH = 8
    HEIGHT = 8
    GAME_NAME = "chess"

    def detect_and_play(self, board):
        pieces = self._detect(board)
        move = self._get_next_move(pieces)
        # TODO: set up return value correctly as specified in solver.py

    def _detect(self, board):
        # TODO
        pass

    def _get_next_move(self, pieces):
        """
        Pieces is a 8x8 numpy array of pieces from the
        list ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']
        or None if no piece.
        """
        # TODO
        pass
