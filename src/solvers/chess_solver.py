"""
This file defines the solver for chess.
"""

from solver import Solver
import time

class ChessSolver(Solver):
    WIDTH = 8
    HEIGHT = 8
    GAME_NAME = "chess"

    def detect_and_play(self, board):
        pieces = self._detect(board)
        move = self._get_next_move(pieces)
        # TODO: set up return value correctly as specified in solver.py
        time.sleep(2)
        return ("chess_solution_%s"%board, 0.5, 0.99)

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
