"""
This file defines the solver for Sudoku.
"""

from solver import Solver

class SudokuSolver(Solver):
    WIDTH = 9
    HEIGHT = 9
    GAME_NAME = "sudoku"

    def detect_and_play(self, board):
        board_repr = self._detect(board)
        solution = self._get_solution(board_repr)
        # TODO: set up return value correctly as specified in solver.py

    def _detect(self, board):
        # TODO
        pass

    def _get_solution(self, board_repr):
        """
        TODO: explain the representation of the board here (Amy)
        """
        # TODO
        pass
