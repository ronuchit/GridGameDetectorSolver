"""
This file defines the solver for Sudoku.
"""

from solver import Solver
import time
from sklearn import datasets, svm, metrics
from sklearn.externals import joblib

class SudokuSolver(Solver):
    WIDTH = 9
    HEIGHT = 9
    GAME_NAME = "sudoku"

    def __init__(self):
        self.classifier = joblib.load('../../lib/mnist_svm/svm.pkl')

    def detect_and_play(self, board):
        board_repr = self._detect(board)
        solution = self._get_solution(board_repr)
        # TODO: set up return value correctly as specified in solver.py
        time.sleep(2)
        return ("sudoku_solution", 0.5, 0.99)

    def _detect(self, board):
        flat = np.asarray([x for x in row for row in board])
        data = digits.images.reshape((n_samples, -1))
        pass

    def _get_solution(self, board_repr):
        """
        TODO: explain the representation of the board here (Amy)
        """
        # TODO
        pass
