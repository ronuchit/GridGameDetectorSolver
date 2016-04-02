"""
This file defines the solver for Sudoku.
"""

from solver import Solver
import time
from sklearn import datasets, svm, metrics
from sklearn.externals import joblib
import re
import cv2
import IPython

class SudokuSolver(Solver):
    WIDTH = 9
    HEIGHT = 9
    GAME_NAME = "sudoku"

    def __init__(self):
        self.templates = {}

    def detect_and_play(self, board):
        board_repr = self._detect(board)
        board_repr = "000000208920004000000208071036000000000709000000000640860401000000900027209000000"
        solution = self._get_solution(board_repr)
        # TODO: set up return value correctly as specified in solver.py
        ret = []
        for idx, c in enumerate(board_repr):
            if c != solution[idx]: # solution differs
                im = self._getTemplate(solution[idx])
                ret.append((self._idx2ij(idx), im))
                IPython.embed()
        return (ret, 0.5, 0.99)

    def _getTemplate(self, num):
        if num not in self.templates:
            self.templates[num] = cv2.imread('../images/' + num + '.png', -1)[:,:,:4]
        return self.templates[num]

    def _idx2ij(self, idx):
        idx = int(idx)
        return idx / 9, idx % 9

    def _detect(self, board):
        pass

    def _get_solution(self, board_repr):
        """
        The board is represented as a string of digits from 0 through 9,
        where 0 represents a blank space.
        """
        a = board_repr
        zeros = [m.start(0) for m in re.finditer("0", a)]
        a = list(a)
        zeros_index = 0
        same_info = {}
        for i in zeros:
            same_info[i] = []
            for j in range(81):
                if self.same_row(i, j) or self.same_col(i, j) or self.same_block(i, j):
                    same_info[i].append(j)
        while True:
            if zeros_index > len(zeros) - 1:
                return "".join(a)
            i = zeros[zeros_index]
            excluded_numbers = set(a[j] for j in same_info[i])
            for m in "123456789":
                if m not in excluded_numbers and a[i] < m:
                    a[i] = m
                    break
            else:
                a[i] = "0"
                zeros_index -= 1
                continue
            zeros_index += 1

    def same_row(self, i, j):
        return (i/9 == j/9)

    def same_col(self, i, j):
        return (i-j) % 9 == 0

    def same_block(self, i, j):
        return (i/27 == j/27 and i%9/3 == j%9/3)
