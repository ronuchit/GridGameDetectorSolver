"""
This file defines the generic interface for a game solver. You need to define 1) detection
and 2) what it means to "solve" your game. Solving could just mean finding the next best move, as in chess.
Implementation-wise, you just need to write your own detect_and_play().
"""

from threading import Thread
import time

class Solver(object):
    WIDTH = None
    HEIGHT = None
    GAME_NAME = ""

    def start(self):
        self.board_to_use = None
        self.result = None
        self.keep_going = True
        Thread(target=self.poll).start()

    def poll(self):
        while self.keep_going:
            if self.board_to_use is not None:
                self.result = self.detect_and_play(self.board_to_use)
                self.board_to_use = None
            time.sleep(0.1) # so that everything isn't super slow

    def detect_and_play(self, board):
        """
        Board is a height-by-width numpy array of RGB images of each square in the grid.
        This method should return the tuple (solution, alpha_blending_coefficient, prob), where prob is the probability
        that this board is of this game type. Solution is a list of (square index, RGB image to overlay) pairs.
        """
        raise NotImplementedError("Override this.")
