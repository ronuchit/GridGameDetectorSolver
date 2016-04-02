"""
This file defines the generic interface for a game solver. You need to define 1) detection
and 2) what it means to "solve" your game. Solving could just mean finding the next best move, as in chess.
"""

class Solver(object):
    def detect_and_play(self, board):
        """
        Board is a height-by-width numpy array of RGB images of each square in the grid.
        This method should return the tuple (solution, alpha_blending_coefficient, prob), where prob is the probability
        that this board is of this game type. Solution is a list of (square index, RGB image to overlay) pairs.
        """
        raise NotImplementedError("Override this.")
