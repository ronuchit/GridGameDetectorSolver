"""
This file defines the solver for chess.
"""

from solver import Solver
import time
import chess as pychess
import chess.uci as pychess_uci
import numpy as np

STOCKFISH_PATH = "../../lib/stockfish-6-mac/src/stockfish"
TIMEOUT_MS = 2000

class ChessSolver(Solver):
    WIDTH = 8
    HEIGHT = 8
    GAME_NAME = "chess"

    def __init__(self):
        self.board = pychess.Board()
        self.engine = pychess_uci.popen_engine(STOCKFISH_PATH)
        self.engine.uci()
        self.engine.setoption({"UCI_Chess960": True})
        self.engine.info_handlers.append(pychess_uci.InfoHandler())
        self.remap = {'bb': 'b', 'bk': 'k', 'bn': 'n', 'bp': 'p', 'bq': 'q', 'br': 'r', 'wb': 'B', 'wk': 'K', 'wn': 'N', 'wp': 'P', 'wq': 'Q', 'wr': 'R', None: None}


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
        pieces = np.reshape(np.array([self.remap[piece] for piece in pieces.flatten()]), [8, 8])
        self.board.set_fen(pieces)
        self.engine.position(self.board)
        self.engine.go(movetime=TIMEOUT_MS, async_callback=True)
        time.sleep(TIMEOUT_MS / 1000)
        ih = self.engine.info_handlers[0]
        best_move = ih.info["pv"][1][0]
        score = ih.info["score"][1].cp
        mate = ih.info["score"][1].mate
        return best_move

    def _get_board_string(self, pieces):
        s = ""
        blank_spaces = 0
        for i in range(pieces.shape[0]):
            for j in range(pieces.shape[1]):
                piece = pieces[i, j]
                if piece:
                    if blank_spaces:
                        s += str(blank_spaces)
                        blank_spaces = 0
                    s += piece
                else:
                    blank_spaces += 1
            if blank_spaces:
                        s += str(blank_spaces)
                        blank_spaces = 0
            if i < pieces.shape[0] - 1:
                s += '/'
        s += " w KQkq - 0 1"
        return s

# def random_board_generator():
#     pieces = ['bb', 'bn', 'br'] * 2 + ['bk', 'bq'] + ['bp'] * 8 + ['wb', 'wb', 'wr'] * 2 + ['wk', 'wq'] + ['wp'] * 8 + [None] * 32
#     random_pieces = np.reshape(np.random.choice(pieces, 64, replace=False), [8, 8])
#     return random_pieces

