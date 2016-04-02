"""
This file defines the solver for chess.
"""

from solver import Solver
import time
import chess as pychess
import chess.uci as pychess_uci
import numpy as np
import cv2
import scipy.spatial as spatial
import scipy.cluster as clstr
from time import time
from collections import defaultdict
from functools import partial
import glob
import caffe
import skimage
import pickle

STOCKFISH_PATH = "../../lib/stockfish-6-mac/src/stockfish"
TIMEOUT_MS = 2000
CAFFENET_DEPLOY_TXT = '../../caffemodels/deploy.prototxt'
CAFFENET_MODEL_FILE = '../../caffemodels/finetune_chess_iter_5554.caffemodel'

categories = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'empty', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']
BATCH_SIZE = 64


class ChessSolver(Solver):
    WIDTH = 8
    HEIGHT = 8
    GAME_NAME = "chess"

    def start(self):
        # Caffe net setup
        net = caffe.Net(CAFFENET_DEPLOY_TXT, CAFFENET_MODEL_FILE, caffe.TEST)
        # Set up transformer for input data
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.array([104.00698793, 116.66876762, 122.67891434]));
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        net.blobs['data'].reshape(BATCH_SIZE, 3, 227, 227)

        Solver.start(self)

    """
    `board` is an 8x8 list of images
    """
    def detect_and_play(self, board):
        pieces, prob = self._detect(board)
        move = self._get_next_move(pieces)
        # TODO: set up return value correctly as specified in solver.py
        time.sleep(2)
        # solution, alpha = 0.5, prob
        return ("chess_solution_%s"%board, 0.5, prob)

    def _detect(self, board):
        input_images = [transformer.preprocess('data', skimage.img_as_float(square).astype(np.float32)) for square in row for row in board]
        net.blobs['data'].data[...] = np.array(input_images)
        out = net.forward()['prob']
        predictions = np.asarray(map(lambda x: categories[x], np.argmax(out, axis=1)))
        prob = np.sum(np.max(out, axis=1))
        return np.reshape(predictions, (8, 8)), prob


    def _get_next_move(self, pieces):
        """
        Pieces is a 8x8 numpy array of pieces from the
        list ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']
        or None if no piece.
        """
        board = pychess.Board()
        board.set_fen(self._get_board_string(pieces))
        engine = pychess_uci.popen_engine(STOCKFISH_PATH)
        engine.uci()
        engine.setoption({"UCI_Chess960": True})
        engine.info_handlers.append(pychess_uci.InfoHandler())
        engine.position(board)
        engine.go(movetime=TIMEOUT_MS, async_callback=True)
        time.sleep(TIMEOUT_MS / 1000)
        ih = engine.info_handlers[0]
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
#     pieces = ['b', 'n', 'r'] * 2 + ['k', 'q'] + ['p'] * 8 + ['B', 'N', 'R'] * 2 + ['K', 'Q'] + ['P'] * 8 + [None] * 32
#     random_pieces = np.reshape(np.random.choice(pieces, 64), [8,8])
#     return random_pieces
#     print random_pieces
