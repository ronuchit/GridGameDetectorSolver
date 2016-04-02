import numpy as np
import time
import cv2
from image_grabber import WebcamImageGetter, OpenNIImageGetter
from solvers.chess_solver import ChessSolver
from solvers.sudoku_solver import SudokuSolver
from grid_detector import GridDetector
import os
import itertools

# if True, pull from webcam; if False, run OpenNI with ROS (used for PrimeSense)
WEBCAM = True
# scale of displayed video
DISP_SCALE = 0.7
# FPS of video
FPS = 15
# to add new game solvers, import them above and add them to this list
SOLVERS = [ChessSolver, SudokuSolver]
# acceptance probability threshold
ACCEPT_PROB_THRESH = 0.9

class Runner(object):
    def run(self):
        print "Press enter to quit."
        # start image thread
        if WEBCAM:
            w = WebcamImageGetter(disp_scale=DISP_SCALE)
        else:
            w = OpenNIImageGetter(disp_scale=DISP_SCALE)
        w.start()
        self.gd = GridDetector()

        solvers = {}
        for s_c in SOLVERS:
            solver = s_c()
            # start solver thread
            solver.start()
            solvers[solver.GAME_NAME] = solver

        self.iteration = 0
        while True:
            self.iteration += 1
            # poll image thread at FPS rate
            time.sleep(1.0 / FPS)
            # frame = w.getFrame()
            frame = cv2.resize(cv2.imread("../images/sample_sudoku.jpg"), dsize=(0, 0), fx=DISP_SCALE, fy=DISP_SCALE)
            if frame is None:
                continue
            board, height, width = self._process_frame(frame)
            for s_name, s in solvers.items():
                if height == s.HEIGHT and width == s.WIDTH:
                    if s.board_to_use is None:
                        # if this solver thread is not already running,
                        # start up detect_and_play
                        s.board_to_use = board
                    # check if there's a solution saved for this game, from a previous solve
                    if s.result is not None:
                        solution, alpha, prob = s.result
                        if prob > ACCEPT_PROB_THRESH:
                            # accept the solution
                            print "Accepted solution for game %s: %s, with alpha = %s and prob = %s."%(s.GAME_NAME, solution, alpha, prob)
                            frame = self._do_overlay(frame, solution, alpha)
                            break

            # display frame
            cv2.imshow("frame", frame)
            # handle quitting: stop threads and close imshow windows
            if cv2.waitKey(1) & 0xFF == 10:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                w.keep_going = False
                w.end()
                for s_name, s in solvers.items():
                    s.keep_going = False
                break

    def _process_frame(self, frame):
        """
        Returns a numpy array of the board squares, to be passed into Solver.detect_and_play,
        along with its height and width.
        """
        col_inds, row_inds = self.gd.detect(frame)
        col_inds = sorted(col_inds)
        row_inds = sorted(row_inds)
        width = len(col_inds) - 1
        height = len(row_inds) - 1
        squares = []
        for i in range(len(col_inds) - 1):
            sq = []
            for j in range(len(row_inds) - 1):
                sq.append(frame[row_inds[j]:row_inds[j+1], col_inds[i]:col_inds[i+1]])
            squares.append(sq)
        return np.array(squares), height, width

    def _do_overlay(self, frame, solution, alpha):
        # TODO
        return frame

if __name__ == "__main__":
    Runner().run()
