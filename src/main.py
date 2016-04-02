import numpy as np
import time
import cv2
from image_grabber import WebcamImageGetter, OpenNIImageGetter
from solvers.sudoku_solver import SudokuSolver
from grid_detector import GridDetector
from scipy import misc
import os
import itertools

# if True, pull from webcam; if False, run OpenNI with ROS (used for PrimeSense)
WEBCAM = True
# scale of displayed video
DISP_SCALE = 0.7
# FPS of video
FPS = 15
# to add new game solvers, import them above and add them to this list
SOLVERS = [SudokuSolver]
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
            board, square_coordinates, height, width = self._process_frame(frame)
            for s_name, s in solvers.items():
                if height == s.HEIGHT and width == s.WIDTH:
                    if s.board_to_use is None:
                        # if this solver thread is not already running,
                        # start up detect_and_play
                        s.board_to_use = board
                    # check if there's a solution saved for this game, from a previous solve
                    if s.result is not None:
                        print s_name
                        solution, alpha, prob = s.result
                        if prob > ACCEPT_PROB_THRESH:
                            # accept the solution
                            # print "Accepted solution for game %s: %s, with alpha = %s and prob = %s."%(s.GAME_NAME, solution, alpha, prob)
                            frame = self._do_overlay(frame, solution, alpha, board, square_coordinates)
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
        col_inds, row_inds, frame = self.gd.detect(frame)
        col_inds = sorted(col_inds)
        row_inds = sorted(row_inds)
        width = len(col_inds) - 1
        height = len(row_inds) - 1
        squares = []
        square_coordinates = []
        for i in range(len(col_inds) - 1):
            sq = []
            coordinates = []
            for j in range(len(row_inds) - 1):
                sq.append(frame[row_inds[j]:row_inds[j+1], col_inds[i]:col_inds[i+1]])
                coordinates.append([row_inds[j], row_inds[j+1], col_inds[i], col_inds[i+1]])
            squares.append(sq)
            square_coordinates.append(coordinates)
        return np.array(squares), np.array(square_coordinates), height, width

    def _do_overlay(self, frame, solution, alpha, board, square_coordinates):
        for answer in solution:
            index = answer[0]
            ans_img = answer[1]
            coord_range = square_coordinates[index[1], index[0]]
            src_img = board[index[1], index[0]]
            height = coord_range[1] - coord_range[0]
            width = coord_range[3] - coord_range[2]
            ans_img = misc.imresize(ans_img, (height, width, 3))
            src_img = misc.imresize(src_img, (height, width, 3))
            import IPython
            IPython.embed()
            dst_img = np.where(ans_img < 20, ans_img, src_img)
            frame[coord_range[0]:coord_range[1], coord_range[2]: coord_range[3]] = dst_img
        return frame

if __name__ == "__main__":
    Runner().run()
