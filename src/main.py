import numpy as np
import time
import cv2
from image_grabber import WebcamImageGetter, OpenNIImageGetter
from solvers.chess_solver import ChessSolver
from solvers.sudoku_solver import SudokuSolver
import os

# if True, pull from webcam; if False, run OpenNI with ROS (used for PrimeSense)
WEBCAM = True
# scale of displayed video
DISP_SCALE = 0.7
# FPS of video
FPS = 15
# to add new game solvers, import them above and add them to this list
SOLVERS = [ChessSolver, SudokuSolver]

class Runner(object):
    def run(self):
        print "Press enter to quit."
        # start image thread
        if WEBCAM:
            w = WebcamImageGetter(disp_scale=DISP_SCALE)
        else:
            import rospy
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
            w = OpenNIImageGetter(disp_scale=DISP_SCALE)
        w.start()

        solvers = {}
        for s_c in SOLVERS:
            solver = s_c()
            # TODO: start solver thread
            solvers[solver.GAME_NAME] = solver

        while True:
            # poll image thread at FPS rate
            time.sleep(1.0 / FPS)
            frame = w.getFrame()
            if frame is None:
                continue
            height, width, squares = self._process_frame()
            for s_name, s in solvers.items():
                if height == s.HEIGHT and width == s.WIDTH:
                    # TODO: run detectAndPlay for that solver thread, if not already running
                    # read solution, perform overlay if accepted, imshow() that instead of frame
                    # if solution accepted, break
                    pass
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == 10:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                w.keep_going = False
                w.end()
                break

    def _process_frame(self):
        # TODO
        return 9, 9, None

if __name__ == "__main__":
    Runner().run()
