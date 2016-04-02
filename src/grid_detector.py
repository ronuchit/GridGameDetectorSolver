import cv2
from scipy.interpolate import griddata
import math
import numpy as np
import IPython

VISUALIZE = False

class GridDetector(object):
    def detect(self, frame):
        h, w, d = frame.shape

        #Get corners
        frame_g = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        corners = cv2.cornerHarris(frame_g, 2, 3, 0.04)
        corner_max = corners.max()
        thresh_corners = []
        for i in range(1, h):
            for j in range(1, w):
                if corners[i][j] > corner_max * 0.15:
                    thresh_corners.append((i, j))

        # Get rows
        col_spaced, row_spaced = self.detect_lines(frame)
        #if VISUALIZE:
        #    for row_idx in row_spaced:
        #        cv2.line(frame, (0, row_idx), (w, row_idx), (0, 0, 255), 1)
        #    for col_idx in col_spaced:
        #        cv2.line(frame, (col_idx, 0), (col_idx, h), (0, 255, 0), 1)
        #    cv2.imshow("image", frame)
        #    cv2.waitKey(0)

        # Find the corners of the board
        min_row = np.min(row_spaced)
        max_row = np.max(row_spaced)
        min_col = np.min(col_spaced)
        max_col = np.max(col_spaced)
        upper_left = self.closest(thresh_corners, min_row, min_col)
        upper_right = self.closest(thresh_corners, min_row, max_col)
        lower_left = self.closest(thresh_corners, max_row, min_col)
        lower_right = self.closest(thresh_corners, max_row, max_col)

        corners = np.array([upper_left, upper_right, lower_right, lower_left])
        # Warp
        frame2 = self.four_point_transform(frame, corners)
        col_spaced, row_spaced = self.detect_lines(frame2)

        if VISUALIZE:
            for row_idx in row_spaced:
                cv2.line(frame2, (0, row_idx), (w, row_idx), (0, 0, 255), 1)
            for col_idx in col_spaced:
                cv2.line(frame2, (col_idx, 0), (col_idx, h), (0, 255, 0), 1)
            cv2.imshow("image", frame2)
            cv2.waitKey(0)
        return col_spaced, row_spaced, frame2

    def detect_lines(self, frame):
        # detect highest-energy columns
        edges = self.get_vert_grad(frame)
        col_counts = self.get_col_counts(edges)
        max_col_indices = sorted(range(len(col_counts)), key=lambda k: col_counts[k], reverse=True)
        col_spaced = self.get_spaced_col_inds(max_col_indices)
        # TODO: fix this hack
        col_spaced = col_spaced[:10]

        # detect highest-energy rows
        frame_r = np.rot90(frame)
        edges_r = self.get_vert_grad(frame_r)
        row_counts = self.get_col_counts(edges_r)
        max_row_indices = sorted(range(len(row_counts)), key=lambda k: row_counts[k], reverse=True)
        row_spaced = self.get_spaced_col_inds(max_row_indices)
        # TODO: fix this hack
        row_spaced = row_spaced[:10]
        return col_spaced, row_spaced



    def four_point_transform(self, img, source, square_length=504):
        destination = np.array([[0, 0], [0, square_length], [square_length, square_length], [square_length, 0]])
        gx, gy = np.mgrid[0:503:504j, 0:503:504j]
        grid_z = griddata(destination, source, (gx, gy), method='cubic')
        map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(square_length,square_length)
        map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(square_length,square_length)
        map_x_32 = map_x.astype('float32')
        map_y_32 = map_y.astype('float32')
        warped = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_CUBIC)
        return warped
        #M = cv2.getPerspectiveTransform(pts1, pts2)
        #im = cv2.warpPerspective(img, M, (square_length, square_length))
        #return im

    def distance(self, x1, y1, x2, y2):
        return math.sqrt( (x2 - x1) ** 2 + (y2 - y1) ** 2)

    def closest(self, corners, x, y):
        dists = [self.distance(corner[0], corner[1], x, y) for corner in corners]
        return corners[np.argmin(dists)]

    def get_vert_grad(self, img):
        sharp_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1,0,1]])
        edges = cv2.filter2D(img, -1, sharp_filter)
        sharp_filter = np.array([[1, 0, -1], [1, 0, -1], [1,0,-1]])
        edges_2 = cv2.filter2D(img, -1, sharp_filter)
        vert_edges = edges | edges_2
        return vert_edges

    def get_col_counts(self, edges, do_row=False):
        h, w, d = edges.shape
        col_counts = []
        for i in xrange(0, w):
            count = self._get_counts(edges, i, w, do_row)
            col_counts.append(count)
        return col_counts

    def _get_counts(self, edge_img, i, max_h, do_row=False):
        window_size = 2
        if (i - window_size/2 < 0):
            count = np.count_nonzero((edge_img[:, 0:i+window_size/2] == 255) - 0)
        elif (i + window_size/2 > max_h-1):
            count = np.count_nonzero((edge_img[:, i-window_size/2:] == 255) - 0)
        else:
            count = np.count_nonzero((edge_img[:, i-window_size/2:i+window_size/2] == 255) - 0)
        return count

    def get_spaced_col_inds(self, max_col_indices):
        spaced_col_inds = []
        for col_idx in max_col_indices:
            in_range = False
            for r_sp in spaced_col_inds:
                if (col_idx < r_sp + 25 and col_idx > r_sp - 25):
                    in_range = True
            if not in_range:
                spaced_col_inds.append(col_idx)
        return spaced_col_inds

if __name__ == "__main__":
    img = cv2.imread("../images/sample_sudoku.jpg")
    GridDetector().detect(img)
