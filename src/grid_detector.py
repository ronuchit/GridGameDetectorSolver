import cv2
import numpy as np

VISUALIZE = False

class GridDetector(object):
    def detect(self, frame):
        h, w, d = frame.shape

        # TODO: get this working
        # frame_g = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # corners = cv2.cornerHarris(frame_g, 2, 3, 0.04)
        # print corners.max()
        # frame[corners > 0.01 * corners.max()] = [0, 0, 255]
        # cv2.imshow("image", frame)
        # cv2.waitKey(0)

        # detect highest-energy columns
        edges = self.get_vert_grad(frame)
        col_counts = self.get_col_counts(edges)
        max_col_indices = sorted(range(len(col_counts)), key=lambda k: col_counts[k], reverse=True)
        col_spaced = self.get_spaced_col_inds(max_col_indices)
        # TODO: fix this hack
        col_spaced = col_spaced[:10]
        if VISUALIZE:
            for col_idx in col_spaced:
                cv2.line(frame, (col_idx, 0), (col_idx, h), (0, 255, 0), 1)

        # detect highest-energy rows
        frame_r = np.rot90(frame)
        edges_r = self.get_vert_grad(frame_r)
        row_counts = self.get_col_counts(edges_r)
        max_row_indices = sorted(range(len(row_counts)), key=lambda k: row_counts[k], reverse=True)
        row_spaced = self.get_spaced_col_inds(max_row_indices)
        # TODO: fix this hack
        row_spaced = row_spaced[:10]
        if VISUALIZE:
            for row_idx in row_spaced:
                cv2.line(frame, (0, row_idx), (w, row_idx), (0, 0, 255), 1)

        if VISUALIZE:
            cv2.imshow("image", frame)
            cv2.waitKey(0)
        return col_spaced, row_spaced

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
