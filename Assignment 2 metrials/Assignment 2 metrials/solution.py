"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""
        padded_right_image = np.pad(right_image, pad_width=((0, 0),
                                                            (dsp_range, dsp_range),
                                                            (0, 0)), mode='constant')
        for d in disparity_values:
            dispared_right_img = padded_right_image[:, d + dsp_range:np.shape(padded_right_image)[1] - (dsp_range - d), :]
            squared_diff = (left_image - dispared_right_img) ** 2
            pad_squared_diff = np.pad(squared_diff, ((1, 1), (1, 1), (0, 0)), mode='constant')
            # Sliding windows
            windows = np.lib.stride_tricks.sliding_window_view(pad_squared_diff,
                                                               (win_size, win_size, 1),
                                                               axis=(0, 1, 2))
            ssd_per_disparity = windows.sum(axis=(-1, -2, -3, -4))
            ssdd_tensor[:, :, d + dsp_range] = ssd_per_disparity


        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]), dtype=int)
        """INSERT YOUR CODE HERE"""
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        """INSERT YOUR CODE HERE"""
        l_slice[:, 0] = c_slice[:, 0]

        for col in range(1, num_of_cols):
            prev_col = l_slice[:, col - 1]

            # p1 penalty (±1 disparities), without wrap-around
            p1_cost = np.full(num_labels, np.inf)
            if num_labels > 1:
                p1_cost[1:] = prev_col[:-1]  # L[d-1]
                p1_cost[:-1] = np.minimum(p1_cost[:-1], prev_col[1:])  # L[d+1]
            p1_cost = p1 + p1_cost

            # p2 penalty (|k| >= 2 disparities)
            prefix = np.minimum.accumulate(prev_col)
            suffix = np.minimum.accumulate(prev_col[::-1])[::-1]
            p2_cost = np.zeros(num_labels)

            for d in range(num_labels):
                left_min = prefix[d - 2] if d - 2 >= 0 else np.inf
                right_min = suffix[d + 2] if d + 2 < num_labels else np.inf
                p2_cost[d] = p2 + min(left_min, right_min)

            # choose best route for each d
            M = np.minimum.reduce([prev_col, p1_cost, p2_cost])

            # recurrence
            l_slice[:, col] = c_slice[:, col] + M - np.min(prev_col)

        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""

        num_of_rows = ssdd_tensor.shape[0]
        for row in range(num_of_rows):
            hor_slice = ssdd_tensor[row, :, :].transpose()
            l[row, :, :] = self.dp_grade_slice(hor_slice, p1, p2).transpose()

        return self.naive_labeling(l)

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""


        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        H, W, D = ssdd_tensor.shape
        directions = range(1, 9)
        L_tensors = np.zeros((8, H, W, D))
        
        for idx, _direction in enumerate(directions):
            slices = self.extract_direction_slices(ssdd_tensor, _direction)
            
            # ניצור רשימה עבור תוצאות ה-DP של כל slice
            L_slices = []

            for path_slice in slices:
                # path_slice יכול להיות בגודל קצר יותר, DP צריך להתמודד עם זה
                L_result = self.dp_grade_slice(path_slice, p1, p2)
                L_slices.append(L_result)

            # מחזירים למבנה המקורי
            L_tensors[idx] = self.paths_to_ssdd_naive(L_slices, direction=_direction, original_shape=(H, W, D))

            # check_tensor= self.naive_labeling(L_tensors[idx])

            # print(_direction)
            # plt.figure()
            # plt.imshow(check_tensor)
            # plt.show()

        # ממוצע של כל 8 הכיוונים
        L_avg = np.mean(L_tensors, axis=0)

        return self.naive_labeling(L_avg)




    def extract_direction_slices(self, ssdd, direction):
        """
        Extract slices from SSDD for 8 directions (45° increments) in DP-ready format.
        Each slice is np.ndarray of shape (D, length).

        Directions mapping:
            1: →  left → right
            2: ↘  main diagonal TL→BR  (was 4)
            3: ↓  vertical top → bottom
            4: ↙  reversed anti-diagonal BL→TR  (was 6)
            5: ←  right → left
            6: ↖  reversed main diagonal BR→TL  (was 8)
            7: ↑  vertical bottom → top
            8: ↗  anti-diagonal TR→BL  (was 2)
        """

        H, W, D = ssdd.shape
        slices = []

        if direction == 1:  # horizontal left→right
            for i in range(H):
                path = ssdd[i, :, :].transpose(1, 0)  # (D, W)
                slices.append(path)

        elif direction == 2:  # main diagonal TL→BR
            for offset in range(-(H-1), W):
                diag = np.array([ssdd[i, i+offset, :] for i in range(max(0, -offset), min(H, W-offset))])
                if diag.size > 0:
                    slices.append(diag.transpose(1, 0))  # (D, length)

        elif direction == 3:  # vertical top→bottom
            for j in range(W):
                path = ssdd[:, j, :].transpose(1, 0)  # (D, H)
                slices.append(path)

        elif direction == 4:  # reversed anti-diagonal BL→TR
            for s in range(H + W - 1):
                diag = np.array([ssdd[i, s-i, :] for i in range(min(H-1, s), max(-1, s-W), -1)])
                if diag.size > 0:
                    slices.append(diag.transpose(1, 0))

        elif direction == 5:  # horizontal right→left
            for i in range(H):
                path = ssdd[i, ::-1, :].transpose(1, 0)  # (D, W)
                slices.append(path)

        elif direction == 6:  # reversed main diagonal BR→TL
            for offset in range(-(H-1), W):
                diag = np.array([ssdd[i, i+offset, :] for i in range(min(H-1, W-1-offset), max(-1, -offset-1), -1)])
                if diag.size > 0:
                    slices.append(diag.transpose(1, 0))

        elif direction == 7:  # vertical bottom→top
            for j in range(W):
                path = ssdd[::-1, j, :].transpose(1, 0)
                slices.append(path)

        elif direction == 8:  # anti-diagonal TR→BL
            for s in range(H + W - 1):
                diag = np.array([ssdd[i, s-i, :] for i in range(max(0, s-W+1), min(H, s+1))])
                if diag.size > 0:
                    slices.append(diag.transpose(1, 0))

        else:
            raise ValueError("Direction must be 1..8")

        return slices

    def paths_to_ssdd_naive(self, slices, direction, original_shape):
        """
        Reconstructs the 3D cost tensor from a list of 2D cost slices
        (D, length) after Dynamic Programming aggregation.
        """
        H, W, D = original_shape
        result = np.zeros((H, W, D))

        def place(i, j, vec):
            """Helper to place a D-vector at (i, j)"""
            # Note: slices are transposed to (D, length) in extract, 
            # but the DP output is usually (length, D) if the DP input was (length, D).
            # Assuming sl[k, :] is the D-vector for pixel k along the path.
            result[i, j, :] = vec

        if direction == 1:  # horizontal L→R: ssdd[i, :, :] -> slices along i
            # i is slice index, k is path index (j)
            for i, sl in enumerate(slices):
                for k in range(sl.shape[1]):
                    place(i, k, sl[:, k])

        elif direction == 5:  # horizontal R→L: ssdd[i, ::-1, :] -> slices along i
            # i is slice index, k is path index (W - 1 - j)
            for i, sl in enumerate(slices):
                for k in range(sl.shape[1]):
                    place(i, W - 1 - k, sl[:, k])

        elif direction == 3:  # vertical T→B: ssdd[:, j, :] -> slices along j
            # j is slice index, k is path index (i)
            for j, sl in enumerate(slices):
                for k in range(sl.shape[1]):
                    place(k, j, sl[:, k])

        elif direction == 7:  # vertical B→T: ssdd[::-1, j, :] -> slices along j
            # j is slice index, k is path index (H - 1 - i)
            for j, sl in enumerate(slices):
                for k in range(sl.shape[1]):
                    place(H - 1 - k, j, sl[:, k])

        # --- Diagonal Directions ---

        elif direction == 2:  # main diagonal TL→BR: i and j both increase
            # offset = j - i, offset ranges from -(H-1) to W-1
            # idx runs from 0 to W+H-2
            for idx, sl in enumerate(slices):
                offset = idx - (H - 1)
                # Starting point (k=0) is the one closest to (0, 0) for this offset
                i_start = max(0, -offset)
                j_start = max(0, offset)
                
                for k in range(sl.shape[1]):
                    i = i_start + k
                    j = j_start + k
                    place(i, j, sl[:, k])

        elif direction == 6:  # reversed main diagonal BR→TL: i and j both decrease
            # Same offsets as D2, but path is reversed (from BR to TL)
            for idx, sl in enumerate(slices):
                offset = idx - (H - 1)
                
                # Starting point (k=0) is the one closest to (H-1, W-1) for this offset
                i_start = min(H - 1, W - 1 - offset)
                j_start = min(W - 1, H - 1 + offset)
                
                for k in range(sl.shape[1]):
                    # Path indices decrease from start
                    i = i_start - k
                    j = j_start - k
                    place(i, j, sl[:, k])

        elif direction == 4:  # reversed anti-diagonal BL→TR: i decreases, j increases
            # s = i + j, s ranges from 0 to H+W-2 (same as idx)
            for s, sl in enumerate(slices):
                # The path in extract (D4) starts at the highest i (BL side) and decreases i
                # Starting point (k=0) for this sum 's'
                i_start = min(H - 1, s)
                j_start = s - i_start
                
                for k in range(sl.shape[1]):
                    # Path indices follow the slice generation (i decreases, j increases)
                    i = i_start - k
                    j = j_start + k
                    place(i, j, sl[:, k])

        elif direction == 8:  # anti-diagonal TR→BL: i increases, j decreases
            # s = i + j, s ranges from 0 to H+W-2
            for s, sl in enumerate(slices):
                # The path in extract (D8) starts at the lowest i (TR side) and increases i
                # Starting point (k=0) for this sum 's'
                i_start = max(0, s - W + 1)
                j_start = s - i_start
                
                for k in range(sl.shape[1]):
                    # Path indices follow the slice generation (i increases, j decreases)
                    i = i_start + k
                    j = j_start - k
                    place(i, j, sl[:, k])

        else:
            raise ValueError("Direction must be 1..8")

        return result