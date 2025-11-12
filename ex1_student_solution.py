"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # return homography
        """INSERT YOUR CODE HERE"""
        ### build A ###
        num_of_points = match_p_dst[0].shape[0]
        # point_matrix = np.zeros(9,num_of_points*2)
        point_matrix = []

        for i in range(num_of_points):
            x,y = match_p_src[0][i],match_p_src[1][i]
            x_hat,y_hat = match_p_dst[0][i],match_p_dst[1][i]

            row_1 = [ -x, -y, -1, 0, 0, 0, x*x_hat, y*x_hat, x_hat ]
            row_2 = [ 0,  0,  0, -x, -y, -1, x*y_hat, y*y_hat, y_hat]
            
            point_matrix.append(row_1)
            point_matrix.append(row_2)

        point_matrix = np.array(point_matrix)
        #SVD
        u_mat, s_mat, vt_mat = np.linalg.svd(point_matrix)
        h = vt_mat[-1, :].reshape(3,3)
        h = h/h[2,2]



        return h

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        new_img = np.zeros(dst_image_shape,dtype =src_image.dtype)
        r_src_shape , c_src_shape = src_image.shape[:2]

        for y in range(r_src_shape):
            for x in range(c_src_shape):
                
                src_pt = np.array([x,y,1.0])

                dst_pt_h = homography @ src_pt

                dst_pt_x = round(dst_pt_h[0]/dst_pt_h[2])
                dst_pt_y = round(dst_pt_h[1]/dst_pt_h[2])

                if 0 <= dst_pt_x < dst_image_shape[1] and 0 <= dst_pt_y < dst_image_shape[0]:
                    new_img[dst_pt_y, dst_pt_x] = src_image[y, x]

        return new_img

        # return new_image
        

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""

        src_H = src_image.shape[0]
        src_W = src_image.shape[1]

        dst_H = dst_image_shape[0]
        dst_W = dst_image_shape[1]

        # mesh 
        x = np.arange(src_W)
        y = np.arange(src_H)
        xv, yv = np.meshgrid(x, y) 

        xv_flat = xv.flatten()
        yv_flat = yv.flatten()
        ones = np.ones_like(xv_flat)
        src_pts_h = np.vstack([xv_flat, yv_flat, ones])  # shape: 3 x (H*W)

        # homography
        dst_pts_h = homography @ src_pts_h  # shape: 3 x 3*(H*W)
        dst_pts_h /= dst_pts_h[2, :] #normalize        

        dst_x = np.round(dst_pts_h[0, :]).astype(int)
        dst_y = np.round(dst_pts_h[1, :]).astype(int)
        dst_x = np.clip(dst_x, 0, dst_W-1)
        dst_y = np.clip(dst_y, 0, dst_H-1)

        # final image
        dst_image = np.zeros(dst_image_shape, dtype=src_image.dtype)
        dst_image[dst_y, dst_x] = src_image[yv_flat, xv_flat]

        return dst_image
        

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # return fit_percent, dist_mse
        """INSERT YOUR CODE HERE"""
        dist_mse = 10 ** 9
        # apply homography to match points in source image
        estimated_p_dst = np.matmul(homography, np.vstack([match_p_src, np.ones((1, match_p_src.shape[1]))]))
        estimated_p_dst = estimated_p_dst[:2, :] / estimated_p_dst[2, :]

        distances = np.linalg.norm(estimated_p_dst - match_p_dst, axis=0)

        # find inliers based on maximum error allowed
        is_inlier = distances < max_err

        # calculate fit_percent and mean squared error for inliers
        n_inliers = np.sum(is_inlier)
        total_points = match_p_src.shape[1]
        fit_percent = n_inliers / total_points

        if n_inliers:
            dist_mse = np.sum(distances[is_inlier]**2) / total_points

        return fit_percent, dist_mse


    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # return mp_src_meets_model, mp_dst_meets_model
        """INSERT YOUR CODE HERE"""
        # apply homography to match points in source image
        estimated_p_dst = np.matmul(homography, np.vstack([match_p_src, np.ones((1, match_p_src.shape[1]))]))
        estimated_p_dst = estimated_p_dst[:2, :] / estimated_p_dst[2, :]

        distances = np.linalg.norm(estimated_p_dst - match_p_dst, axis=0)
        is_inlier = distances < max_err

        return match_p_src[:, is_inlier], match_p_dst[:, is_inlier]

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # use class notations:
        w = inliers_percent
        # t = max_err
        # p = parameter determining the probability of the algorithm to
        # succeed
        p = 0.99
        # the minimal probability of points which meets with the model
        d = 0.5
        # number of points sufficient to compute the model
        n = 4
        # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1

        """INSERT YOUR CODE HERE"""

        min_score = np.inf
        output_homography = np.zeros(shape=(3, 3))

        for i in range(k):
            rand_n = sample(range(0, match_p_src.shape[1]), n)
            rand_p_src = match_p_src[:, rand_n]
            rand_p_dst = match_p_dst[:, rand_n]

            homography = self.compute_homography_naive(rand_p_src, rand_p_dst)
            fit_percent, dist_mse = self.test_homography(homography, match_p_src, match_p_dst, max_err)
            if fit_percent > d:
                inliers_p_src, inliers_p_dst = (
                    self.meet_the_model_points(homography, match_p_src, match_p_dst, max_err))
                inliers_homography = self.compute_homography_naive(inliers_p_src, inliers_p_dst)
                fit_percent, dist_mse = self.test_homography(inliers_homography, inliers_p_src, inliers_p_dst, max_err)
                if dist_mse < min_score:
                    output_homography = inliers_homography
                    min_score = dist_mse

        output_homography /= output_homography[2,2]

        return output_homography
    
    # @staticmethod
    # def compute_backward_mapping(
    #         backward_projective_homography: np.ndarray,
    #         src_image: np.ndarray,
    #         dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
    #     """Compute backward mapping.

    #     (1) Create a mesh-grid of columns and rows of the destination image.
    #     (2) Create a set of homogenous coordinates for the destination image
    #     using the mesh-grid from (1).
    #     (3) Compute the corresponding coordinates in the source image using
    #     the backward projective homography.
    #     (4) Create the mesh-grid of source image coordinates.
    #     (5) For each color channel (RGB): Use scipy's interpolation.griddata
    #     with an appropriate configuration to compute the bi-cubic
    #     interpolation of the projected coordinates.

    #     Args:
    #         backward_projective_homography: 3x3 Projective Homography matrix.
    #         src_image: HxWx3 source image.
    #         dst_image_shape: tuple of length 3 indicating the destination shape.

    #     Returns:
    #         The source image backward warped to the destination coordinates.
    #     """

    #     # return backward_warp
    #     """INSERT YOUR CODE HERE"""
    #     # Placeholder
    #     backward_warp = np.zeros(shape=dst_image_shape)
    #     intermediate = np.zeros(shape=src_image.shape)

    #     # Create a meshgrid and reshape to a vector with a third row of 1's:
    #     xx, yy = np.meshgrid(range(dst_image_shape[1]), range(dst_image_shape[0]))
    #     combined = np.stack((xx, yy, np.ones(dst_image_shape[:2])))
    #     combined = combined.reshape((3, -1))

    #     # Transform all points using homography:
    #     coord = np.matmul(backward_projective_homography, combined)

    #     coord = (coord / coord[2])[:2, :]
    #     # coord2 = coord.astype(int)
    #     copy_data = np.append(coord, combined[:2, :], axis=0)

    #     # Note that x,y in camera coordinates is different from numpy axes!
    #     # Filter out all pixels outside of source coordinate range.
    #     # Cropped the outer ring so that bilinear can be calculated!

    #     dropout = copy_data.min(axis=0) >= 0
    #     copy_data = copy_data[:, dropout]
    #     dropout = copy_data[0, :] < src_image.shape[1]
    #     copy_data = copy_data[:, dropout]
    #     dropout = copy_data[1, :] < src_image.shape[0]
    #     copy_data = copy_data[:, dropout]

    #     indices = np.floor(copy_data[0:2, :]).astype(int)
    #     values = src_image[indices[1, :], indices[0, :], :]
    #     backward_warp = griddata(copy_data[2:, :].transpose(), values, (xx, yy))

    #     return np.round(backward_warp).astype(int)

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping (Backward Warping).

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix (H_inv).
            src_image: H_src x W_src x 3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape (H_dst, W_dst, C).

        Returns:
            The source image backward warped to the destination coordinates (H_dst x W_dst x 3).
        """

        # 1. & 2. Create destination coordinates and homogeneous coordinates vector
        dst_height, dst_width, _ = dst_image_shape
        
        # Create meshgrid for destination coordinates (column index, row index)
        dst_col_indices, dst_row_indices = np.meshgrid(range(dst_width), range(dst_height))
        
        dst_homogeneous_coords = np.stack(
            (dst_col_indices.ravel(), dst_row_indices.ravel(), np.ones(dst_height * dst_width))
        )  # Shape (3, N)

        # 3. Compute corresponding coordinates in the source image (Forward Projection)
        src_homogeneous_coords = np.matmul(backward_projective_homography, dst_homogeneous_coords) # Shape (3, N)
        
        # Normalize coordinates (divide by the third component) to get (x_src, y_src)
        src_norm_coords = src_homogeneous_coords / src_homogeneous_coords[2, :] # Shape (3, N)
        
        # Extract non-homogeneous source coordinates (x_src, y_src)
        src_coords = src_norm_coords[:2, :] # Shape (2, N)

        # 4. Prepare data for interpolation and filter out invalid coordinates
        all_coords = np.append(src_coords, dst_homogeneous_coords[:2, :], axis=0) # Shape (4, N)
        src_height, src_width, _ = src_image.shape
        
        # Filter points outside the source image boundaries (0 <= x_src < W_src, 0 <= y_src < H_src)
        valid_mask = (all_coords[0, :] >= 0) & (all_coords[0, :] < src_width) & \
                     (all_coords[1, :] >= 0) & (all_coords[1, :] < src_height) & \
                     (all_coords[1, :] >= 0) 
        valid_coords = all_coords[:, valid_mask] # Shape (4, N_valid)
        
        # 5. Interpolation using griddata
        src_col_indices_int = np.floor(valid_coords[0, :]).astype(int)
        src_row_indices_int = np.floor(valid_coords[1, :]).astype(int)

        sampled_colors = src_image[src_row_indices_int, src_col_indices_int, :] # Shape (N_valid, 3)
        dst_coords_for_griddata = valid_coords[2:, :].T # Shape (N_valid, 2) - (x_dst, y_dst)
        
        destination_grid = (dst_col_indices, dst_row_indices) 
        
        # Perform interpolation (bi-cubic)
        backward_warp_float = griddata(
            points=dst_coords_for_griddata, # The valid destination coordinates (x_dst, y_dst)
            values=sampled_colors,          # The sampled color values from the source image
            xi=destination_grid,            # The full grid (xx, yy) to interpolate onto
            method='linear'                  # Using 'cubic' for bi-cubic interpolation
            ) # Shape (dst_height, dst_width, 3)
        
        backward_warp_float[np.isnan(backward_warp_float)] = 0
        backward_warp = np.round(backward_warp_float).astype(np.uint8) # Use uint8 for image data

        return backward_warp
    # @staticmethod
    # def compute_backward_mapping(
    #         backward_projective_homography: np.ndarray,
    #         src_image: np.ndarray,
    #         dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
    #     """Compute backward mapping.

    #     (1) Create a mesh-grid of columns and rows of the destination image.
    #     (2) Create a set of homogenous coordinates for the destination image
    #     using the mesh-grid from (1).
    #     (3) Compute the corresponding coordinates in the source image using
    #     the backward projective homography.
    #     (4) Create the mesh-grid of source image coordinates.
    #     (5) For each color channel (RGB): Use scipy's interpolation.griddata
    #     with an appropriate configuration to compute the bi-cubic
    #     interpolation of the projected coordinates.

    #     Args:
    #         backward_projective_homography: 3x3 Projective Homography matrix.
    #         src_image: HxWx3 source image.
    #         dst_image_shape: tuple of length 3 indicating the destination shape.

    #     Returns:
    #         The source image backward warped to the destination coordinates.
    #     """
    #     dst_H = dst_image_shape[0]
    #     dst_W = dst_image_shape[1]
    #     src_H = src_image.shape[0]
    #     src_W = src_image.shape[1]

    #     # mesh 
    #     #dst
    #     x = np.arange(dst_W)
    #     y = np.arange(dst_H)
    #     xv, yv = np.meshgrid(x, y)
    #     xv_flat = xv.flatten()
    #     yv_flat = yv.flatten()
    #     ones = np.ones_like(xv_flat)
        
    #     #src
    #     src_y_grid, src_x_grid = np.indices(src_image.shape[:2])
    #     src_rows, src_cols = np.meshgrid(range(src_image.shape[0]), range(src_image.shape[1]))
    #     points_to_sample_from = (src_x_grid.flatten(), src_y_grid.flatten())

    #     # return backward_warp
    #     """INSERT YOUR CODE HERE"""
    #     backward_projective_homography = np.linalg.inv(backward_projective_homography)
    #     #matrix multipication
    #     dst_coordinates = np.vstack([xv_flat, yv_flat, ones])  # shape: 3 x (H*W)
    #     src_coordinates = backward_projective_homography @ dst_coordinates
    #     src_coordinates[:2, :] /= src_coordinates[2,:]

    #     src_x_map = src_coordinates[0, :]
    #     src_y_map = src_coordinates[1, :]

    #     backward_warp = np.zeros(dst_image_shape, dtype=np.uint8)

    #     for channel_num in range(src_image.shape[2]):
    #         channel_data = src_image[:, :, channel_num].flatten()
    #         warped_channel = griddata(  points_to_sample_from,
    #                                     channel_data,
    #                                     (src_x_map, src_y_map),
    #                                     method='cubic',
    #                                     fill_value=0
    #                                   )
    #         backward_warp[:, :, channel_num] = warped_channel.reshape(dst_H, dst_W)

    #     return backward_warp

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([0, 0, 1])
        src_edges['upper right corner'] = np.array([src_cols_num - 1, 0, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num - 1, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        pad_up = int(np.ceil(pad_up))
        pad_down = int(np.ceil(pad_down))
        pad_left = int(np.ceil(pad_left))
        pad_right = int(np.ceil(pad_right))  

        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homoraphy and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""
        translation = np.eye(3)
        translation[0, 2] = -pad_left
        translation[1, 2] = -pad_up
        final_homography = np.matmul(backward_homography, translation)
        # final_homography /= np.linalg.norm(final_homography)
        final_homography /= final_homography[2, 2]

        return final_homography

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        
        """INSERT YOUR CODE HERE"""

        #   (1) Compute the forward homography and the panorama shape.
        print('RANSAC Homography')
        ransac_homography = self.compute_homography(match_p_src,
                                                    match_p_dst,
                                                    inliers_percent,
                                                    max_err)
        
        
        print('calculating panorama shape and padding') 
        
        panorama_rows_num, panorama_cols_num, pad_struct = self.find_panorama_shape(src_image,
                                                                                    dst_image,
                                                                                    ransac_homography
                                                                                    )
        pad_left = pad_struct.pad_left
        pad_right = pad_struct.pad_right
        pad_down = pad_struct.pad_down
        pad_up = pad_struct.pad_up

        H_dst, W_dst, C = dst_image.shape
        panorama_shape = (panorama_rows_num, panorama_cols_num, C)

        #   (2) Compute the backward homography.
        print("calculate backward projective homography")

        backward_projective_homography = np.linalg.inv(ransac_homography)

        #   (3) Add the appropriate translation to the homography so that the 
        #   source image will plant in place.
        
        
        print("calculate backward projective homography with translation")

        final_homography = self.add_translation_to_backward_homography(backward_projective_homography,
                                                                        pad_left,
                                                                        pad_up
                                                                        )
        #   (4) Compute the backward warping with the appropriate translation.

        print("backward mapping")
        backward_warp = self.compute_backward_mapping(final_homography,
                                                      src_image,
                                                      dst_image_shape = panorama_shape
                                                      ) 
        
        #   (5) Create the an empty panorama image and plant there the
        #       destination image.
        print("panorama creation from padded destination image that have zeros , and source values that are n")
        dst_padded = np.pad(dst_image.astype(float),
                            ((pad_up, pad_down),
                             (pad_left, pad_right),
                             (0, 0)),
                            mode='constant', 
                            constant_values=0)
        
        img_panorama = dst_padded.copy()

        # #   (6) place the backward warped image in the indices where the panorama
        # #   image is zero.
        dst_black_mask = np.all(img_panorama == 0, axis=2)
        src_valid_mask = np.any(backward_warp != 0, axis=2)
        merge_mask = dst_black_mask & src_valid_mask
        img_panorama[merge_mask] = backward_warp[merge_mask]

        
        

        #   (7) Don't forget to clip the values of the image to [0, 255].
        
        return np.clip(img_panorama, 0, 255).astype(np.uint8)    
        
