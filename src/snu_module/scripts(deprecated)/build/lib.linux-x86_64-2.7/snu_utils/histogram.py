"""
SNU Integrated Module v2.0
  - Code for histogram
"""
# Import Modules
import numpy as np

# Import Source Modules
import general_functions as gfuncs


# Get Weighted Depth Histogram
def get_depth_histogram(depth_patch, dhist_bin, min_value, max_value):
    np.set_printoptions(threshold=np.inf)
    patch_max_value = np.max(depth_patch.flatten())
    if patch_max_value < 0:
        depth_hist = np.zeros(dhist_bin, dtype=int)
        depth_idx = np.zeros(dhist_bin+1, dtype=float)
    else:
        gauss_mean = 0
        gauss_stdev = 0.6

        # Gaussian Window for Histogram Counting (Maximal at the center)
        gauss_window = gfuncs.generate_gaussian_window(depth_patch.shape[0], depth_patch.shape[1],
                                                       gauss_mean, gauss_stdev, min_val=1, max_val=10)
        #########################
        # Weighted Histogram
        depth_hist, depth_idx = np.histogram(depth_patch, bins=dhist_bin, range=(min_value, max_value), weights=gauss_window)
        # depth_hist, depth_idx = np.histogram(depth_patch, bins=dhist_bin, range=(min_value, max_value))

    return depth_hist, depth_idx













