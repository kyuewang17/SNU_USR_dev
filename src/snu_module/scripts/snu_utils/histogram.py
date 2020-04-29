"""
SNU Integrated Module v2.0
  - Code for histogram
"""
# Import Modules
import numpy as np
import matplotlib.pyplot as plt

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


# Get Weighted Histogram of Pixel Values of a Sensor Patch
def histogramize_patch(sensor_patch, dhist_bin, min_value, max_value, count_window=None):
    if sensor_patch is []:
        hist, idx = [], []
    else:
        # Check for Channel Dimension of the Patch
        if len(sensor_patch.shape) != 2:
            assert 0, "Patch Dimension Error!"

        np.set_printoptions(threshold=np.inf)
        patch_max_value = np.max(sensor_patch.flatten())
        if patch_max_value < 0:
            hist = np.zeros(dhist_bin, dtype=int)
            idx = np.zeros(dhist_bin+1, dtype=float)
        else:
            # Weighted Histogram
            if count_window is not None:
                hist, idx = np.histogram(sensor_patch, bins=dhist_bin, range=(min_value, max_value), weights=count_window)
            else:
                hist, idx = np.histogram(sensor_patch, bins=dhist_bin, range=(min_value, max_value))

    return hist, idx


# Plot Histogram
def plot_histogram__(hist, idx, xlabel=None, ylabel=None):
    bin_median_list = []
    for bin_idx, bin_val in enumerate(idx):
        if bin_idx == len(idx)-1:
            continue
        else:
            bin_median_list.append((bin_val+idx[bin_idx+1])*0.5)
    hist_list = hist.tolist()
    plt.bar(bin_median_list, hist_list, align="center")
    plt.axis([min(bin_median_list), max(bin_median_list), 0, max(hist_list)])
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()
    plt.isinteractive()


# Plot Histogram (for real-time plot)
def plot_histogram():
    """
    https://github.com/nrsyed/computer-vision/blob/master/real_time_histogram/real_time_histogram.py
    :return:
    """
    return









