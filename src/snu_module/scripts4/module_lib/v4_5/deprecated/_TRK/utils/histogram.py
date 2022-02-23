"""
SNU Integrated Module v5.0
  - Code for Histogram

"""
import numpy as np
from fast_histogram import histogram1d


def histogramize(src, N, hist_range=None, count_weights=None):
    # Set Range
    if hist_range is None:
        hist_range = (np.min(src), np.max(src))
    else:
        assert isinstance(hist_range, (list, tuple)) and len(hist_range) == 2 and hist_range[0] < hist_range[1]
        hist_range = tuple(hist_range)

    if count_weights is None:
        hist = histogram1d(src, bins=N, range=hist_range)
    else:
        hist = histogram1d(src, bins=N, range=hist_range, weights=count_weights)
    idx = None

    return hist, idx


# TODO: Implement Histogram Index (idx) -- fast_histogram does not return this compared to np.histogram
def histogramize_patch(sensor_patch, dhist_bin, min_value, max_value, count_window=None, linearize=False):
    if sensor_patch is [] or (sensor_patch.shape[0] == 0 or sensor_patch.shape[1] == 0):
        hist, idx = [], None
    else:
        # Check for Counting Window Shape (row/col)
        if count_window is not None:
            assert sensor_patch.shape[0] == count_window.shape[0] and sensor_patch.shape[1] == count_window.shape[1]
            count_weights = count_window.flatten()
        else:
            count_weights = None

        # Array Print Options (not sure if this is needed)
        np.set_printoptions(threshold=np.inf)

        # Get Number of Patch Channels
        assert len(sensor_patch.shape) in [2, 3]
        C = sensor_patch.shape[2] if len(sensor_patch.shape) == 3 else 1

        # this case...
        if np.max(sensor_patch) < min_value:
            hist = np.zeros(dhist_bin, dtype=int)
            idx = np.zeros(dhist_bin + 1, dtype=float)

        # Get Histogram
        if C == 1:
            flattened_sensor_patch = sensor_patch.flatten()
            hist, idx = histogramize(flattened_sensor_patch, dhist_bin, (min_value, max_value), count_weights)
        else:
            hist_list, idx_list = [], []
            for channel_idx in range(C):
                flattened_sensor_patch = sensor_patch[:, :, channel_idx].flatten()
                hist, idx = histogramize(flattened_sensor_patch, dhist_bin, (min_value, max_value), count_weights)
                hist_list.append(hist)
                idx_list.append(idx)
            hist, idx = np.vstack(hist_list), np.vstack(idx_list)
            if linearize is True:
                hist, idx = hist.reshape(-1), idx.reshape(-1)

    # Return
    return hist, idx


if __name__ == "__main__":
    pass
