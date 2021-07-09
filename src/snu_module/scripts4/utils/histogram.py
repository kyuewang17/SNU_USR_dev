"""
SNU Integrated Module v5.0

"""
import numba
import numpy as np


@numba.jit(nopython=True)
def get_bin_edges(a, bins, min_val=None, max_val=None):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min() if min_val is None else min_val
    a_max = a.max() if max_val is None else max_val
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@numba.jit(nopython=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@numba.jit(nopython=True)
def numba_histogram(a, bins, min_val=None, max_val=None):
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins, min_val=min_val, max_val=max_val)

    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges


# Get Weighted Histogram of Pixel Values of a Sensor Patch
def histogramize_patch(sensor_patch, dhist_bin, min_value, max_value):
    if sensor_patch is [] or (sensor_patch.shape[0] == 0 or sensor_patch.shape[1] == 0):
        hist, idx = [], []
    else:
        sensor_patch_vec = sensor_patch.flatten()
        patch_max_value = sensor_patch_vec.max()
        if patch_max_value < 0:
            hist = np.zeros(dhist_bin, dtype=int)
            idx = np.zeros(dhist_bin+1, dtype=float)
        else:
            hist, idx = numba_histogram(sensor_patch_vec, dhist_bin, min_val=min_value, max_val=max_value)
    return hist, idx