"""
SNU Integrated Module v5.0

"""
import numpy as np

import numba
from numba import cuda


@numba.jit(nopython=True)
def compute_bin(x, n, xmin, xmax):
    # special case to mirror NumPy behavior for last bin
    if x == xmax:
        return n - 1 # a_max always in last bin

    # SPEEDTIP: Remove the float64 casts if you don't need to exactly reproduce NumPy
    bin = np.int32(n * (np.float64(x) - np.float64(xmin)) / (np.float64(xmax) - np.float64(xmin)))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin

@cuda.jit
def histogram(x, xmin, xmax, histogram_out):
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, x.shape[0], stride):
        # note that calling a numba.jit function from CUDA automatically
        # compiles an equivalent CUDA device function!
        bin_number = compute_bin(x[i], nbins, xmin, xmax)

        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            cuda.atomic.add(histogram_out, bin_number, 1)


@cuda.jit
def min_max(x, min_max_array):
    nelements = x.shape[0]

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Array already seeded with starting values appropriate for x's dtype
    # Not a problem if this array has already been updated
    local_min = min_max_array[0]
    local_max = min_max_array[1]

    for i in range(start, x.shape[0], stride):
        element = x[i]
        local_min = min(element, local_min)
        local_max = max(element, local_max)

    # Now combine each thread local min and max
    cuda.atomic.min(min_max_array, 0, local_min)
    cuda.atomic.max(min_max_array, 1, local_max)


def dtype_min_max(dtype):
    '''Get the min and max value for a numeric dtype'''
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    else:
        info = np.finfo(dtype)
    return info.min, info.max


@numba.jit(nopython=True)
def get_bin_edges(a, nbins, a_min, a_max):
    bin_edges = np.empty((nbins+1,), dtype=np.float64)
    delta = (a_max - a_min) / nbins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


def numba_gpu_histogram(a, bins):
    # Move data to GPU so we can do two operations on it
    a_gpu = cuda.to_device(a)

    ### Find min and max value in array
    dtype_min, dtype_max = dtype_min_max(a.dtype)
    # Put them in the array in reverse order so that they will be replaced by the first element in the array
    min_max_array_gpu = cuda.to_device(np.array([dtype_max, dtype_min], dtype=a.dtype))
    min_max[64, 64](a_gpu, min_max_array_gpu)
    a_min, a_max = min_max_array_gpu.copy_to_host()

    # SPEEDTIP: Skip this step if you don't need to reproduce the NumPy histogram edge array
    bin_edges = get_bin_edges(a, bins, a_min, a_max) # Doing this on CPU for now

    ### Bin the data into a histogram
    histogram_out = cuda.to_device(np.zeros(shape=(bins,), dtype=np.int32))
    histogram[64, 64](a_gpu, a_min, a_max, histogram_out)

    return histogram_out.copy_to_host(), bin_edges


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
            # hist, idx = numba_histogram(sensor_patch_vec, dhist_bin, min_val=min_value, max_val=max_value)
            hist, idx = numba_gpu_histogram(sensor_patch_vec, dhist_bin)
    return hist, idx