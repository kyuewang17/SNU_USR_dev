"""
SNU Integrated Module v2.0
  - General Utility Functions
"""
# Future Import
from __future__ import print_function

# Import Libraries
import cv2
import numpy as np
from itertools import groupby


# Get Maximum Tracklet ID among all Tracklets
def get_maximum_id(trks, max_trk_id):
    if len(trks) != 0:
        max_id = -1
        for trk in trks:
            if trk.id >= max_id:
                max_id = trk.id
    else:
        max_id = max_trk_id

    return max_id


# Get Maximum Consecutive Values in a List
def get_max_consecutive(src_list, find_val):
    max_count = 0
    for val, sub_list in groupby(src_list):
        sub_list = list(sub_list)

        if val is find_val:
            if len(sub_list) >= max_count:
                max_count = len(sub_list)

    return max_count


# Select Specific Indices from the List
def select_from_list(src_list, sel_indices):
    if len(sel_indices) == 0:
        return []
    else:
        selected_list = list(src_list[i] for i in sel_indices)
        return selected_list


# Exclude Specific Indices from the List
def exclude_from_list(src_list, excl_indices):
    if len(excl_indices) == 0:
        return src_list
    else:
        survived_indices = []
        for src_idx, _ in enumerate(src_list):
            if src_idx not in excl_indices:
                survived_indices.append(src_idx)
        return select_from_list(src_list, survived_indices)


# Search Common Points
def search_common_points(points1, points2):
    common_x, cind_x_1, cind_x_2 = np.intersect1d(points1[:, 0], points2[:, 0], return_indices=True)
    common_y, cind_y_1, cind_y_2 = np.intersect1d(points1[:, 1], points2[:, 1], return_indices=True)

    cind_1 = np.intersect1d(cind_x_1, cind_y_1)
    cind_2 = np.intersect1d(cind_x_2, cind_y_2)

    if len(cind_1) == 0 or len(cind_2) == 0:
        return [], [], []
    else:
        if not np.array_equal(points1[cind_1, :], points2[cind_2, :]):
            assert 0
        else:
            cpoints = points1[cind_1, :]

        return cpoints, cind_1, cind_2


# Search Uncommon Points
def search_uncommon_points(points1, points2):
    _, cind_1, cind_2 = search_common_points(points1, points2)

    excl_points1 = []
    pidx_1 = []
    for pidx, point1 in enumerate(points1):
        if pidx not in cind_1:
            pidx_1.append(pidx)
            excl_points1.append(point1)

    excl_points2 = []
    pidx_2 = []
    for pidx, point2 in enumerate(points2):
        if pidx not in cind_2:
            pidx_2.append(pidx)
            excl_points2.append(point2)

    return np.array(excl_points1).reshape(-1, 2), np.array(excl_points2).reshape(-1, 2), pidx_1, pidx_2


# Generate 2D Gaussian Window
def generate_gaussian_window(row_size, col_size, mean, stdev, min_val=0, max_val=1):
    # Construct 2-D Gaussian Window
    x, y = np.meshgrid(np.linspace(-1, 1, col_size), np.linspace(-1, 1, row_size))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-((d - mean) ** 2 / (2.0 * stdev ** 2)))

    # cv2.imshow("Unnormalized Gaussian Map", g)

    # Normalize So that the matrix is scaled between "eps" and "1"
    gvec = g.flatten()
    gauss_map = min_val + ((g - np.min(gvec)) * (max_val - min_val) / (np.max(gvec) - np.min(gvec)))
    return gauss_map













