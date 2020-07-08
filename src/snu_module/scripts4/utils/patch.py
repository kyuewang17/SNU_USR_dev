"""
SNU Integrated Module v2.0
  - Code regarding Patch (ex. image patch)
"""
# Import Module
import numpy as np


# Get Patch
def get_patch(img, bbox, enlarge_factor=1):
    # Overflow
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[1] = 0
    if bbox[2] > img.shape[1]:
        bbox[2] = img.shape[1]
    if bbox[3] > img.shape[0]:
        bbox[3] = img.shape[0]

    x1 = np.floor(((1+enlarge_factor)*bbox[0] + (1-enlarge_factor)*bbox[2]) / 2).astype(int)
    y1 = np.floor(((1+enlarge_factor)*bbox[1] + (1-enlarge_factor)*bbox[3]) / 2).astype(int)
    x2 = np.floor(((1-enlarge_factor)*bbox[0] + (1+enlarge_factor)*bbox[2]) / 2).astype(int)
    y2 = np.floor(((1-enlarge_factor)*bbox[1] + (1+enlarge_factor)*bbox[3]) / 2).astype(int)
    patch = img[np.maximum(y1, 0):np.minimum(y2, img.shape[0]), np.maximum(x1, 0):np.minimum(x2, img.shape[1])]
    return patch


# Black-out 2D Array Patch of a squared region
def blankout_from_patch(patch, blank_bbox, blank_val=0):
    blank_row_min = np.maximum(blank_bbox[1], 0)
    blank_row_max = np.minimum(blank_bbox[3], patch.shape[0])
    blank_col_min = np.maximum(blank_bbox[0], 0)
    blank_col_max = np.minimum(blank_bbox[2], patch.shape[1])
    patch[blank_row_min:blank_row_max, blank_col_min:blank_col_max] = blank_val
    return patch
