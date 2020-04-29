"""
SNU Integrated Module v2.0
  - Code regarding Patch (ex. image patch)
"""
# Import Module
import numpy as np


# Get Patch
def get_patch(img, bbox):
    patch_row_min = np.maximum(bbox[1], 0)
    patch_row_max = np.minimum((bbox[1] + bbox[3]), img.shape[0])
    patch_col_min = np.maximum(bbox[0], 0)
    patch_col_max = np.minimum((bbox[0] + bbox[2]), img.shape[1])
    patch = img[patch_row_min:patch_row_max, patch_col_min:patch_col_max]
    return patch


# Black-out 2D Array Patch of a squared region
def blankout_from_patch(patch, blank_bbox, blank_val=0):
    blank_row_min = np.maximum(blank_bbox[1], 0)
    blank_row_max = np.minimum((blank_bbox[1] + blank_bbox[3]), patch.shape[0])
    blank_col_min = np.maximum(blank_bbox[0], 0)
    blank_col_max = np.minimum((blank_bbox[0] + blank_bbox[2]), patch.shape[1])
    patch[blank_row_min:blank_row_max, blank_col_min:blank_col_max] = blank_val
    return patch


#




































