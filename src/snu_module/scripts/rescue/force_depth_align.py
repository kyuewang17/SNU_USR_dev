"""
Force Depth Align with Heuristics

Algorithm by MIPAL,SNU

"""
import numpy as np
import cv2

lt_ratio = (0.175, 0.200)
rb_ratio = (0.810, 0.825)


# Align Depth
def force_align_depth(depth_image):
    h, w = depth_image.shape[:2]
    l, t = int(lt_ratio[0] * w), int(lt_ratio[1] * h)
    r, b = int(rb_ratio[0] * w), int(rb_ratio[1] * h)
    aligned_depth_image = cv2.resize(depth_image[t:b, l:r], dsize=(w, h))
    return aligned_depth_image


