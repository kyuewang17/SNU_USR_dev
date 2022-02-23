"""
SNU Integrated Module v5.0
  - Code which computes depth inside BBOX, given LiDAR UV-array inside the BBOX

"""
import numpy as np
from module_lib.v4_5._TRK.objects.bbox import BBOX


def compute_depth(uv_array, pc_distances, patch_bbox):
    # Define Steepness Parameter w.r.t. Standard Deviation of Point-cloud Distance Distribution
    # PC_stdev is Low -> Gradual Counting Weight (to consider more samples)
    #             High  -> Steep Counting Weight (to consider less samples near average point)
    # stnp = (1.0 / (np.std(pc_distances) + 1e-6))
    stnp = np.std(pc_distances)

    # Check for patch_bbox type
    assert isinstance(patch_bbox, BBOX)

    # Compute Counting Weight for LiDAR UV-points, w.r.t. center L2-distance
    _denom = (patch_bbox[2] - patch_bbox[0]) ** 2 + (patch_bbox[3] - patch_bbox[1]) ** 2
    cx, cy = patch_bbox.x, patch_bbox.y
    _num = np.sum((uv_array - np.array([cx, cy])) ** 2, axis=1)
    _w = np.exp(-stnp * 4.0 * (_num / (_denom + 1e-6)))
    counting_weight = _w / _w.sum()

    # Weighted Distance Sum
    depth_value = np.inner(pc_distances, counting_weight)
    if np.isnan(depth_value):
        depth_value = np.median(pc_distances)
    return depth_value


if __name__ == "__main__":
    pass
