"""
SNU Integrated Module v2.0
  - Code for data association (Multimodal Multi-target Tracking)
"""
# Import Modules
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment as hungarian

# Import Source Modules
import bounding_box as fbbox


# Data Association Function
def associate(cost_matrix, cost_thresh, workers, works):
    # Hungarian Algorithm
    matched_indices = hungarian(-cost_matrix)

    # Collect Unmatched Worker Indices
    unmatched_worker_indices = []
    for worker_idx, _ in enumerate(workers):
        if worker_idx not in matched_indices[:, 0]:
            unmatched_worker_indices.append(worker_idx)

    # Collect Unmatched Work Indices
    unmatched_work_indices = []
    for work_idx, _ in enumerate(works):
        if work_idx not in matched_indices[:, 1]:
            unmatched_work_indices.append(work_idx)

    # Filter out Matched with Low Cost
    matches = []
    for m in matched_indices:
        if cost_matrix[m[0], m[1]] < cost_thresh:
            unmatched_worker_indices.append(m[0])
            unmatched_work_indices.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, unmatched_worker_indices, unmatched_work_indices


# Detection-to-TrackletCandidate Association
def asso_det_trkcand(dets, trk_cands, cost_thresh):
    # Initialize Cost Matrix Variable
    cost_matrix = np.zeros((len(dets), len(trk_cands)), dtype=np.float32)

    # Calculate Cost Matrix
    for det_idx, det in enumerate(dets):
        for trk_cand_idx, trk_cand in enumerate(trk_cands):
            if trk_cand.z[-1] is None:
                cost_val = -1
            else:
                trk_cand_bbox, _ = fbbox.zx_to_bbox(trk_cand.z[-1])

                # Calculate Cost
                cost_val = fbbox.iou(det, trk_cand_bbox)

            # Cost Matrix
            cost_matrix[det_idx, trk_cand_idx] = cost_val

    # Associate Using Hungarian Algorithm
    matches, unmatched_det_indices, unmatched_trk_cand_indices = \
        associate(cost_matrix, cost_thresh, dets, trk_cands)

    return matches, unmatched_det_indices, unmatched_trk_cand_indices


# Detection-to-Tracklet Association
def asso_det_tracklet(dets, trks, cost_thresh):
    # Initialize Cost Matrix Variable
    cost_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)

    # Calculate Cost Matrix
    for det_idx, det in enumerate(dets):
        for trk_idx, trk in enumerate(trks):
            trk_bbox, _ = fbbox.zx_to_bbox(trk.pred_states[-1])

            # Get Velocity
            # det_vel = np.zeros(2)
            # trk_vel = np.array(trk.pred_states[-1][2:4].reshape(2))

            # Calculate Cost
            cost_val = fbbox.iou(det, trk_bbox)
            # cost_val = vel_iou(det, det_vel, trk_bbox, trk_vel)

            # Cost Matrix
            cost_matrix[det_idx, trk_idx] = cost_val

    # Associate Using Hungarian Algorithm
    matches, unmatched_det_indices, unmatched_trk_indices = \
        associate(cost_matrix, cost_thresh, dets, trks)

    # Return
    return matches, unmatched_det_indices, unmatched_trk_indices



























