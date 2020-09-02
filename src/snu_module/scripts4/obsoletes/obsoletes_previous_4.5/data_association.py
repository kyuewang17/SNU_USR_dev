"""
SNU Integrated Module v2.05
  - Code for data association (Multimodal Multi-target Tracking)

    [Update Logs]
    - 190916
        (1) Detection Label-aware Association Method

"""
# Import Modules
import cv2
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment as hungarian

# Import Source Modules
import patch as ptch
import bounding_box as fbbox
import general_functions as gfuncs


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

    # Filter-out Matched with Cost lower than the threshold
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


# Detection to Tracklet Candidate Association (equivalent to detection-detection)
def asso_det_trkcand(imgStruct_dict, detections, trk_cands, cost_thresh):
    # Unpack Detections
    dets, confs, labels = detections['dets'], detections['confs'], detections['labels']

    # Initialize Cost Matrix Variable
    cost_matrix = np.zeros((len(dets), len(trk_cands)), dtype=np.float32)

    # Calculate Cost Matrix
    for det_idx, det in enumerate(dets):
        for trk_cand_idx, trk_cand in enumerate(trk_cands):
            if trk_cand.z[-1] is None:
                cost_val = -1
            else:
                det_zx = fbbox.bbox_to_zx(det)
                trk_cand_bbox, _ = fbbox.zx_to_bbox(trk_cand.z[-1])

                # Calculate Cost
                # [IMPLEMENT THIS!!!!!!!!!!!]
                # Label and Confidence-aware Cost Value
                # cost_val = fbbox.iou(det, trk_cand_bbox)

                # Version 02
                iou_cost = fbbox.iou(det, trk_cand_bbox)
                l2_distance = gfuncs.l2_distance_dim2(det_zx[0], det_zx[1], trk_cand.z[-1][0], trk_cand.z[-1][1])

                cost_val = (iou_cost + 1e-12) / (l2_distance + 1e-12)

            # Push Cost Value to the Cost Matrix
            cost_matrix[det_idx, trk_cand_idx] = cost_val

    # Associate Using Hungarian Algorithm
    matches, unmatched_det_indices, unmatched_trk_cand_indices = \
        associate(cost_matrix, cost_thresh, dets, trk_cands)

    return matches, unmatched_det_indices, unmatched_trk_cand_indices


# Detection to Tracklet Association
def asso_det_tracklet(imgStruct_dict, detections, trks, cost_thresh):
    # Unpack Detections
    dets, confs, labels = detections['dets'], detections['confs'], detections['labels']

    # Initialize Cost Matrix Variable
    cost_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)

    # Calculate Cost Matrix
    for det_idx, det in enumerate(dets):
        for trk_idx, trk in enumerate(trks):
            det_zx = fbbox.bbox_to_zx(det)

            # Get Predicted State of Tracklet
            trk_bbox, _ = fbbox.zx_to_bbox(trk.pred_states[-1])

            # Crop RGB Patches
            det_hsv_patch = cv2.cvtColor(ptch.get_patch(imgStruct_dict["rgb"].frame.raw, det), cv2.COLOR_RGB2HSV)
            trk_hsv_patch = cv2.cvtColor(ptch.get_patch(imgStruct_dict["rgb"].frame.raw, trk_bbox), cv2.COLOR_RGB2HSV)

            # Histogram (HSV)
            if det_hsv_patch == [] or trk_hsv_patch == []:
                det_hsv_hist = cv2.calcHist(det_hsv_patch, [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256]).reshape(-1, 1)
                trk_hsv_hist = cv2.calcHist(trk_hsv_patch, [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256]).reshape(-1, 1)

                hist_similarity = np.matmul(det_hsv_hist.transpose(), trk_hsv_hist) / (np.linalg.norm(det_hsv_hist) * np.linalg.norm(trk_hsv_hist))
            else:
                hist_similarity = 1

            # Calculate Cost
            # [IMPLEMENT THIS!!!!!!!!!!!]
            # Label and Confidence-aware Cost Value
            # cost_val = fbbox.iou(det, trk_bbox)

            # Version 02
            iou_cost = fbbox.iou(det, trk_bbox)
            l2_distance = gfuncs.l2_distance_dim2(det_zx[0], det_zx[1], trk.pred_states[-1][0], trk.pred_states[-1][1])

            # Cost
            cost_val = (iou_cost*hist_similarity + 1e-12) / (l2_distance + 1e-12)

            # Push Cost Value to the Cost Matrix
            cost_matrix[det_idx, trk_idx] = cost_val

    # Associate Using Hungarian Algorithm
    matches, unmatched_det_indices, unmatched_trk_indices = \
        associate(cost_matrix, cost_thresh, dets, trks)

    # Return
    return matches, unmatched_det_indices, unmatched_trk_indices





























