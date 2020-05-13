"""
SNU Integrated Module v3.0
    - Multimodal Multiple Target Tracking


"""

# Import Modules
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment as hungarian

# Import Custom Modules
import snu_utils.patch as snu_patch
import snu_utils.bounding_box as snu_bbox
import snu_utils.general_functions as snu_gfuncs
import snu_utils.histogram as snu_hist

# Import Class Objects
from class_objects import TrackletCandidate, Tracklet


def tracker(sync_data_dict, fidx, detections, max_trk_id, opts, trks, trk_cands):
    # Initialize New Tracklet Variable
    new_trks = []

    # Destroy Tracklets with following traits
    destroy_trk_indices = []
    for trk_idx, trk in enumerate(trks):
        # (1) Tracklets with tiny bounding box
        if trk.x3[5] * trk.x3[6] < 10:
            destroy_trk_indices.append(trk_idx)

        # (2) Prolonged Consecutively Unassociated Tracklets
        if snu_gfuncs.get_max_consecutive(trk.is_associated, False) == \
                opts.tracker.association["trk_destroy_age"]:
            destroy_trk_indices.append(trk_idx)

        # Remove Duplicate Indices
        destroy_trk_indices = list(set(destroy_trk_indices))
    trks = snu_gfuncs.exclude_from_list(trks, destroy_trk_indices)

    # Destroy Prolonged Tracklet Candidates
    destroy_trkc_indices = []
    for trkc_idx, trk_cand in enumerate(trk_cands):
        if snu_gfuncs.get_max_consecutive(trk_cand.is_associated, False) == \
                opts.tracker.association["trkc_destroy_age"]:
            destroy_trkc_indices.append(trkc_idx)
    trk_cands = snu_gfuncs.exclude_from_list(trk_cands, destroy_trkc_indices)

    # Associate Detections with Tracklets
    if len(trks) != 0:
        # Get Association Match Information
        matches_d2t, unasso_det_indices, unasso_trk_indices = \
            asso_dets_trks(
                sync_data_dict=sync_data_dict, detections=detections,
                trks=trks, cost_thresh=opts.tracker.association['cost_thresh_d2trk']
            )

        # Update Associated Tracklets
        for match in matches_d2t:
            matched_det = detections['dets'][match[0]]
            matched_conf, matched_label = detections['confs'][match[0]], detections['labels'][match[0]]

            matched_trk = trks[match[1]]

            # < Double-check for Label Consistency >
            if matched_label != matched_trk.label:
                unasso_det_indices.append(match[0]), unasso_trk_indices.append(match[1])
                print("[TEST] Inconsistent Label Test!")
            else:
                # TODO: Depth Observation (Matched)
                matched_trk.get_depth(sync_data_dict, opts)

                # If passed, update Tracklet
                matched_trk.update(fidx, matched_det, matched_conf)
                trks[match[1]] = matched_trk
            del matched_trk

        # Update Unassociated Tracklets
        for unasso_trk_idx in unasso_trk_indices:
            unasso_trk = trks[unasso_trk_idx]

            # TODO: Depth Observation (Unmatched)
            unasso_trk.get_depth(sync_data_dict, opts)

            # Update
            unasso_trk.update(fidx)
            trks[unasso_trk_idx] = unasso_trk
            del unasso_trk

        # Remove Associated Detections and Collect Residual Detections
        residual_dets = np.empty((len(unasso_det_indices), 4))
        residual_confs, residual_labels = np.empty((len(unasso_det_indices), 1)), np.empty((len(unasso_det_indices), 1))
        for residual_det_idx, unasso_det_idx in enumerate(unasso_det_indices):
            residual_dets[residual_det_idx, :] = detections['dets'][unasso_det_idx]
            residual_confs[residual_det_idx] = detections['confs'][unasso_det_idx]
            residual_labels[residual_det_idx] = detections['labels'][unasso_det_idx]
        detections = {'dets': residual_dets, 'confs': residual_confs, 'labels': residual_labels}
        del residual_dets, residual_confs, residual_labels

    # Associate Residual Detections with Tracklet Candidates
    if len(trk_cands) == 0:
        # Initialize New Tracklet Candidates
        for det_idx, det in enumerate(detections["dets"]):
            new_trk_cand = TrackletCandidate(
                bbox=det, conf=detections["confs"][det_idx], label=detections["labels"][det_idx],
                init_fidx=fidx
            )
            trk_cands.append(new_trk_cand)
            del new_trk_cand
    else:
        # Get Association Match
        matches_d2tc, unasso_det_indices, unasso_trkc_indices = \
            asso_resdets_trkcands(
                sync_data_dict=sync_data_dict, residual_detections=detections,
                trk_cands=trk_cands, fidx=fidx, cost_thresh=opts.tracker.association['cost_thresh_d2trkc']
            )

        # Update Associated Tracklet Candidates
        for match in matches_d2tc:
            # Matched Detection
            matched_det = detections['dets'][match[0]]
            matched_conf, matched_label = detections['confs'][match[0]], detections['labels'][match[0]]

            # Matched Tracklet Candidate
            matched_trk_cand = trk_cands[match[1]]

            # Update Tracklet Candidate
            if matched_label != matched_trk_cand.label:
                unasso_det_indices.append(match[0]), unasso_trkc_indices.append(match[1])
            else:
                matched_trk_cand.update(fidx, matched_det, matched_conf)
                trk_cands[match[1]] = matched_trk_cand
            del matched_trk_cand

        # Update Unassociated Tracklet Candidates
        for unasso_trkc_idx in unasso_trkc_indices:
            unasso_trk_cand = trk_cands[unasso_trkc_idx]

            # Update
            unasso_trk_cand.update(fidx=fidx)
            trk_cands[unasso_trkc_idx] = unasso_trk_cand
            del unasso_trk_cand

        # Generate New Tracklet Candidates with the Unassociated Detections
        for unasso_det_idx in unasso_det_indices:
            new_trk_cand = TrackletCandidate(
                bbox=detections['dets'][unasso_det_idx],
                conf=detections['confs'][unasso_det_idx],
                label=detections['labels'][unasso_det_idx],
                init_fidx=fidx
            )
            trk_cands.append(new_trk_cand)
            del new_trk_cand

    # Generate New Tracklets from Tracklet Candidates
    if len(trk_cands) != 0:
        # Associate Tracklet Candidates with Detection Associated Consecutively for < k > frames
        selected_trkc_indices = []
        for trkc_idx, trk_cand in enumerate(trk_cands):
            if snu_gfuncs.get_max_consecutive(trk_cand.is_associated, True) == \
                    opts.tracker.association["trk_init_age"]:
                selected_trkc_indices.append(trkc_idx)
        sel_trk_cands = snu_gfuncs.select_from_list(trk_cands, selected_trkc_indices)

        # Initialize New Tracklets
        for sel_trkc_idx, sel_trk_cand in enumerate(sel_trk_cands):
            # Get New Tracklet ID
            new_trk_id = max_trk_id + 1 + sel_trkc_idx

            # Initialize New Tracklet
            new_trk = sel_trk_cand.init_tracklet(
                disparity_frame=sync_data_dict["disparity"].get_data(),
                trk_id=new_trk_id, fidx=fidx, opts=opts
            )
            new_trks.append(new_trk)
            del new_trk
        del sel_trk_cands

        # Destroy Associated Tracklet Candidates
        trk_cands = snu_gfuncs.exclude_from_list(trk_cands, selected_trkc_indices)

    # Append New Tracklets
    for new_trk in new_trks:
        trks.append(new_trk)
        del new_trk
    del new_trks

    # Predict Tracklet States and Print Tracking Message
    msg_fidx_part = "Frame #[%08d] --> {Tracklets}: " % fidx
    tracklet_recursive_msg = ""
    for trk_idx, trk in enumerate(trks):
        trk.predict()
        trks[trk_idx] = trk

        if trk_idx < len(trks)-1:
            add_tracklet_msg = "[%d]," % trk.id
        else:
            add_tracklet_msg = "[%d]" % trk.id
        tracklet_recursive_msg += add_tracklet_msg
        del trk

    msg_trk = msg_fidx_part + tracklet_recursive_msg
    print(msg_trk)
    # print(len(trk_cands))

    return trks, trk_cands


# Associate Residual Detections with Tracklet Candidates
def asso_resdets_trkcands(sync_data_dict, residual_detections, trk_cands, fidx, cost_thresh):
    # Unpack Residual Detections
    dets, confs, labels = \
        residual_detections["dets"], residual_detections["confs"], residual_detections["labels"]

    # Initialize Cost Matrix Variable
    cost_matrix = np.zeros((len(dets), len(trk_cands)), dtype=np.float32)

    # Calculate Cost Matrix
    for det_idx, det in enumerate(dets):
        for trk_cand_idx, trk_cand in enumerate(trk_cands):
            if trk_cand.z[-1] is None:
                cost_val = -1
            else:
                det_zx = snu_bbox.bbox_to_zx(det)
                trk_cand_bbox, _ = snu_bbox.zx_to_bbox(trk_cand.z[-1])

                # Consider IOU and L2-distance
                iou_cost = snu_bbox.iou(det, trk_cand_bbox)
                l2_distance = snu_gfuncs.l2_distance_dim2(
                    x1=det_zx[0], y1=det_zx[1],
                    x2=trk_cand.z[-1][0], y2=trk_cand.z[-1][1]
                )

                # Cost
                cost_val = (iou_cost + 1e-12) / (l2_distance + 1e-12)

            # to Cost Matrix
            cost_matrix[det_idx, trk_cand_idx] = cost_val

    # Associate using Hungarian Algorithm
    matches, unmatched_det_indices, unmatched_trk_cand_indices = \
        associate(
            cost_matrix=cost_matrix, cost_thresh=cost_thresh,
            workers=dets, works=trk_cands
        )

    return matches, unmatched_det_indices, unmatched_trk_cand_indices


# Associate Detections with Tracklets
def asso_dets_trks(sync_data_dict, detections, trks, cost_thresh):
    # Unpack Detections
    dets, confs, labels = detections["dets"], detections["confs"], detections["labels"]

    # Initialize Cost Matrix Variable
    cost_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)

    # Get Concatenated Frame
    color_frame = sync_data_dict["color"].get_data()
    normalized_disparity_frame = sync_data_dict["disparity"].get_normalized_data(0, 255)
    rgbd_frame = np.dstack((color_frame, normalized_disparity_frame))

    # Calculate Cost Matrix
    for det_idx, det in enumerate(dets):
        for trk_idx, trk in enumerate(trks):
            det_zx = snu_bbox.bbox_to_zx(det)

            # Get Predicted State of Tracklet
            trk_bbox, _ = snu_bbox.zx_to_bbox(trk.pred_states[-1])

            # Get Detection and Tracklet Patch RGBD Histograms
            det_hist, det_hist_idx = snu_hist.histogramize_patch(
                sensor_patch=snu_patch.get_patch(rgbd_frame, det),
                dhist_bin=128, min_value=0, max_value=255, count_window=None
            )
            trk_hist, trk_hist_idx = snu_hist.histogramize_patch(
                sensor_patch=snu_patch.get_patch(rgbd_frame, trk_bbox),
                dhist_bin=128, min_value=0, max_value=255, count_window=None
            )

            # Get Histogram Similarity
            if len(det_hist) == 0 or len(trk_hist) == 0:
                hist_similarity = 1.0
            else:
                hist_product = np.matmul(det_hist.reshape(-1, 1).transpose(), trk_hist.reshape(-1, 1))
                hist_similarity = hist_product / (np.linalg.norm(det_hist) * np.linalg.norm(trk_hist))

            # Get IOU
            iou_cost = snu_bbox.iou(det, trk_bbox)

            # Get L2-distance (center distance)
            l2_distance = snu_gfuncs.l2_distance_dim2(
                x1=det_zx[0], y1=det_zx[1],
                x2=trk.pred_states[-1][0], y2=trk.pred_states[-1][1]
            )

            # Cost
            cost_val = (iou_cost * hist_similarity + 1e-12) / (l2_distance + 1e-12)

            # to Cost Matrix
            cost_matrix[det_idx, trk_idx] = cost_val

    # Associate Using Hungarian Algorithm
    matches, unmatched_det_indices, unmatched_trk_indices = \
        associate(cost_matrix, cost_thresh, dets, trks)

    return matches, unmatched_det_indices, unmatched_trk_indices


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

    # Filter-out Matched with Cost lower then the threshold
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





















































