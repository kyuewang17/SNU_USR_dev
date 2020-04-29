"""
Outdoor Unmanned Surveillance Robot SNU Multimodal MOT Module v1.5

    - Code partially based on SORT (ICIP2016)

    - Code written/modified by : Kyuewang Lee (kyuewang5056@gmail.com)

"""

from __future__ import print_function

import numpy as np
import mot_module as mot
from mot_class import TrackletCandidate, Tracklet


# Mutimodal MOT Tracker (ProtoType)
def tracker(color_img, lidar_pc, fidx, dets, motparams, max_trk_id, trks=[], trk_cands=[]):
    # Unpack motparams
    unasso_trk_destroy = motparams.unasso_trk_destroy
    unasso_trkc_destroy = motparams.unasso_trkc_destroy
    cost_thresh = motparams.cost_thresh
    trkc_to_trk_asso_age = motparams.trkc_to_trk_asso_age
    dhist_bin = motparams.dhist_bin

    depth_min_distance = motparams.depth_min_distance
    depth_max_distance = motparams.depth_max_distance

    kitti_calib = motparams.calib

    # D435 Parameters
    color_cam_K = motparams.color_cam_K
    color_cam_P = motparams.color_cam_P
    depth_cam_K = motparams.depth_cam_K
    depth_cam_P = motparams.depth_cam_P

    # Project Point Clouds with Calibration and Projection Matrices
    if kitti_calib is not None:
        pts2d, dist_vec = mot.project_velo_to_image(lidar_pc, kitti_calib)
    else:
        pts2d = None
        dist_vec = None

    # Initialize New Tracklet Variable
    new_trks = []

    # Destroy Prolonged Consecutively Unassociated Tracklets
    destroy_trk_indices = []
    for trk_idx, trk in enumerate(trks):
        consec_unasso = mot.get_max_consecutive(trk.is_associated, False)
        if consec_unasso == unasso_trk_destroy:
            destroy_trk_indices.append(trk_idx)
    trks = mot.exclude_from_list(trks, destroy_trk_indices)

    # Destroy Prolonged Tracklet Candidates
    destroy_trkc_indices = []
    for trkc_idx, trk_cand in enumerate(trk_cands):
        if trk_cand.age == unasso_trkc_destroy:
            destroy_trkc_indices.append(trkc_idx)
    trk_cands = mot.exclude_from_list(trk_cands, destroy_trkc_indices)

    # Associate Detections with Tracklets
    if len(trks) != 0:
        # Get Association Match INFO
        matches_d2t, unasso_det_indices, unasso_trk_indices = \
            mot.asso_det_tracklet(dets, trks, pts2d, dist_vec, dhist_bin, cost_thresh)

        # Update Associated Tracklets
        for match in matches_d2t:
            matched_detection = dets[match[0]]
            matched_trk = trks[match[1]]

            # Update
            matched_trk.update(matched_detection)
            trks[match[1]] = matched_trk
            del matched_trk

        # Update Unassociated Tracklets
        for unasso_trk_idx in unasso_trk_indices:
            unasso_trk = trks[unasso_trk_idx]

            # Update
            unasso_trk.update()
            trks[unasso_trk_idx] = unasso_trk
            del unasso_trk

        # Remove Associated Detections
        residual_dets = np.empty((len(unasso_det_indices), 4))
        for residual_det_idx, unasso_det_idx in enumerate(unasso_det_indices):
            residual_dets[residual_det_idx, :] = dets[unasso_det_idx]
        dets = residual_dets

    # Associate Residual Detections with Tracklet Candidates
    if len(trk_cands) == 0:
        for det in dets:
            new_trk_cand = TrackletCandidate(det)
            trk_cands.append(new_trk_cand)
            del new_trk_cand
    else:
        # Get Association Matches
        matches_d2tc, unasso_det_indices, unasso_trkc_indices = \
            mot.asso_det_trkcand(dets, trk_cands, cost_thresh)

        # Associate and Update Tracklet Candidates
        for match in matches_d2tc:
            matched_detection = dets[match[0]]
            matched_trk_cand = trk_cands[match[1]]

            # Update
            matched_trk_cand.update(matched_detection)
            trk_cands[match[1]] = matched_trk_cand
            del matched_trk_cand

        # # Destroy Unassociated Tracklet Candidates
        # trk_cands = mot.exclude_from_list(trk_cands, unasso_trkc_indices)

        # Update Unassociated Tracklet Candidates
        for unasso_trkc_idx in unasso_trkc_indices:
            unasso_trk_cand = trk_cands[unasso_trkc_idx]

            # Update
            unasso_trk_cand.update()
            trk_cands[unasso_trkc_idx] = unasso_trk_cand
            del unasso_trk_cand

        # Generate New Tracklet Candidates with the unassociated detections
        for unasso_det_idx in unasso_det_indices:
            new_trk_cand = TrackletCandidate(dets[unasso_det_idx])
            trk_cands.append(new_trk_cand)
            del new_trk_cand

    # Get Current Frame Maximum Tracklet ID
    # this part has very significant error (when there are no tracklets in a frame)
    max_id = mot.get_maximum_id(trks, max_trk_id)

    # Generate New Tracklets from Tracklet Candidates
    if len(trk_cands) != 0:
        # Associate Only Tracklet Candidates with Detection Associated Consecutively for k frames
        selected_trkc_indices = []
        for trkc_idx, trk_cand in enumerate(trk_cands):
            max_consec_asso = mot.get_max_consecutive(trk_cand.is_associated, True)
            if max_consec_asso == trkc_to_trk_asso_age:
                selected_trkc_indices.append(trkc_idx)
        sel_trk_cands = mot.select_from_list(trk_cands, selected_trkc_indices)

        # Initialize Tracklets
        for sel_trkc_idx, sel_trk_cand in enumerate(sel_trk_cands):
            selected_trkc_bbox, _ = mot.zx_to_bbox(sel_trk_cand.z[-1])

            # Generate Tracklet
            tracklet = Tracklet(selected_trkc_bbox, fidx, max_id + 1 + sel_trkc_idx)
            new_trks.append(tracklet)
            del tracklet
        del sel_trk_cands

        # Destroy Associated Tracklet Candidates
        trk_cands = mot.exclude_from_list(trk_cands, selected_trkc_indices)

    # Append New Tracklets
    for new_trk in new_trks:
        trks.append(new_trk)
        del new_trk
    del new_trks

    # Predict Tracklet States
    for trk_idx, trk in enumerate(trks):
        trk.predict()
        trks[trk_idx] = trk

    # Assign Projected Point Clouds to Tracklets
    if pts2d is not None:
        trks = mot.assign_projected_pointcloud(trks, pts2d, dist_vec)

    # Two-stage Depth Inference
    if pts2d is not None:
        trks = mot.depth_inference(trks, dhist_bin)
    else:
        #####################################################
        # D435 --> depth_img is "lidar_pc", since this is temporary code
        # depth_list = mot.depth_inference_d435(trks, depth_img=lidar_pc, dhist_bin=dhist_bin)
        depth_list, depth_idx_list = mot.depth_inference_d435_simple(trks, depth_img=lidar_pc, dhist_bin=dhist_bin, min_value=depth_min_distance, max_value=depth_max_distance)
        for depth_idx, depth in enumerate(depth_list):
            trk = trks[depth_idx]
            trk.update_depth(depth, depth_idx_list[depth_idx])

    # Get Pseudo-inverse of Projection Matrix
    color_cam_P_inverse = np.linalg.pinv(color_cam_P)

    # Image Coordinate (u,v,d) to Camera Coordinate (x,y,z)
    for trk_idx, trk in enumerate(trks):

        # Calculate 3D in camera coordinates
        trk.get_3d_cam_coord(color_cam_P_inverse)

        # Kalman Estimation (Update)
        if trk.cam_coord_predicted_state is None:
            trk.cam_coord_estimated_state = trk.cam_coord_raw_state
        else:
            trk.update_3d_cam_coord()

        # Compute RPY
        trk.compute_rpy()

        # Kalman Prediction
        trk.predict_3d_cam_coord()

        # Feed-in
        trks[trk_idx] = trk

    # MOT Message
    mot_mesg = "[Frame: " + str(fidx) + "] Tracking " + str(len(trks)) + " Tracklets..!"
    print(mot_mesg)

    # Get Maximum Tracklet ID among all Tracklets
    max_id = mot.get_maximum_id(trks, max_id)

    return trks, trk_cands, max_id




















