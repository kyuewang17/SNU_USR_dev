"""
SNU Integrated Module v2.0
  - Multimodal Multi-target Tracking

"""
# Import Module
import numpy as np

# Import Source Modules
import utils.patch as ptch
import utils.bounding_box as fbbox
import utils.general_functions as gfuncs
import utils.data_association_old as fasso
import utils.histogram as fhist
from class_objects_old import TrackletCandidate, Tracklet


# Depth Inference
def depth_inference(trks, depth_img, dhist_bin, min_value, max_value):
    if len(trks) != 0 and depth_img != []:
        depth_list = [None] * len(trks)
        depth_idx_list = [None] * len(trks)

        for trk_idx, trk in enumerate(trks):
            trk_state_bbox, _ = fbbox.zx_to_bbox(trk.x)
            trk_state_bbox = trk_state_bbox.astype(int)

            # Get Depth Patch
            trk_depth_patch = ptch.get_patch(depth_img, trk_state_bbox)

            # Get Depth Histogram
            depth_histogram, depth_idx = fhist.get_depth_histogram(trk_depth_patch, dhist_bin, min_value, max_value)

            # Push into List
            depth_list[trk_idx] = depth_histogram
            depth_idx_list[trk_idx] = depth_idx
    else:
        depth_list = []
        depth_idx_list = []

    return depth_list, depth_idx_list


# Multimodal Depth Inference
def multimodal_depth_inference(trks, imgStruct, mdparams):
    if len(trks) != 0:
        # Initialize
        depth_list, depth_idx_list = [None] * len(trks), [None] * len(trks)

        # Set Gaussian Parameters of the Histogram Counting Window
        gauss_mean, gauss_stdev = 0, 0.6

        for trk_idx, trk in enumerate(trks):
            # Get Tracklet Bounding Box Position
            trk_state_bbox, _ = fbbox.zx_to_bbox(trk.x)
            trk_state_bbox = trk_state_bbox.astype(int)

            # [1] Get d435 depth patch
            trk_depth_patch = ptch.get_patch(imgStruct.depth.processed, trk_state_bbox)
            # Counting Window for Weighted Histogram
            count_window_depth = gfuncs.generate_gaussian_window(trk_depth_patch.shape[0], trk_depth_patch.shape[1],
                                                                 gauss_mean, gauss_stdev, min_val=1, max_val=10)

            # [2] Get LIDAR image patch
            if imgStruct.lidar.processed is not None:
                trk_lidar_patch = ptch.get_patch(imgStruct.lidar.processed, trk_state_bbox)
                count_window_lidar = gfuncs.generate_gaussian_window(trk_lidar_patch.shape[0], trk_lidar_patch.shape[1],
                                                                     gauss_mean, gauss_stdev, min_val=1, max_val=10)

                # Concatenate Patches
                trk_concat_patch = np.concatenate((trk_depth_patch, trk_lidar_patch), axis=1)
                count_concat_window = np.concatenate((count_window_depth, count_window_lidar), axis=1)

                # Get Weighted Histogram
                depth_histogram, depth_idx = fhist.histogramize_patch(trk_concat_patch, mdparams.tracker.hist_bin,
                                                                      min_value=0, max_value=50000, count_window=count_concat_window)
            else:
                depth_histogram, depth_idx = fhist.histogramize_patch(trk_depth_patch, mdparams.tracker.hist_bin,
                                                                      min_value=0, max_value=50000, count_window=count_window_depth)

            # Push into List
            depth_list[trk_idx] = depth_histogram
            depth_idx_list[trk_idx] = depth_idx
    else:
        depth_list, depth_idx_list = [], []

    return depth_list, depth_idx_list


# Multimodal Multi-target Tracker
def tracker(imgStruct, fidx, dets, tparams, max_trk_id, trks=[], trk_cands=[]):
    # Initialize New Tracklet Variable
    new_trks = []

    # Parse Detection Results
    confs, labels = dets[:, 4:5], dets[:, 5:6]
    dets = dets[:, 0:4]

    # Get Maximum Tracklet ID (before destroying prolonged unassociated Tracklets)
    max_id = gfuncs.get_maximum_id(trks, max_trk_id)

    # Destroy Prolonged Consecutively Unassociated Tracklets
    destroy_trk_indices = []
    for trk_idx, trk in enumerate(trks):
        consec_unasso = gfuncs.get_max_consecutive(trk.is_associated, False)
        if consec_unasso == tparams.tracker.association.age.trk_destroy:
            destroy_trk_indices.append(trk_idx)
    trks = gfuncs.exclude_from_list(trks, destroy_trk_indices)

    # Destroy Prolonged Tracklet Candidates
    destroy_trkc_indices = []
    for trkc_idx, trk_cand in enumerate(trk_cands):
        consec_unasso = gfuncs.get_max_consecutive(trk_cand.is_associated, False)
        if consec_unasso == tparams.tracker.association.age.trkc_destroy:
            destroy_trkc_indices.append(trkc_idx)

    # Associate Detections with Tracklets
    if len(trks) != 0:
        # Get Association Match INFO
        matches_d2t, unasso_det_indices, unasso_trk_indices = \
            fasso.asso_det_tracklet(dets, trks, tparams.tracker.association.threshold_loose)

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
        for det_idx, det in enumerate(dets):
            new_trk_cand = TrackletCandidate(det, confs[det_idx, :], labels[det_idx, :])
            trk_cands.append(new_trk_cand)
            del new_trk_cand
    else:
        # Get Association Match
        matches_d2tc, unasso_det_indices, unasso_trkc_indices = \
            fasso.asso_det_trkcand(dets, trk_cands, tparams.tracker.association.threshold_tight)

        # Associate and Update Tracklet Candidates
        for match in matches_d2tc:
            matched_detection = dets[match[0]]
            matched_trk_cand = trk_cands[match[1]]

            # Update
            matched_trk_cand.update(matched_detection)
            trk_cands[match[1]] = matched_trk_cand
            del matched_trk_cand

        # Update Unassociated Tracklet Candidates
        # (Tracklet Candidate Update Scheme Needs to Incorporate different methods to that of the Tracklet's)
        for unasso_trkc_idx in unasso_trkc_indices:
            unasso_trk_cand = trk_cands[unasso_trkc_idx]

            # Update
            unasso_trk_cand.update()
            trk_cands[unasso_trkc_idx] = unasso_trk_cand
            del unasso_trk_cand

        # Generate New Tracklet Candidates with the Unassociated Detections
        for unasso_det_idx in unasso_det_indices:
            new_trk_cand = TrackletCandidate(dets[unasso_det_idx],
                                             confs[unasso_det_idx, :],
                                             labels[unasso_det_idx, :])
            trk_cands.append(new_trk_cand)
            del new_trk_cand

    # Generate New Tracklets from Tracklet Candidates
    if len(trk_cands) != 0:
        # Associate Only Tracklet Candidates with Detection Associated Consecutively for k frames
        selected_trkc_indices = []
        for trkc_idx, trk_cand in enumerate(trk_cands):
            max_consec_asso = gfuncs.get_max_consecutive(trk_cand.is_associated, True)
            if max_consec_asso == tparams.tracker.association.age.trk_init:
                selected_trkc_indices.append(trkc_idx)
        sel_trk_cands = gfuncs.select_from_list(trk_cands, selected_trkc_indices)

        # Initialize Tracklets
        for sel_trkc_idx, sel_trk_cand in enumerate(sel_trk_cands):
            selected_trkc_bbox, _ = fbbox.zx_to_bbox(sel_trk_cand.z[-1])
            selected_trkc_bbox_conf = sel_trk_cand.conf
            selected_trkc_bbox_label = sel_trk_cand.label

            # Generate Tracklet
            tracklet = Tracklet(selected_trkc_bbox,
                                selected_trkc_bbox_conf,
                                selected_trkc_bbox_label,
                                fidx, max_id + 1 + sel_trkc_idx, tparams.visualization.tracklet_color)
            new_trks.append(tracklet)
            del tracklet
        del sel_trk_cands

        # Destroy Associated Tracklet Candidates
        trk_cands = gfuncs.exclude_from_list(trk_cands, selected_trkc_indices)

    # Append New Tracklets
    for new_trk in new_trks:
        trks.append(new_trk)
        del new_trk
    del new_trks

    # Predict Tracklet States
    for trk_idx, trk in enumerate(trks):
        trk.predict()
        trks[trk_idx] = trk

    # Depth Inference (using depth_img)
    if hasattr(imgStruct, 'depth') is True:
        depth_list, depth_idx_list = multimodal_depth_inference(trks, imgStruct, tparams)

        # depth_list, depth_idx_list = depth_inference(trks, imgStruct.depth.processed, tparams.tracker.hist_bin.depth,
        #                                              tparams.image.depth.CLIP_MIN_DISTANCE, tparams.image.depth.CLIP_MAX_DISTANCE)
        for depth_idx, depth in enumerate(depth_list):
            trk = trks[depth_idx]
            trk.update_depth(depth, depth_idx_list[depth_idx])

    # Get Pseudo-inverse of Projection Matrix (color_cam_P)
    if hasattr(tparams.calibration, 'rgb_P') is True:
        rgb_cam_P_inverse = np.linalg.pinv(tparams.calibration.rgb_P)
    else:
        print("[WARNING] Projection Matrix Does NOT Exist!")
        rgb_cam_P_inverse = np.zeros((4, 3))

    # Image Coordinate (u,v,d) to Camera Coordinate (x,y,z)
    for trk_idx, trk in enumerate(trks):

        # Calculate 3D in camera coordinates
        trk.get_3d_cam_coord(rgb_cam_P_inverse, is_camera_static=False)

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
    # mmt_mesg = "[Frame: " + str(fidx) + "] Tracking " + str(len(trks)) + " Tracklets..!"
    # print(mmt_mesg)

    # MOT Message (2)
    mmt_mesg = "(Frame #%08d)__[Tracklets]: " % fidx
    mmt_trk_str = ""
    for _, trk in enumerate(trks):
        mmt_trk_str += "[%d]-" % trk.id
    mmt_mesg += mmt_trk_str
    print(mmt_mesg)

    # Get Maximum Tracklet ID among all Tracklets
    max_id = gfuncs.get_maximum_id(trks, max_trk_id)

    return trks, trk_cands, max_id























