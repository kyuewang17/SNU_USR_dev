"""
SNU Integrated Module v2.05
  - Multimodal Multi-target Tracking

"""
# Import Module
import copy
import numpy as np

# Import Source Modules
import snu_utils.patch as ptch
import snu_utils.bounding_box as fbbox
import snu_utils.general_functions as gfuncs
import snu_utils.data_association as fasso
import snu_utils.histogram as fhist
from class_objects import TrackletCandidate, Tracklet

# import rescue.force_thermal_align_iitp_final_night as rgb_t_align

# # Depth Inference Test {02}
# def mmt_depth_inference_test_2(trks, imgStruct_dict, opts):
#     if len(trks) != 0:
#         # Initialize Depth Histograms
#         depth_list, depth_idx_list = [None] * len(trks), [None] * len(trks)
#
#         # Get D435 Depth Image (aligned depth)
#         depth_frame = imgStruct_dict['depth'].frame.processed
#
#         # Compute Depth Histogram (values, bins) for all tracklets
#         for trk_idx, trk in enumerate(trks):
#             # Get Tracklet Bounding Box Position
#             trk_state_bbox, _ = fbbox.zx_to_bbox(trk.x)
#             trk_state_bbox = trk_state_bbox.astype(int)
#
#             # Get D435 Depth Patch with bounding box length enlarge factor
#             trk_depth_patch =
#
#     pass


# Depth Inference Test {01}
def mmt_depth_inference_test(trks, imgStruct_dict, opts):
    if len(trks) != 0:
        # Initialize Depth Histograms
        depth_list, depth_idx_list = [None] * len(trks), [None] * len(trks)

        # Get D435 Depth Image (aligned depth)
        depth_frame = imgStruct_dict['depth'].frame.processed

        # Compute Depth Histogram (values, bins) for all tracklets
        for trk_idx, trk in enumerate(trks):
            # Get Tracklet Bounding Box Position
            trk_state_bbox, _ = fbbox.zx_to_bbox(trk.x)
            trk_state_bbox = trk_state_bbox.astype(int)

            # Get D435 Depth Patches
            trk_depth_fpatch = ptch.get_patch(depth_frame, trk_state_bbox, opts.tracker.depth_params["extract_depth_roi_rate"])
            # trk_depth_spatch = ptch.get_patch(depth_frame, trk_state_bbox, enlarge_factor=opts.tracker.depth_params['spatch_len_factor'])
            # trk_depth_lpatch = ptch.get_patch(depth_frame, trk_state_bbox, enlarge_factor=opts.tracker.depth_params['lpatch_len_factor'])

            # Get Depth Histogram from Smaller Patch Area
            try:
                fdepth_hist, fdepth_hist_idx = \
                    fhist.histogramize_patch(trk_depth_fpatch, opts.tracker.depth_params['hist_bin'],
                                             min_value=0, max_value=opts.sensors.depth['clip_distance']['max'])
            except:
                fdepth_hist, fdepth_hist_idx = [], []
            # sdepth_hist, sdepth_hist_idx = \
            #     fhist.histogramize_patch(trk_depth_spatch, opts.tracker.depth_params['hist_bin'],
            #                              min_value=0, max_value=opts.sensors.depth['clip_distance']['max'])
            # ldepth_hist, ldepth_hist_idx = \
            #     fhist.histogramize_patch(trk_depth_lpatch, opts.tracker.depth_params['hist_bin'],
            #                              min_value=0, max_value=opts.sensors.depth['clip_distance']['max'])

            # Element-wise Multiplication of Three Histograms
            # depth_hist = np.power(np.multiply(np.multiply(fdepth_hist, sdepth_hist), ldepth_hist), 1/3.0)
            # depth_hist_idx = fdepth_hist_idx

            # (Tentative) show histogram
            # fhist.plot_histogram(depth_hist, depth_hist_idx, xlabel="Distance", ylabel="Instances")

            # Push into List
            depth_list[trk_idx] = fdepth_hist
            depth_idx_list[trk_idx] = fdepth_hist_idx
    else:
        depth_list, depth_idx_list = [], []

    return depth_list, depth_idx_list


def mmt_depth_inference(trks, imgStruct_dict, opts):
    if len(trks) != 0:
        # Initialize Depth Histograms
        depth_list, depth_idx_list = [None] * len(trks), [None] * len(trks)

        # Get D435 Depth Image (aligned to D435 RGB Camera)
        raw_depth_frame = imgStruct_dict['depth'].frame.raw
        depth_frame = imgStruct_dict['depth'].frame.processed
        depth_clip_value = opts.sensors.depth["clip_value"]

        # Get LIDAR Image (projected to D435 RGB Camera)
        lidar1_frame = imgStruct_dict['lidar1'].frame.processed
        lidar1_clip_value = opts.sensors.lidar1["clip_value"]

        for trk_idx, trk in enumerate(trks):
            # Get Tracklet Bounding Box Position
            trk_state_bbox, _ = fbbox.zx_to_bbox(trk.x)
            trk_state_bbox = trk_state_bbox.astype(int)

            # Get D435 Raw Depth Patch
            trk_raw_depth_patch = ptch.get_patch(raw_depth_frame, trk_state_bbox)

            # Get D435 Depth Patch
            trk_depth_patch = ptch.get_patch(depth_frame, trk_state_bbox)
            count_window_depth, _, depth_max_count = \
                gfuncs.generate_gaussian_window(trk_depth_patch.shape[0], trk_depth_patch.shape[1],
                                                opts.tracker.depth_params['hist_gaussian_mean'], opts.tracker.depth_params['hist_gaussian_stdev'],
                                                min_val=1, max_val=10)
            # Test
            # count_window_depth = np.ones((count_window_depth.shape[0], count_window_depth.shape[1]))

            # Get Weighted D435 Depth Histogram
            depth_hist, depth_hist_idx = \
                fhist.histogramize_patch(trk_depth_patch, opts.tracker.depth_params['hist_bin'],
                                         min_value=0, max_value=50000, count_window=count_window_depth)

            # Get LIDAR Image Patch if exists
            if lidar1_frame is not None:
                depth_patch_valid_pxl_number = \
                    gfuncs.get_pixel_numbers(trk_depth_patch, depth_clip_value, 'g')

                trk_lidar_patch = ptch.get_patch(lidar1_frame, trk_state_bbox)
                lidar_patch_valid_pxl_number = \
                    gfuncs.get_pixel_numbers(trk_lidar_patch, lidar1_clip_value, 'g')

                # (Depth / LIDAR) Count Weight Penalty
                count_weight = (depth_patch_valid_pxl_number + gfuncs.epsilon()) / \
                               (lidar_patch_valid_pxl_number + gfuncs.epsilon())

                # Count Weight-compensated Gaussian Window
                count_window_lidar, _, _ = \
                    gfuncs.generate_gaussian_window(trk_lidar_patch.shape[0], trk_lidar_patch.shape[1],
                                                    opts.tracker.depth_params['hist_gaussian_mean'], opts.tracker.depth_params['hist_gaussian_stdev'],
                                                    min_val=1, max_val=int(depth_max_count*count_weight))
                # Get Weighted LIDAR Histogram
                lidar_hist, lidar_hist_idx = fhist.histogramize_patch(trk_lidar_patch, opts.tracker.depth_params['hist_bin'],
                                                                      min_value=0, max_value=50000, count_window=count_window_lidar)
                # Merge D435 Depth Histogram and LIDAR Histogram
                if (depth_hist_idx == lidar_hist_idx).sum() == len(depth_hist_idx):
                    depth_hist = np.add(depth_hist, lidar_hist)
                else:
                    assert 0, "???"

            # Push into Lst
            depth_list[trk_idx] = depth_hist
            depth_idx_list[trk_idx] = depth_hist_idx
    else:
        depth_list, depth_idx_list = [], []

    return depth_list, depth_idx_list


# Multimodal Multi-target Tracker
def tracker(imgStruct_dict, fidx, detections, max_trk_id, opts, trks=[], trk_cands=[]):
    # Initialize New Tracklet Variable
    new_trks = []

    # Get Maximum Tracklet ID
    max_id = gfuncs.get_maximum_id(trks, max_trk_id)

    # Destroy Tracklets with too small bounding box
    tiny_destroy_trk_indices = []
    for trk_idx, trk in enumerate(trks):
        w, h = trk.x[4], trk.x[5]
        if w * h < 10:
            tiny_destroy_trk_indices.append(trk_idx)
    trks = gfuncs.exclude_from_list(trks, tiny_destroy_trk_indices)

    # Destroy Prolonged Consecutively Unassociated Tracklets
    destroy_trk_indices = []
    for trk_idx, trk in enumerate(trks):
        consec_unasso = gfuncs.get_max_consecutive(trk.is_associated, False)
        if consec_unasso == opts.tracker.association['trk_destroy_age']:
            destroy_trk_indices.append(trk_idx)
    trks = gfuncs.exclude_from_list(trks, destroy_trk_indices)

    # Destroy Prolonged Tracklet Candidates
    destroy_trkc_indices = []
    for trkc_idx, trk_cand in enumerate(trk_cands):
        consec_unasso = gfuncs.get_max_consecutive(trk_cand.is_associated, False)
        if consec_unasso == opts.tracker.association['trkc_destroy_age']:
            destroy_trkc_indices.append(trkc_idx)
    trk_cands = gfuncs.exclude_from_list(trk_cands, destroy_trkc_indices)

    # Associate Detections with Tracklets
    if len(trks) != 0:
        # Get Association Match Information
        matches_d2t, unasso_det_indices, unasso_trk_indices = \
            fasso.asso_det_tracklet(imgStruct_dict, detections, trks, opts.tracker.association['cost_thresh_d2trk'])

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
                # If passed, update Tracklet
                matched_trk.update(matched_det, matched_conf)
                trks[match[1]] = matched_trk
            del matched_trk

        # Update Unassociated Tracklets
        for unasso_trk_idx in unasso_trk_indices:
            unasso_trk = trks[unasso_trk_idx]

            # Update
            unasso_trk.update()
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
        for det_idx, det in enumerate(detections['dets']):
            # Initialize New Tracklet Candidate
            new_trk_cand = TrackletCandidate(det, detections['confs'][det_idx], detections['labels'][det_idx])
            trk_cands.append(new_trk_cand)
            del new_trk_cand
    else:
        # Get Association Match
        matches_d2tc, unasso_det_indices, unasso_trkc_indices = \
            fasso.asso_det_trkcand(imgStruct_dict, detections, trk_cands, opts.tracker.association['cost_thresh_d2trkc'])

        # Associate and Update Tracklet Candidates
        for match in matches_d2tc:
            matched_det = detections['dets'][match[0]]
            matched_conf, matched_label = detections['confs'][match[0]], detections['labels'][match[0]]

            matched_trk_cand = trk_cands[match[1]]

            # Update
            matched_trk_cand.update(matched_det, matched_conf)

            # < Double-check for Label Consistency >
            if matched_label != matched_trk_cand.label:
                unasso_det_indices.append(match[0]), unasso_trkc_indices.append(match[1])
                print("[TEST 2] Inconsistent Label Test 2")
            else:
                # If passed, update Tracklet Candidate
                matched_trk_cand.update(matched_det, matched_conf)
                trk_cands[match[1]] = matched_trk_cand
            del matched_trk_cand

        # Update Unassociated Tracklet Candidates
        for unasso_trkc_idx in unasso_trkc_indices:
            unasso_trk_cand = trk_cands[unasso_trkc_idx]

            # Update
            unasso_trk_cand.update()
            trk_cands[unasso_trkc_idx] = unasso_trk_cand
            del unasso_trk_cand

        # Generate New Tracklet Candidates with the Unassociated Detections
        for unasso_det_idx in unasso_det_indices:
            new_trk_cand = TrackletCandidate(detections['dets'][unasso_det_idx],
                                             detections['confs'][unasso_det_idx],
                                             detections['labels'][unasso_det_idx])
            trk_cands.append(new_trk_cand)
            del new_trk_cand

    # Generate New Tracklets from Tracklet Candidates
    if len(trk_cands) != 0:
        # Associate Tracklet Candidates with Detection Associated Consecutively for < k > frames
        selected_trkc_indices = []
        for trkc_idx, trk_cand in enumerate(trk_cands):
            max_consec_asso = gfuncs.get_max_consecutive(trk_cand.is_associated, True)
            if max_consec_asso == opts.tracker.association['trk_init_age']:
                selected_trkc_indices.append(trkc_idx)
        sel_trk_cands = gfuncs.select_from_list(trk_cands, selected_trkc_indices)

        # Initialize Tracklets
        for sel_trkc_idx, sel_trk_cand in enumerate(sel_trk_cands):
            selected_trkc_bbox, _ = fbbox.zx_to_bbox(sel_trk_cand.z[-1])

            # Generate Tracklet
            tracklet = Tracklet(selected_trkc_bbox,
                                sel_trk_cand.conf[-1],
                                sel_trk_cand.label,
                                fidx, max_id + 1 + sel_trkc_idx, opts.tracker.tracklet_colors, opts.tracker.trk_color_refresh_period)
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

    # # Check for out-of-slicing position(prediction)
    # rgb_to_thermal_obj = rgb_t_align.heuristic_camera_params(src_modal="rgb", dest_modal="thermal")
    # x_slicings, y_slicings = rgb_to_thermal_obj.x_slicings, rgb_to_thermal_obj.y_slicings
    # exclude_trk_indices = []
    # for trk_idx, trk in enumerate(trks):
    #     xp = trk.xp
    #     bbox_xp, _ = fbbox.zx_to_bbox(xp)
    #     bbox_xp__x = np.array([bbox_xp[0], bbox_xp[2]]).reshape(2, 1)
    #     bbox_xp__y = np.array([bbox_xp[1], bbox_xp[3]]).reshape(2, 1)
    #
    #     if bbox_xp__x[0] < x_slicings[0] or bbox_xp__x[1] > x_slicings[1]:
    #         if bbox_xp__y[0] < y_slicings[0] or bbox_xp__y[1] > y_slicings[1]:
    #             exclude_trk_indices.append(trk_idx)
    # trks = gfuncs.exclude_from_list(trks, exclude_trk_indices)

    # Multimodal Multi-target Depth Inference (multimodal inference)
    # depth_list, depth_idx_list = mmt_depth_inference(trks, imgStruct_dict, opts)
    depth_list, depth_idx_list = mmt_depth_inference_test(trks, imgStruct_dict, opts)
    # depth_list, depth_idx_list = mmt_depth_inference_test_2(trks, imgStruct_dict, opts)

    for depth_idx, depth_hist in enumerate(depth_list):
        # Plot HISTOGRAM (TEMP CODE for VERIFICATION)
        # fhist.plot_histogram(depth_hist, depth_idx_list[depth_idx])

        trk = trks[depth_idx]
        trk.update_depth(depth_hist, depth_idx_list[depth_idx], opts.tracker.depth_params['depth_update_weights'])
        trks[depth_idx] = trk
        del trk

    # # Exclude Too far away Tracklets
    # depth_destroy_trk_indices = []
    # for trk_idx, trk in enumerate(trks):
    #     if trk.depth >= opts.tracker.depth_params['max_depth_threshold'] or trk.depth <= 0:
    #         depth_destroy_trk_indices.append(trk_idx)
    # trks = gfuncs.exclude_from_list(trks, depth_destroy_trk_indices)

    # # If depth is too far, try estimate again with smaller roi
    # residual_trks, corresponding_trk_idx = [], []
    # for trk_idx, trk in enumerate(trks):
    #     if trk.depth >= opts.tracker.depth_params['max_depth_threshold'] or trk.depth <= 0:
    #         corresponding_trk_idx.append(trk_idx)
    #         residual_trks.append(trk)
    #     del trk
    # trks = gfuncs.exclude_from_list(trks, corresponding_trk_idx)
    #
    # opts.tracker.depth_params["extract_depth_roi_rate"] = 0.33
    #
    # res_depth_list, res_depth_idx_list = mmt_depth_inference_test(residual_trks, imgStruct_dict, opts)
    # for res_depth_idx, res_depth_hist in enumerate(res_depth_list):
    #     res_trk = residual_trks[res_depth_idx]
    #     res_trk.update_depth(res_depth_hist, res_depth_idx_list[res_depth_idx],  opts.tracker.depth_params['depth_update_weights'])
    #     residual_trks[res_depth_idx] = res_trk
    #     del res_trk
    #
    # # Merge Residual Tracklets
    # for res_trk in residual_trks:
    #     trks.append(res_trk)
    #     del res_trk
    # del residual_trks

    # Get Pseudo-inverse of Projection Matrix
    if opts.agent_type is "dynamic":
        if imgStruct_dict['rgb'].sensor_params['P'] is not None:
            rgb_cam_P_inverse = np.linalg.pinv(imgStruct_dict['rgb'].sensor_params['P'])
        else:
            rgb_cam_P_inverse = np.zeros((4, 3))
    elif opts.agent_type is "static":
        intrinsic_matrix = imgStruct_dict['rgb'].sensor_params.intrinsic_matrix
        extrinsic_matrix = imgStruct_dict['rgb'].sensor_params.extrinsic_matrix

        rgb_cam_P_inverse = np.linalg.pinv(np.matmul(intrinsic_matrix, extrinsic_matrix))
    else:
        assert 0, "Agent Type Error!"

    # Image Coordinate (u,v,d) to Camera Coordinate (x,y,z)
    for trk_idx, trk in enumerate(trks):

        # Calculate 3D in camera coordinates
        if opts.agent_type == "dynamic":
            trk.get_3d_cam_coord(rgb_cam_P_inverse, is_camera_static=False)
        elif opts.agent_type == "static":
            trk.get_3d_cam_coord(rgb_cam_P_inverse, is_camera_static=True)
        else:
            assert 0, "undefined agent type!"

        # Compute RPY
        trk.compute_rpy()

        # Feed-in
        trks[trk_idx] = trk

    # MMT Message
    mmt_mesg_fidx_part = "Frame #[%08d] --> {Tracklets}: " % fidx
    tracklet_recursive_mesg = ""
    for trk_idx, trk in enumerate(trks):
        if trk_idx < len(trks)-1:
            add_tracklet_mesg = "[%d]," % trk.id
        else:
            add_tracklet_mesg = "[%d]" % trk.id
        tracklet_recursive_mesg += add_tracklet_mesg
    mmt_mesg = mmt_mesg_fidx_part + tracklet_recursive_mesg
    print(mmt_mesg)

    # Get Maximum Tracklet ID among all Tracklets
    max_id = gfuncs.get_maximum_id(trks, max_trk_id)

    return trks, trk_cands, max_id
















