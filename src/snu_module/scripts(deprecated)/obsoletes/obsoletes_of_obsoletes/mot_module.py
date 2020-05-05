"""
SNU MOT Module v1.0
    - Code written by Kyuewang Lee (kyuewang5056@gmail.com)
    - from Seoul National University, South Korea

Baseline Code Link
    - [Link] https://github.com/abewley/sort
    - Bewley et. al. "Simple Online and Realtime Tracking"
    - [2016 IEEE International Conference on Image Processing, ICIP]

* Note that this code merely referenced the coding style of Bewley's work

"""
# Future Import
from __future__ import print_function

# Import Libraries
import cv2
import numpy as np
from itertools import groupby
from sklearn.utils.linear_assignment_ import linear_assignment as hungarian


# Get Maximum Tracklet ID among all Tracklets
def get_maximum_id(trks, max_trk_id):
    if len(trks) != 0:
        max_id = -1
        for trk in trks:
            if trk.id >= max_id:
                max_id = trk.id
    else:
        max_id = max_trk_id

    return max_id


# Get Maximum Consecutive Values in a List
def get_max_consecutive(src_list, find_val):
    max_count = 0
    for val, sub_list in groupby(src_list):
        sub_list = list(sub_list)

        if val is find_val:
            if len(sub_list) >= max_count:
                max_count = len(sub_list)
    return max_count


# Select Specific Indices from the List
def select_from_list(src_list, sel_indices):
    if len(sel_indices) == 0:
        return []
    else:
        selected_list = list(src_list[i] for i in sel_indices)
        return selected_list


# Exclude Specific Indices from the List
def exclude_from_list(src_list, excl_indices):
    if len(excl_indices) == 0:
        return src_list
    else:
        survived_indices = []
        for src_idx, _ in enumerate(src_list):
            if src_idx not in excl_indices:
                survived_indices.append(src_idx)
        return select_from_list(src_list, survived_indices)


# Extract Point Cloud in 2D BBOX
def extract_pc_in_box2d(pc, box2d):
    # pc: (N,2), box2d: (xmin,ymin,xmax,ymax)

    # In-built function
    def in_hull(p, hull):
        from scipy.spatial import Delaunay
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        return hull.find_simplex(p) >= 0

    box2d_corners = np.zeros((4, 2))
    box2d_corners[0, :] = [box2d[0], box2d[1]]
    box2d_corners[1, :] = [box2d[2], box2d[1]]
    box2d_corners[2, :] = [box2d[2], box2d[3]]
    box2d_corners[3, :] = [box2d[0], box2d[3]]
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d_corners)

    return pc[box2d_roi_inds, :], box2d_roi_inds


# Convert bbox to z(or x)
def bbox_to_zx(bbox, velocity=None):
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2
    y = bbox[1]+h/2
    if velocity is None:
        return np.array([x, y, w, h]).reshape(4, 1)
    else:
        return np.array([x, y, velocity[0], velocity[1], w, h]).reshape(6, 1)


# Convert z(or x) to bbox
def zx_to_bbox(zx):
    u1 = zx[0]-zx[4]/2
    v1 = zx[1]-zx[5]/2
    u2 = zx[0]+zx[4]/2
    v2 = zx[1]+zx[5]/2
    bbox = np.array([u1, v1, u2, v2]).reshape(4)
    velocity = np.array([zx[2], zx[3]]).reshape(2)
    return bbox, velocity


# Get Intersection BBOX Coordinates
def intersection_bbox(bbox1, bbox2):
    xx1 = np.maximum(bbox1[0], bbox2[0])
    yy1 = np.maximum(bbox1[1], bbox2[1])
    xx2 = np.minimum(bbox1[2], bbox2[2])
    yy2 = np.minimum(bbox1[3], bbox2[3])
    return np.array([xx1, yy1, (xx2-xx1), (yy2-yy1)]).reshape(4)


# Get Sub-image (Patch)
def get_patch(img, bbox):
    patch_row_min = np.maximum(bbox[1], 0)
    patch_row_max = np.minimum((bbox[1] + bbox[3]), img.shape[0])
    patch_col_min = np.maximum(bbox[0], 0)
    patch_col_max = np.minimum((bbox[0] + bbox[2]), img.shape[1])
    patch = img[patch_row_min:patch_row_max, patch_col_min:patch_col_max]
    return patch


# Generate 2-D Gaussian Window
def generate_gaussian_window(row_size, col_size, mean, stdev, min_val=0, max_val=1):
    # Construct 2-D Gaussian Window
    x, y = np.meshgrid(np.linspace(-1, 1, col_size), np.linspace(-1, 1, row_size))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-((d - mean) ** 2 / (2.0 * stdev ** 2)))

    cv2.imshow("Unnormalized Gaussian Map", g)

    # Normalize So that the matrix is scaled between "eps" and "1"
    gvec = g.flatten()
    gauss_map = min_val + ((g - np.min(gvec)) * (max_val - min_val) / (np.max(gvec) - np.min(gvec)))
    return gauss_map


# Blank-out 2-D Array Patch of a squared region
def blankout_from_patch(patch, blank_bbox, blank_val=0):
    blank_row_min = np.maximum(blank_bbox[1], 0)
    blank_row_max = np.minimum((blank_bbox[1] + blank_bbox[3]), patch.shape[0])
    blank_col_min = np.maximum(blank_bbox[0], 0)
    blank_col_max = np.minimum((blank_bbox[0] + blank_bbox[2]), patch.shape[1])
    patch[blank_row_min:blank_row_max, blank_col_min:blank_col_max] = blank_val
    return patch


# Calculate IOU between Two bboxes
def iou(bbox1, bbox2):
    common_bbox = intersection_bbox(bbox1, bbox2)
    w = np.maximum(0., common_bbox[2])
    h = np.maximum(0., common_bbox[3])
    wh = w * h
    o = wh / ((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
              + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - wh)
    return o


# [v3-Candidate] Calculate Velocity-Aware IOU between Two bboxes
def vel_iou(bbox1, velocity1, bbox2, velocity2):
    # Get Center Points of BBOXes and Compare them
    cx_1 = (bbox1[0] + bbox1[2]) / 2.0
    cy_1 = (bbox1[1] + bbox1[3]) / 2.0
    cx_2 = (bbox2[0] + bbox2[2]) / 2.0
    cy_2 = (bbox2[1] + bbox2[3]) / 2.0

    # Check for [width] case and Span
    if cx_1 <= cx_2:
        bbox1[2] += velocity1[0]
        bbox2[0] += velocity2[0]

        bbox1[0] -= velocity1[0]
        bbox2[2] -= velocity2[0]
    else:
        bbox1[0] += velocity1[0]
        bbox2[2] += velocity2[0]

        bbox1[2] -= velocity1[0]
        bbox2[0] -= velocity2[0]

    # Check for [height] case and Span
    if cy_1 <= cy_2:
        bbox1[1] += velocity1[1]
        bbox2[3] += velocity2[1]

        bbox1[3] -= velocity1[1]
        bbox2[1] -= velocity2[1]
    else:
        bbox1[3] += velocity1[1]
        bbox2[1] += velocity2[1]

        bbox1[1] -= velocity1[1]
        bbox2[3] -= velocity2[1]

    # Now finally calculate IOU (This is the velociy-aware IOU)
    return iou(bbox1, bbox2)


# Association Core Function
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
                trk_cand_bbox, _ = zx_to_bbox(trk_cand.z[-1])

                # Calculate Cost
                cost_val = iou(det, trk_cand_bbox)

            # Cost Matrix
            cost_matrix[det_idx, trk_cand_idx] = cost_val

    # Associate Using Hungarian Algorithm
    matches, unmatched_det_indices, unmatched_trk_cand_indices = \
        associate(cost_matrix, cost_thresh, dets, trk_cands)

    # Return
    return matches, unmatched_det_indices, unmatched_trk_cand_indices


# Detection-to-Tracklet Association
def asso_det_tracklet(dets, trks, pts2d, dist_vec, dhist_bin, cost_thresh):
    # Initialize Cost Matrix Variable
    cost_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)

    if (pts2d is not None) and (dist_vec is not None):
        # Calculate Cost Matrix
        for det_idx, det in enumerate(dets):
            # Extract Point Cloud from Detection bbox
            det_pts, assn_idx = extract_pc_in_box2d(pts2d, det)
            det_pts_depth = dist_vec[assn_idx]

            # Get Depth Histogram of PC from Detection
            det_z = bbox_to_zx(det)
            det_depth_hist = depth_histogram_estimation(det_pts, det_pts_depth, dhist_bin, is_weight=True, bbox_center=det_z[0:2])

            # Detection Depth Estimation
            max_bin = det_depth_hist.argmax() + 1
            det_depth = (max_bin / (1.0 * len(det_depth_hist))) * np.max(det_pts_depth)

            for trk_idx, trk in enumerate(trks):
                trk_bbox, _ = zx_to_bbox(trk.pred_states[-1])

                # IOU Cost
                cost_val = iou(det, trk_bbox)

                # BBOX Distance Cost
                if trk.depth is not None:
                    trk_bbox_pos = np.array([trk.x[0], trk.x[1], trk.depth])
                    det_bbox_pos = np.array([det_z[0], det_z[1], det_depth])
                    bbox_distance = np.linalg.norm(trk_bbox_pos-det_bbox_pos)
                else:
                    bbox_distance = np.linalg.norm(trk.x[0:2].reshape(2), det_z[0:2].reshape(2))
                cost_val *= (1 / (bbox_distance + 1e-6))

                # Cost Matrix
                cost_matrix[det_idx, trk_idx] = cost_val

        # Associate Using Hungarian Algorithm
        matches, unmatched_det_indices, unmatched_trk_indices = \
            associate(cost_matrix, cost_thresh, dets, trks)
    else:
        # Calculate Cost Matrix
        for det_idx, det in enumerate(dets):
            for trk_idx, trk in enumerate(trks):
                trk_bbox, _ = zx_to_bbox(trk.pred_states[-1])

                # Get Velocity
                det_vel = np.zeros(2)
                trk_vel = np.array(trk.pred_states[-1][2:4].reshape(2))

                # Calculate Cost
                # cost_val = iou(det, trk_cand_bbox)
                cost_val = vel_iou(det, det_vel, trk_bbox, trk_vel)

                # Cost Matrix
                cost_matrix[det_idx, trk_idx] = cost_val

        # Associate Using Hungarian Algorithm
        matches, unmatched_det_indices, unmatched_trk_indices = \
            associate(cost_matrix, cost_thresh, dets, trks)

    # Return
    return matches, unmatched_det_indices, unmatched_trk_indices


# Get Weighted Depth Histogram
def get_depth_histogram(depth_patch, dhist_bin, min_value, max_value):
    np.set_printoptions(threshold=np.inf)
    patch_max_value = np.max(depth_patch.flatten())
    if patch_max_value < 0:
        depth_hist = np.zeros(dhist_bin, dtype=int)
        depth_idx = np.zeros(dhist_bin+1, dtype=float)
    else:
        gauss_mean = 0
        gauss_stdev = 0.6

        # Gaussian Window for Histogram Counting (Maximal at the center)
        gauss_window = generate_gaussian_window(depth_patch.shape[0], depth_patch.shape[1],
                                                gauss_mean, gauss_stdev, min_val=1, max_val=10)
        #########################
        # Weighted Histogram
        depth_hist, depth_idx = np.histogram(depth_patch, bins=dhist_bin, range=(min_value, max_value), weights=gauss_window)
        # depth_hist, depth_idx = np.histogram(depth_patch, bins=dhist_bin, range=(min_value, max_value))

    return depth_hist, depth_idx


# Project Point Clouds with Calibration and Projection Matrices
def project_velo_to_image(lidar_pc, kitti_calib):
    Prect = kitti_calib.P_rectdfddd
    Rrect = kitti_calib.R_rect_20
    T_cam_velo = kitti_calib.T_cam2_velo

    # Prepare the Velodyne LiDAR Point Clouds
    pts3d_raw = lidar_pc
    pts3d = pts3d_raw[pts3d_raw[:, 3] > 0, :]
    pts3d[:, 3] = 1
    pts3d = pts3d.transpose()

    # 3D Points in Camera Reference Frame
    pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))

    # Before projecting, keep only points with z>0
    # (points that are in front of the camera)
    idx = (pts3d_cam[2, :] >= 0)
    pts3d_cam_corresp = pts3d_cam[:, idx]
    pts2d_cam = Prect.dot(pts3d_cam_corresp)

    pts2d_cam = np.divide(pts2d_cam[0:2, :], pts2d_cam[2, :])
    dist_vec = pts3d_cam_corresp[2, :].transpose()

    return pts2d_cam.transpose(), dist_vec.reshape((-1, 1))


# Assign Projected Point Clouds to Tracklets
def assign_projected_pointcloud(trks, pts2d, dist_vec):
    for trk_idx, trk in enumerate(trks):
        trk_bbox, _ = zx_to_bbox(trk.x)

        # Find Point Clouds
        trk_pts2d, assn_idx = extract_pc_in_box2d(pts2d, trk_bbox)
        trk_dist_vec = dist_vec[assn_idx]

        trk.update_lidar_points(trk_pts2d, trk_dist_vec)
        trks[trk_idx] = trk

    return trks


# Search Common Points
def search_common_points(points1, points2):
    common_x, cind_x_1, cind_x_2 = np.intersect1d(points1[:, 0], points2[:, 0], return_indices=True)
    common_y, cind_y_1, cind_y_2 = np.intersect1d(points1[:, 1], points2[:, 1], return_indices=True)

    cind_1 = np.intersect1d(cind_x_1, cind_y_1)
    cind_2 = np.intersect1d(cind_x_2, cind_y_2)

    if len(cind_1) == 0 or len(cind_2) == 0:
        return [], [], []
    else:
        if not np.array_equal(points1[cind_1, :], points2[cind_2, :]):
            assert 0
        else:
            cpoints = points1[cind_1, :]

        return cpoints, cind_1, cind_2


# Search Uncommon Points
def search_uncommon_points(points1, points2):
    _, cind_1, cind_2 = search_common_points(points1, points2)

    excl_points1 = []
    pidx_1 = []
    for pidx, point1 in enumerate(points1):
        if pidx not in cind_1:
            pidx_1.append(pidx)
            excl_points1.append(point1)

    excl_points2 = []
    pidx_2 = []
    for pidx, point2 in enumerate(points2):
        if pidx not in cind_2:
            pidx_2.append(pidx)
            excl_points2.append(point2)

    return np.array(excl_points1).reshape(-1, 2), np.array(excl_points2).reshape(-1, 2), pidx_1, pidx_2


# Depth Histogram Estimation (based on center-displacement)
def depth_histogram_estimation(ppoints, ppoint_depth, dhist_bin, is_weight=False, bbox_center=None):
    if is_weight is False:
        depth_hist, _ = np.histogram(ppoint_depth, bins=dhist_bin, range=(0, np.ceil(np.max(ppoint_depth))))
    else:
        cent_displ = ppoints - bbox_center.reshape(2)
        square_diff = np.multiply(cent_displ, cent_displ)
        square_root_sum = np.sqrt(square_diff[:, 0] + square_diff[:, 1]).reshape(-1, 1)
        hist_penalty_vec = 1.0 - (square_root_sum / float(np.max(square_root_sum)))

        # Weighted Histogram
        depth_hist, _ = np.histogram(ppoint_depth, bins=dhist_bin, range=(0, np.ceil(np.max(ppoint_depth))), weights=hist_penalty_vec)

    return depth_hist


# Two-stage Depth Inference
def depth_inference(trks, dhist_bin):
    # For Tracklets
    for trk1_idx, trk1 in enumerate(trks):
        for trk2_idx, trk2 in enumerate(trks):
            # Prevent Redundant Symmetric Calculation
            if trk1_idx < trk2_idx:
                trk1_ppoints = trk1.ppoints
                trk1_ppoint_depth = trk1.ppoint_depth
                trk2_ppoints = trk2.ppoints
                trk2_ppoint_depth = trk2.ppoint_depth

                # Check for Unique Points
                trk1_unique_points, trk2_unique_points, trk1_unique_indices, trk2_unique_indices = \
                    search_uncommon_points(trk1_ppoints, trk2_ppoints)

                # Net Depth Values
                trk1.update_lidar_points(trk1_unique_points, trk1_ppoint_depth[trk1_unique_indices])
                trk2.update_lidar_points(trk2_unique_points, trk2_ppoint_depth[trk2_unique_indices])

            # Update Tracklets
            trks[trk1_idx] = trk1
            trks[trk2_idx] = trk2

    # For Tracklets
    for trk_idx, trk in enumerate(trks):
        ppoint_depth = trk.ppoint_depth
        if len(ppoint_depth) != 0:
            depth_hist = depth_histogram_estimation(trk.ppoints, ppoint_depth, dhist_bin, is_weight=True, bbox_center=trk.x[0:2])
            trk.update_depth(depth_hist, np.max(ppoint_depth))
            trks[trk_idx] = trk

    return trks


# D435 Depth Inference (Two-stage)
def depth_inference_d435(trks, depth_img, dhist_bin):
    # How to speed-up? and calculate efficiently
    #
    if len(trks) != 0 and depth_img != []:
        # 2-dimensional List for storing net depth histogram
        net_depth_hist_matrix = [[None]*len(trks) for _ in range(len(trks))]

        # For Tracklets
        for trk1_idx, trk1 in enumerate(trks):
            trk1_state_bbox, _ = zx_to_bbox(trk1.x)
            trk1_state_bbox = trk1_state_bbox.astype(int)
            trk1_patch = get_patch(depth_img, trk1_state_bbox)
            for trk2_idx, trk2 in enumerate(trks):
                if trk1_idx < trk2_idx:
                    trk2_state_bbox, _ = zx_to_bbox(trk2.x)
                    trk2_state_bbox = trk2_state_bbox.astype(int)
                    trk2_patch = get_patch(depth_img, trk2_state_bbox)

                    # Get Common BBOX Region and Blank-out
                    #
                    # [DEBUGGING ISSUE]
                    # (1) - If all depth patch pixels are greater than the clipping distance,
                    #       all depth patch pixels are clipped and set to value of "-1"
                    #       in this case, the maximum value of the depth patch pixels becomes "-1"
                    #       ==>> Histogram Code (error)
                    common_bbox = intersection_bbox(trk1_state_bbox, trk2_state_bbox)
                    trk1_net_patch = blankout_from_patch(trk1_patch, common_bbox, blank_val=-1)
                    trk2_net_patch = blankout_from_patch(trk2_patch, common_bbox, blank_val=-1)

                    # Get Patch Histogram and Store
                    # ERROR IS OCCURRED HERE!
                    depth_hist_1 = get_depth_histogram(trk1_net_patch, dhist_bin)
                    depth_hist_2 = get_depth_histogram(trk2_net_patch, dhist_bin)
                    net_depth_hist_matrix[trk1_idx][trk2_idx] = depth_hist_1
                    net_depth_hist_matrix[trk2_idx][trk1_idx] = depth_hist_2
                elif trk1_idx == trk2_idx:
                    depth_hist = get_depth_histogram(trk1_patch, dhist_bin)
                    net_depth_hist_matrix[trk1_idx][trk2_idx] = depth_hist

        # Final Depth List Initialization
        depth_list = [None] * len(trks)

        # For Tracklets, infer depth
        # Later, consider weighting each depth portion with "meaningful distance" between tracklet objects
        for mat_row_idx in range(len(trks)):
            depth_probability = np.ones(dhist_bin)
            for mat_col_idx in range(len(trks)):
                # Current Depth Probability (Histogram)
                curr_depth_prob = net_depth_hist_matrix[mat_row_idx][mat_col_idx]

                # MAP-like inference
                depth_probability = np.multiply(depth_probability, curr_depth_prob)

            # Push into List
            depth_list[mat_row_idx] = depth_probability
    else:
        depth_list = []

    return depth_list


###########################################
# D435 Depth Inference (Simple Version)
def depth_inference_d435_simple(trks, depth_img, dhist_bin, min_value, max_value):
    if len(trks) != 0 and depth_img != []:
        depth_list = [None] * len(trks)
        depth_idx_list = [None] * len(trks)

        for trk_idx, trk in enumerate(trks):
            trk_state_bbox, _ = zx_to_bbox(trk.x)
            trk_state_bbox = trk_state_bbox.astype(int)

            # Get Depth Patch
            trk_depth_patch = get_patch(depth_img, trk_state_bbox)

            # Get Depth Histogram
            depth_histogram, depth_idx = get_depth_histogram(trk_depth_patch, dhist_bin, min_value, max_value)

            # Push into List
            depth_list[trk_idx] = depth_histogram
            depth_idx_list[trk_idx] = depth_idx
    else:
        depth_list = []
        depth_idx_list = []

    return depth_list, depth_idx_list














