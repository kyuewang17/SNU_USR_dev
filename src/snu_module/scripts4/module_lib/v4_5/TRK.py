"""
SNU Integrated Module v4.5
    - Multimodal Multiple Target Tracking
    - Changed Terminologies and Corrected Mis-used Terms
        - Tracklet -> Trajectory
        - Cost -> (corrected regarding to its definition)

"""

# Import Modules
import random
from utils.profiling import Timer
import cv2
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment as hungarian

# Import Custom Modules
import utils.patch as snu_patch
import utils.bounding_box as snu_bbox
import utils.general_functions as snu_gfuncs
import utils.histogram as snu_hist

# Import Class Objects
from tracking_objects import TrajectoryCandidate


class SNU_MOT(object):
    def __init__(self, opts):
        # Load Options
        self.opts = opts

        # Trajectories and Trajectory Candidates
        self.trks, self.trk_cands = [], []

        # Max Trajectory ID
        self.max_trk_id = 0

        # Frame Index
        self.fidx = None

        # Trajectory BBOX Size Limit
        self.trk_bbox_size_limits = None

    def __len__(self):
        return len(self.trks)

    def __repr__(self):
        return "SNU_MOT"

    def destroy_trajectories(self):
        # Destroy Trajectories with Following traits
        destroy_trk_indices = []
        for trk_idx, trk in enumerate(self.trks):
            # (1) Prolonged Consecutively Unassociated Trajectories
            if snu_gfuncs.get_max_consecutive(trk.is_associated, False) == \
                    self.opts.tracker.association["trk"]["destroy_age"]:
                destroy_trk_indices.append(trk_idx)

        # Remove Duplicate Indices
        destroy_trk_indices = list(set(destroy_trk_indices))
        self.trks = snu_gfuncs.exclude_from_list(self.trks, destroy_trk_indices)

    def destroy_trajectory_candidates(self):
        # Destroy Prolonged Trajectory Candidates
        destroy_trkc_indices = []
        for trkc_idx, trk_cand in enumerate(self.trk_cands):
            # (1) Trajectory Candidates with Abnormal Size
            if self.trk_bbox_size_limits is not None and trk_cand.z[-1] is not None:
                trkc_size = trk_cand.z[-1][4]*trk_cand.z[-1][5]
                if trkc_size < min(self.trk_bbox_size_limits) or trkc_size > max(self.trk_bbox_size_limits):
                    destroy_trkc_indices.append(trkc_idx)

            # (2) Prolonged Consecutively Unassociated Trajectory Candidates
            if snu_gfuncs.get_max_consecutive(trk_cand.is_associated, False) == \
                    self.opts.tracker.association["trk_cand"]["destroy_age"]:
                destroy_trkc_indices.append(trkc_idx)

        # Remove Duplicate Indices
        destroy_trkc_indices = list(set(destroy_trkc_indices))
        self.trk_cands = snu_gfuncs.exclude_from_list(self.trk_cands, destroy_trkc_indices)

    @staticmethod
    def associate(similarity_matrix, similarity_thresh, workers, works):
        # Hungarian Algorithm
        matched_indices = hungarian(-similarity_matrix)

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
            if similarity_matrix[m[0], m[1]] < similarity_thresh:
                unmatched_worker_indices.append(m[0])
                unmatched_work_indices.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, unmatched_worker_indices, unmatched_work_indices

    # Associate Detections with Trajectories
    def associate_detections_with_trajectories(self, sync_data_dict, detections):
        # Unpack Detections
        dets, confs, labels = detections["dets"], detections["confs"], detections["labels"]

        # Initialize Similarity Matrix Variable
        similarity_matrix = np.zeros((len(dets), len(self.trks)), dtype=np.float32)

        # Get (Color / Disparity) Frames
        color_frame = sync_data_dict["color"].get_data()
        # disparity_frame = sync_data_dict["disparity"].get_data(is_processed=False)

        # Normalize Disparity Frame to uint8 scale (0~255)
        if sync_data_dict["disparity"] is not None:
            normalized_disparity_frame = sync_data_dict["disparity"].get_normalized_data(
                min_value=0.0, max_value=255.0
            )
        else:
            normalized_disparity_frame = None

        # Concatenate
        if normalized_disparity_frame is not None:
            frame = np.dstack((color_frame, normalized_disparity_frame.astype(np.uint8)))
        else:
            frame = color_frame

        # Calculate Similarity Matrix
        for det_idx, det in enumerate(dets):
            for trk_idx, trk in enumerate(self.trks):
                det_zx = snu_bbox.bbox_to_zx(det)

                # Get Predicted State of Trajectory
                trk_bbox, trk_velocity = snu_bbox.zx_to_bbox(trk.pred_states[-1])

                # Get RGBD Patches
                det_patch = snu_patch.get_patch(frame, det)
                trk_patch = snu_patch.get_patch(frame, trk_bbox)

                # Resize RGBD Patches
                resized_det_patch = cv2.resize(det_patch, dsize=(64, 64))
                resized_trk_patch = cv2.resize(trk_patch, dsize=(64, 64))

                # Get RGBD Histograms of Detection and Trajectory Patch
                det_hist, det_hist_idx = snu_hist.histogramize_patch(
                    sensor_patch=resized_det_patch,
                    dhist_bin=128, min_value=0, max_value=255, count_window=None
                )
                trk_hist, trk_hist_idx = snu_hist.histogramize_patch(
                    sensor_patch=resized_trk_patch,
                    dhist_bin=128, min_value=0, max_value=255, count_window=None
                )

                # [1] Get Histogram Similarity
                if len(det_hist) == 0 or len(trk_hist) == 0:
                    hist_similarity = 1.0
                else:
                    hist_product = np.matmul(det_hist.reshape(-1, 1).transpose(), trk_hist.reshape(-1, 1))
                    hist_similarity = np.sqrt(hist_product / (np.linalg.norm(det_hist) * np.linalg.norm(trk_hist)))
                    hist_similarity = hist_similarity[0, 0]

                # [2] Get IOU Similarity
                aug_LT_coord = trk_bbox[0:2] - trk_velocity*0.5
                aug_RB_coord = trk_bbox[2:4] + trk_velocity*1.5
                aug_trk_bbox = np.concatenate((aug_LT_coord, aug_RB_coord))
                iou_similarity = 1.0 if snu_bbox.iou(det, aug_trk_bbox) > 0 else 0.0

                # [3] Get Distance Similarity
                l2_distance = snu_gfuncs.l2_distance_dim2(
                    x1=det_zx[0], y1=det_zx[1],
                    x2=trk.pred_states[-1][0], y2=trk.pred_states[-1][1]
                )
                dist_similarity = np.exp(-l2_distance)[0]

                # Get Total Similarity
                s_w_dict = self.opts.tracker.association["trk"]["similarity_weights"]
                similarity = \
                    s_w_dict["histogram"] * hist_similarity + \
                    s_w_dict["iou"] * iou_similarity + \
                    s_w_dict["distance"] * dist_similarity
                # print("T2D Similarity Value: {:.3f}".format(similarity))

                # to Similarity Matrix
                similarity_matrix[det_idx, trk_idx] = similarity

        # Get Similarity Threshold
        similarity_thresh = self.opts.tracker.association["trk"]['similarity_thresh']

        # Associate Using Hungarian Algorithm
        matches, unmatched_det_indices, unmatched_trk_indices = \
            self.associate(
                similarity_matrix=similarity_matrix, similarity_thresh=similarity_thresh,
                workers=dets, works=self.trks
            )

        # Update Associated Trajectories
        for match in matches:
            matched_det = detections['dets'][match[0]]
            matched_conf, matched_label = detections['confs'][match[0]], detections['labels'][match[0]]

            matched_trk = self.trks[match[1]]
            matched_trk.get_depth(sync_data_dict, self.opts)

            # If passed, update Trajectory
            matched_trk.update(self.fidx, matched_det, matched_conf)
            self.trks[match[1]] = matched_trk
            del matched_trk

        # Update Unassociated Trajectories
        for unasso_trk_idx in unmatched_trk_indices:
            unasso_trk = self.trks[unasso_trk_idx]

            unasso_trk.get_depth(sync_data_dict, self.opts)

            # Update Trajectory
            unasso_trk.update(self.fidx)
            self.trks[unasso_trk_idx] = unasso_trk
            del unasso_trk

        # Remove Associated Detections and Collect Residual Detections
        residual_dets = np.empty((len(unmatched_det_indices), 4))
        residual_confs, residual_labels = np.empty((len(unmatched_det_indices), 1)), np.empty((len(unmatched_det_indices), 1))
        for residual_det_idx, unasso_det_idx in enumerate(unmatched_det_indices):
            residual_dets[residual_det_idx, :] = detections['dets'][unasso_det_idx]
            residual_confs[residual_det_idx] = detections['confs'][unasso_det_idx]
            residual_labels[residual_det_idx] = detections['labels'][unasso_det_idx]
        detections = {'dets': residual_dets, 'confs': residual_confs, 'labels': residual_labels}

        return detections

    # Associate Detections with Detections
    def associate_resdets_trkcands(self, sync_data_dict, residual_detections):
        # Unpack Residual Detections
        dets, confs, labels = \
            residual_detections["dets"], residual_detections["confs"], residual_detections["labels"]

        # Initialize Similarity Matrix Variable
        similarity_matrix = np.zeros((len(dets), len(self.trk_cands)), dtype=np.float32)

        # Calculate Cost Matrix
        for det_idx, det in enumerate(dets):
            for trk_cand_idx, trk_cand in enumerate(self.trk_cands):
                if trk_cand.z[-1] is None:
                    similarity = -1
                else:
                    det_zx = snu_bbox.bbox_to_zx(det)
                    trk_cand_bbox, _ = snu_bbox.zx_to_bbox(trk_cand.z[-1])

                    # [1] Get IOU Similarity w.r.t. SOT-predicted BBOX
                    predicted_bbox = trk_cand.predict(sync_data_dict["color"].get_data(), trk_cand_bbox)
                    iou_similarity = snu_bbox.iou(det, predicted_bbox)

                    # [2] Get Distance Similarity
                    l2_distance = snu_gfuncs.l2_distance_dim2(
                        x1=det_zx[0], y1=det_zx[1],
                        x2=trk_cand.z[-1][0], y2=trk_cand.z[-1][1]
                    )
                    dist_similarity = np.exp(-l2_distance)[0]

                    # Get Total Similarity
                    s_w_dict = self.opts.tracker.association["trk_cand"]["similarity_weights"]
                    similarity = \
                        s_w_dict["iou"] * iou_similarity + \
                        s_w_dict["distance"] * dist_similarity
                    # print("D2D Similarity Value: {:.3f}".format(similarity))

                # to Similarity Matrix
                similarity_matrix[det_idx, trk_cand_idx] = similarity

        # Get Similarity Threshold
        similarity_thresh = self.opts.tracker.association["trk_cand"]["similarity_thresh"]

        # Associate using Hungarian Algorithm
        matches, unmatched_det_indices, unmatched_trk_cand_indices = \
            self.associate(
                similarity_matrix=similarity_matrix, similarity_thresh=similarity_thresh,
                workers=dets, works=self.trk_cands
            )

        # Update Associated Trajectory Candidates
        for match in matches:
            # Matched Detection
            matched_det = residual_detections['dets'][match[0]]
            matched_conf, matched_label = residual_detections['confs'][match[0]], residual_detections['labels'][match[0]]

            # Matched Trajectory Candidate
            matched_trk_cand = self.trk_cands[match[1]]

            # Update Trajectory Candidate
            if matched_label != matched_trk_cand.label:
                unmatched_det_indices.append(match[0]), unmatched_trk_cand_indices.append(match[1])
            else:
                matched_trk_cand.update(self.fidx, matched_det, matched_conf)
                self.trk_cands[match[1]] = matched_trk_cand
            del matched_trk_cand

        # Update Unassociated Trajectory Candidates
        for unasso_trkc_idx in unmatched_trk_cand_indices:
            unasso_trk_cand = self.trk_cands[unasso_trkc_idx]

            # Update Trajectory Candidate
            unasso_trk_cand.update(fidx=self.fidx)
            self.trk_cands[unasso_trkc_idx] = unasso_trk_cand
            del unasso_trk_cand

        # Generate New Trajectory Candidates with the Unassociated Detections
        for unasso_det_idx in unmatched_det_indices:
            new_trk_cand = TrajectoryCandidate(
                frame=sync_data_dict["color"].get_data(),
                bbox=residual_detections['dets'][unasso_det_idx],
                conf=residual_detections['confs'][unasso_det_idx],
                label=residual_detections['labels'][unasso_det_idx],
                init_fidx=self.fidx, opts=self.opts
            )
            self.trk_cands.append(new_trk_cand)
            del new_trk_cand

    def generate_new_trajectories(self, sync_data_dict, new_trks):
        # Select Trajectory Candidates that are consecutively associated for < k > frames
        selected_trkc_indices = []
        for trkc_idx, trk_cand in enumerate(self.trk_cands):
            if snu_gfuncs.get_max_consecutive(trk_cand.is_associated, True) == \
                    self.opts.tracker.association["trk"]["init_age"]:
                selected_trkc_indices.append(trkc_idx)
        sel_trk_cands = snu_gfuncs.select_from_list(self.trk_cands, selected_trkc_indices)

        # Initialize New Trajectories
        for sel_trkc_idx, sel_trk_cand in enumerate(sel_trk_cands):
            # Get New Trajectory ID
            new_trk_id = self.max_trk_id + 1 + sel_trkc_idx

            # Initialize New Trajectory
            disparity_frame = sync_data_dict["disparity"].get_data(is_processed=False) if sync_data_dict["disparity"] is not None else None
            new_trk = sel_trk_cand.init_tracklet(
                disparity_frame=disparity_frame,
                trk_id=new_trk_id, fidx=self.fidx, opts=self.opts
            )
            new_trks.append(new_trk)
            del new_trk
        del sel_trk_cands

        # Destroy Associated Trajectory Candidates
        self.trk_cands = snu_gfuncs.exclude_from_list(self.trk_cands, selected_trkc_indices)

        return new_trks

    def __call__(self, sync_data_dict, fidx, detections):
        if self.trk_bbox_size_limits is None:
            _width = sync_data_dict["color"].get_data().shape[1]
            _height = sync_data_dict["color"].get_data().shape[0]

            size_min_limit = 10
            size_max_limit = _width*_height / 2.0
            self.trk_bbox_size_limits = [size_min_limit, size_max_limit]

        # Load Point-Cloud XYZ Data
        if sync_data_dict["lidar"] is not None:
            sync_data_dict["lidar"].load_pc_xyz_data()

        # Initialize New Trajectory Variable
        new_trks = []

        # Destroy Trajectories with Following traits
        self.destroy_trajectories()

        # Destroy Prolonged Trajectory Candidates
        self.destroy_trajectory_candidates()

        # Associate Detections with Trajectories (return residual detections)
        if len(self.trks) != 0:
            detections = self.associate_detections_with_trajectories(
                sync_data_dict=sync_data_dict, detections=detections
            )

        # Associate Residual Detections with Trajectory Candidates
        if len(self.trk_cands) == 0:
            # Initialize New Tracklet Candidates
            for det_idx, det in enumerate(detections["dets"]):
                new_trk_cand = TrajectoryCandidate(
                    frame=sync_data_dict["color"].get_data(),
                    bbox=det, conf=detections["confs"][det_idx], label=detections["labels"][det_idx],
                    init_fidx=fidx, opts=self.opts
                )
                self.trk_cands.append(new_trk_cand)
                del new_trk_cand
        else:
            self.associate_resdets_trkcands(
                sync_data_dict=sync_data_dict, residual_detections=detections
            )

        # Generate New Trajectories from Trajectory Candidates
        new_trks = self.generate_new_trajectories(sync_data_dict=sync_data_dict, new_trks=new_trks)

        # Append New Trajectories and Update Maximum Trajectory ID
        for new_trk in new_trks:
            if new_trk.id >= self.max_trk_id:
                self.max_trk_id = new_trk.id
            self.trks.append(new_trk)
            del new_trk
        del new_trks

        # Get Pseudo-inverse of Projection Matrix
        color_P_inverse = sync_data_dict["color"].get_sensor_params().pinv_projection_matrix

        # Trajectory Prediction, Projection, and Message
        for trk_idx, trk in enumerate(self.trks):
            # Predict Tracklet States (time-ahead Kalman Prediction)
            trk.predict()

            # Project Image Coordinate State (x3) to Camera Coordinate State (c3)
            trk.img_coord_to_cam_coord(
                inverse_projection_matrix=color_P_inverse, opts=self.opts
            )

            # Compute RPY
            trk.compute_rpy(roll=0.0)

            # Adjust to Trajectory List
            self.trks[trk_idx] = trk
            del trk


if __name__ == "__main__":
    pass
