"""
SNU Integrated Module v4.0
    - Multimodal Multiple Target Tracking


"""

# Import Modules
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment as hungarian

# Import Custom Modules
import utils.patch as snu_patch
import utils.bounding_box as snu_bbox
import utils.general_functions as snu_gfuncs
import utils.histogram as snu_hist

# Import Class Objects
from class_objects_v4 import TrackletCandidate, Tracklet


class SNU_MOT(object):
    def __init__(self, opts):
        # Load Options
        self.opts = opts

        # Tracklets and Tracklet Candidates
        self.trks, self.trk_cands = [], []

        # Max Tracklet ID
        self.max_trk_id = None

        # Frame Index
        self.fidx = None

        # Tracklet BBOX Size Limit
        self.trk_bbox_size_lower_limit = 10

    def __len__(self):
        return len(self.trks)

    def __repr__(self):
        return "SNU_MOT"

    def destroy_tracklets(self):
        # Destroy Tracklets with Following traits
        destroy_trk_indices = []
        for trk_idx, trk in enumerate(self.trks):
            # (1) Tracklets with tiny bounding box
            if trk.x3[5] * trk.x3[6] < self.trk_bbox_size_lower_limit:
                destroy_trk_indices.append(trk_idx)

            # (2) Prolonged Consecutively Unassociated Tracklets
            if snu_gfuncs.get_max_consecutive(trk.is_associated, False) == \
                    self.opts.tracker.association["trk_destroy_age"]:
                destroy_trk_indices.append(trk_idx)

            # Remove Duplicate Indices
            destroy_trk_indices = list(set(destroy_trk_indices))
        self.trks = snu_gfuncs.exclude_from_list(self.trks, destroy_trk_indices)

    def destroy_tracklet_candidates(self):
        # Destroy Prolonged Tracklet Candidates
        destroy_trkc_indices = []
        for trkc_idx, trk_cand in enumerate(self.trk_cands):
            if snu_gfuncs.get_max_consecutive(trk_cand.is_associated, False) == \
                    self.opts.tracker.association["trkc_destroy_age"]:
                destroy_trkc_indices.append(trkc_idx)
        self.trk_cands = snu_gfuncs.exclude_from_list(self.trk_cands, destroy_trkc_indices)

    @staticmethod
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

    # Associate Detections with Tracklets
    def associate_detections_with_tracklets(self, sync_data_dict, detections):
        # Unpack Detections
        dets, confs, labels = detections["dets"], detections["confs"], detections["labels"]

        # Initialize Cost Matrix Variable
        cost_matrix = np.zeros((len(dets), len(self.trks)), dtype=np.float32)

        # Get Concatenated Frame
        color_frame = sync_data_dict["color"].get_data()
        disparity_frame = sync_data_dict["disparity"].frame

        # Normalize Disparity Frame
        d_max, d_min = disparity_frame.max(), disparity_frame.min()
        normalized_disparity_frame = \
            ((disparity_frame - d_min) / (d_max - d_min)) * 255

        # Concatenate
        rgbd_frame = np.dstack((color_frame, normalized_disparity_frame.astype(np.uint8)))

        # Calculate Cost Matrix
        for det_idx, det in enumerate(dets):
            for trk_idx, trk in enumerate(self.trks):
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

                # Get Height-Distance Similarity
                h_det, h_trk = det_zx[3], trk.pred_states[-1][6]
                l2_distance = snu_gfuncs.l2_distance_dim2(
                    x1=det_zx[0], y1=det_zx[1],
                    x2=trk.pred_states[-1][0], y2=trk.pred_states[-1][1]
                )
                hd_cost = min(h_det / h_trk, h_trk / h_det) ** l2_distance

                # Cost
                cost_val = iou_cost * hist_similarity * hd_cost
                # print(cost_val)

                # to Cost Matrix
                cost_matrix[det_idx, trk_idx] = cost_val

        # Get Cost Threshold
        cost_thresh = self.opts.tracker.association['cost_thresh_d2trk']

        # Associate Using Hungarian Algorithm
        matches, unmatched_det_indices, unmatched_trk_indices = \
            self.associate(cost_matrix, cost_thresh, dets, self.trks)

        # Update Associated Tracklets
        for match in matches:
            matched_det = detections['dets'][match[0]]
            matched_conf, matched_label = detections['confs'][match[0]], detections['labels'][match[0]]

            matched_trk = self.trks[match[1]]
            matched_trk.get_depth(sync_data_dict, self.opts)

            # If passed, update Tracklet
            matched_trk.update(self.fidx, matched_det, matched_conf)
            self.trks[match[1]] = matched_trk
            del matched_trk

        # Update Unassociated Tracklets
        for unasso_trk_idx in unmatched_trk_indices:
            unasso_trk = self.trks[unasso_trk_idx]

            unasso_trk.get_depth(sync_data_dict, self.opts)

            # Update
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

    def associate_resdets_trkcands(self, sync_data_dict, residual_detections):
        # Unpack Residual Detections
        dets, confs, labels = \
            residual_detections["dets"], residual_detections["confs"], residual_detections["labels"]

        # Initialize Cost Matrix Variable
        cost_matrix = np.zeros((len(dets), len(self.trk_cands)), dtype=np.float32)

        # Calculate Cost Matrix
        for det_idx, det in enumerate(dets):
            for trk_cand_idx, trk_cand in enumerate(self.trk_cands):
                if trk_cand.z[-1] is None:
                    cost_val = -1
                else:
                    det_zx = snu_bbox.bbox_to_zx(det)
                    trk_cand_bbox, _ = snu_bbox.zx_to_bbox(trk_cand.z[-1])

                    # [1] IOU Similarity
                    iou_cost = snu_bbox.iou(det, trk_cand_bbox)

                    # [2] Height-Distance Similarity
                    h_det, h_trkc = det_zx[3], trk_cand.z[-1][5]
                    l2_distance = snu_gfuncs.l2_distance_dim2(
                        x1=det_zx[0], y1=det_zx[1],
                        x2=trk_cand.z[-1][0], y2=trk_cand.z[-1][1]
                    )
                    hd_cost = min(h_det / h_trkc, h_trkc / h_det) ** l2_distance

                    # Cost
                    cost_val = iou_cost * hd_cost
                    # print(cost_val)

                # to Cost Matrix
                cost_matrix[det_idx, trk_cand_idx] = cost_val

        # Get Cost Threshold
        cost_thresh = self.opts.tracker.association['cost_thresh_d2trkc']

        # Associate using Hungarian Algorithm
        matches, unmatched_det_indices, unmatched_trk_cand_indices = \
            self.associate(
                cost_matrix=cost_matrix, cost_thresh=cost_thresh,
                workers=dets, works=self.trk_cands
            )

        # Update Associated Tracklet Candidates
        for match in matches:
            # Matched Detection
            matched_det = residual_detections['dets'][match[0]]
            matched_conf, matched_label = residual_detections['confs'][match[0]], residual_detections['labels'][match[0]]

            # Matched Tracklet Candidate
            matched_trk_cand = self.trk_cands[match[1]]

            # Update Tracklet Candidate
            if matched_label != matched_trk_cand.label:
                unmatched_det_indices.append(match[0]), unmatched_trk_cand_indices.append(match[1])
            else:
                matched_trk_cand.update(self.fidx, matched_det, matched_conf)
                self.trk_cands[match[1]] = matched_trk_cand
            del matched_trk_cand

        # Update Unassociated Tracklet Candidates
        for unasso_trkc_idx in unmatched_trk_cand_indices:
            unasso_trk_cand = self.trk_cands[unasso_trkc_idx]

            # Update
            unasso_trk_cand.update(fidx=self.fidx)
            self.trk_cands[unasso_trkc_idx] = unasso_trk_cand
            del unasso_trk_cand

        # Generate New Tracklet Candidates with the Unassociated Detections
        for unasso_det_idx in unmatched_det_indices:
            new_trk_cand = TrackletCandidate(
                bbox=residual_detections['dets'][unasso_det_idx],
                conf=residual_detections['confs'][unasso_det_idx],
                label=residual_detections['labels'][unasso_det_idx],
                init_fidx=self.fidx
            )
            self.trk_cands.append(new_trk_cand)
            del new_trk_cand

    def generate_new_tracklets(self, sync_data_dict, new_trks):
        # Associate Tracklet Candidates with Detection Associated Consecutively for < k > frames
        selected_trkc_indices = []
        for trkc_idx, trk_cand in enumerate(self.trk_cands):
            if snu_gfuncs.get_max_consecutive(trk_cand.is_associated, True) == \
                    self.opts.tracker.association["trk_init_age"]:
                selected_trkc_indices.append(trkc_idx)
        sel_trk_cands = snu_gfuncs.select_from_list(self.trk_cands, selected_trkc_indices)

        # Initialize New Tracklets
        for sel_trkc_idx, sel_trk_cand in enumerate(sel_trk_cands):
            # Get New Tracklet ID
            new_trk_id = self.max_trk_id + 1 + sel_trkc_idx

            # Initialize New Tracklet
            new_trk = sel_trk_cand.init_tracklet(
                disparity_frame=sync_data_dict["disparity"].get_data(),
                trk_id=new_trk_id, fidx=self.fidx, opts=self.opts
            )
            new_trks.append(new_trk)
            del new_trk
        del sel_trk_cands

        # Destroy Associated Tracklet Candidates
        self.trk_cands = snu_gfuncs.exclude_from_list(self.trk_cands, selected_trkc_indices)

        return new_trks

    def __call__(self, sync_data_dict, fidx, detections):
        # Initialize New Tracklet Variable
        new_trks = []

        # Destroy Tracklets with Following traits
        self.destroy_tracklets()

        # Destroy Prolonged Tracklet Candidates
        self.destroy_tracklet_candidates()

        # Associate Detections with Tracklets (return residual detections)
        if len(self.trks) != 0:
            detections = self.associate_detections_with_tracklets(
                    sync_data_dict=sync_data_dict, detections=detections
                )

        # Associate Residual Detections with Tracklet Candidates
        if len(self.trk_cands) == 0:
            # Initialize New Tracklet Candidates
            for det_idx, det in enumerate(detections["dets"]):
                new_trk_cand = TrackletCandidate(
                    bbox=det, conf=detections["confs"][det_idx], label=detections["labels"][det_idx],
                    init_fidx=fidx
                )
                self.trk_cands.append(new_trk_cand)
                del new_trk_cand
        else:
            self.associate_resdets_trkcands(
                sync_data_dict=sync_data_dict, residual_detections=detections
            )

        # Generate New Tracklets from Tracklet Candidates
        new_trks = self.generate_new_tracklets(sync_data_dict=sync_data_dict, new_trks=new_trks)

        # Append New Tracklets and Update Maximum Tracklet ID
        max_trk_id = -1
        for new_trk in new_trks:
            if new_trk.id >= max_trk_id:
                max_trk_id = new_trk.id
            self.trks.append(new_trk)
            del new_trk
        del new_trks
        self.max_trk_id = max_trk_id

        # Get Pseudo-inverse of Projection Matrix
        color_P_inverse = sync_data_dict["color"].sensor_params.pinv_projection_matrix

        # Tracklet Prediction, Projection, and Message
        for trk_idx, trk in enumerate(self.trks):
            # Predict Tracklet States (time-ahead Kalman Prediction)
            trk.predict()

            # Project Image Coordinate State (x3) to Camera Coordinate State (c3)
            trk.img_coord_to_cam_coord(
                inverse_projection_matrix=color_P_inverse, opts=self.opts
            )

            # Compute RPY
            trk.compute_rpy(roll=0.0)

            # Adjust to Tracklet List
            self.trks[trk_idx] = trk
            del trk


if __name__ == "__main__":
    pass
