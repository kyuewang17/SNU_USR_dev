"""
SNU Integrated Module v3.0
    - Multimodal Multiple Target Tracking

"""

# Import Module
import cv2
import numpy as np
import logging
from sklearn.utils.linear_assignment_ import linear_assignment as hungarian

# Import Custom Modules
import utils.patch as snu_patch
import utils.bounding_box as snu_bbox
import utils.general_functions as snu_gfuncs
import utils.histogram as snu_hist

# Import Class Objects
from class_objects import TrackletCandidate, Tracklet


class MultipleObjects(object):
    # Initialization
    def __init__(self, opts):
        # Define Tracklet and Tracklet Candidate List
        self.trks, self.trk_cands = [], []

        # List of Current Tracklet IDs
        self.trk_ids = []

        # Check if there were any Tracklets
        self.maximum_id = None

        # Options
        self.opts = opts

        # Color Camera Parameters (convert 2D image coords to 3D camera coords)
        # NOTE: make this an input argument for the __init__ function!
        self.color_params = {}

        # LiDAR Parameters (Rectification...etc)
        self.lidar_params = {}

        pass

    # Destructor
    def __del__(self):
        pass

    # Get Number of Tracklets
    def __len__(self):
        return len(self.trks)

    # Call as Function
    def __call__(self, sync_data_dict, fidx, detections, is_default_device=True):
        # Initialize New Tracklet Variable
        new_trks = []

        # Destroy Tracklets and Tracklet Candidates
        self.destroy_objects()

        # Associate Detections with Tracklets
        residual_detections = self.asso_dets_trks(
            sync_data_dict=sync_data_dict, detections=detections
        )

        # Associate Residual Detections with Tracklet Candidates
        self.asso_resdets_trkcand(
            residual_detections=residual_detections, fidx=fidx
        )

        # Generate New Tracklets from Tracklet Candidates
        new_trks = self.gen_trks_from_trk_cands(
            sync_data_dict=sync_data_dict, new_trks=new_trks, fidx=fidx
        )

        # Append New Tracklets to Original Tracklets
        for new_trk in new_trks:
            self.trks.append(new_trk)
            del new_trk
        del new_trks

        # Predict Tracklet States
        for trk_idx, trk in enumerate(self.trks):
            trk.predict()
            self.trks[trk_idx] = trk

        # Extract Tracklet IDs and Get Maximum ID
        for trk_idx, trk in enumerate(self.trks):
            self.trk_ids.append(trk.id)
        if len(self.trk_ids) != 0:
            self.maximum_id = max(self.trk_ids)

    # Destroy Objects(Tracklets and Tracklet Candidates)
    def destroy_objects(self):
        # Destroy Tracklets with following traits
        destroy_trk_indices = []
        for trk_idx, trk in enumerate(self.trks):
            # (1) Tracklets with tiny bounding box
            if trk.x[4] * trk[5] < 10:
                destroy_trk_indices.append(trk_idx)

            # (2) Prolonged Consecutively Unassociated Tracklets
            if snu_gfuncs.get_max_consecutive(trk.is_associated, False) == \
                    self.opts.tracker.association["trk_destroy_age"]:
                destroy_trk_indices.append(trk_idx)

            # Sort out same Tracklet ids
            destroy_trk_indices = list(set(destroy_trk_indices))
        self.trks = snu_gfuncs.exclude_from_list(self.trks, destroy_trk_indices)

        # Destroy Prolonged Tracklet Candidates
        destroy_trkc_indices = []
        for trkc_idx, trk_cand in enumerate(self.trk_cands):
            if snu_gfuncs.get_max_consecutive(trk_cand.is_associated, False) == \
                    self.opts.tracker.association["trkc_destroy_age"]:
                destroy_trkc_indices.append(trkc_idx)
        self.trk_cands = snu_gfuncs.exclude_from_list(self.trk_cands, destroy_trkc_indices)

    # Associate Detections with Tracklets
    def asso_dets_trks(self, sync_data_dict, detections):
        if len(self.trks) != 0:
            # Unpack Detections
            dets, confs, labels = detections["dets"], detections["confs"], detections["labels"]

            # Initialize Cost Matrix Variable
            cost_matrix = np.zeros((len(dets), len(self.trks)), dtype=np.float32)

            # Get Main Modal Frame (Tentative)
            # TODO: Later, consider multimodal images and Extract Multimodal Patch
            main_modal = "color"
            frame = sync_data_dict[main_modal].frame

            # Calculate Cost Matrix
            for det_idx, det in enumerate(dets):
                for trk_idx, trk in enumerate(self.trks):
                    det_zx = snu_bbox.bbox_to_zx(det)

                    # Get Predicted State of Tracklet
                    trk_bbox, _ = snu_bbox.zx_to_bbox(trk.pred_states[-1])

                    # Get Detection and Tracklet Patch Histograms
                    det_hist = snu_hist.histogramize_patch(
                        sensor_patch=snu_patch.get_patch(frame, det),
                        dhist_bin=128, min_value=0, max_value=255, count_window=None
                    )
                    trk_hist = snu_hist.histogramize_patch(
                        sensor_patch=snu_patch.get_patch(frame, trk_bbox),
                        dhist_bin=128, min_value=0, max_value=255, count_window=None
                    )

                    # Get Histogram Similarity
                    hist_similarity = 1.0
                    pass

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
            matches_d2t, unasso_det_indices, unasso_trk_indices = \
                associate(
                    cost_matrix=cost_matrix, cost_thresh=self.opts.tracker.association["cost_thresh_d2trk"],
                    workers=dets, works=self.trks
                )

            # Update Associated Tracklets
            for match in matches_d2t:
                # Matched Detection
                matched_det, matched_conf, matched_label = dets[match[0]], confs[match[0]], labels[match[0]]

                # Matched Tracklet
                matched_trk = self.trks[match[1]]

                # Update Tracklet
                if matched_label != matched_trk.label:
                    unasso_det_indices.append(match[0]), unasso_trk_indices.append(match[1])
                else:
                    matched_trk.update(matched_det, matched_conf)
                    self.trks[match[1]] = matched_trk
                del matched_trk

            # Update Unassociated Tracklets
            for unasso_trk_idx in unasso_trk_indices:
                unasso_trk = self.trks[unasso_trk_idx]

                # Update
                unasso_trk.update()
                self.trks[unasso_trk_idx] = unasso_trk
                del unasso_trk

            # Remove Associated Detections and Collect Residual Detections
            residual_dets = np.empty((len(unasso_det_indices), 4))
            residual_confs, residual_labels = np.empty((len(unasso_det_indices), 1)), np.empty((len(unasso_det_indices), 1))
            for residual_det_idx, unasso_det_idx in enumerate(unasso_det_indices):
                residual_dets[residual_det_idx, :] = dets[unasso_det_idx]
                residual_confs[residual_det_idx] = confs[unasso_det_idx]
                residual_labels[residual_det_idx] = labels[unasso_det_idx]

            # Pack Unassociated Detections
            residual_detections = {
                "dets": residual_dets, "confs": residual_confs, "labels": residual_labels
            }

            return residual_detections
        else:
            return detections

    # Associate Residual Detections with Tracklet Candidates
    def asso_resdets_trkcand(self, residual_detections, fidx):
        if len(self.trk_cands) == 0:
            # Initialize New Tracklet Candidates
            for res_det_idx, res_det in enumerate(residual_detections["dets"]):
                new_trk_cand = TrackletCandidate(
                    bbox=res_det, conf=residual_detections["confs"][res_det_idx], label=residual_detections["labels"][res_det_idx],
                    init_fidx=fidx
                )
                self.trk_cands.append(new_trk_cand)
                del new_trk_cand
        else:
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
            matches_d2tc, unasso_det_indices, unasso_trkc_indices = \
                associate(
                    cost_matrix=cost_matrix, cost_thresh=self.opts.tracker.association["cost_thresh_d2trkc"],
                    workers=dets, works=self.trk_cands
                )

            # Update Associated Tracklet Candidates
            for match in matches_d2tc:
                # Matched Detection
                matched_det, matched_conf, matched_label = dets[match[0]], confs[match[0]], labels[match[0]]

                # Matched Tracklet Candidate
                matched_trk_cand = self.trk_cands[match[1]]

                # Update Tracklet Candidate
                if matched_label != matched_trk_cand.label:
                    unasso_det_indices.append(match[0]), unasso_trkc_indices.append(match[1])
                else:
                    matched_trk_cand.update(fidx, matched_det, matched_conf)
                    self.trk_cands[match[1]] = matched_trk_cand
                del matched_trk_cand

            # Update Unassociated Tracklet Candidates
            for unasso_trkc_idx in unasso_trkc_indices:
                unasso_trk_cand = self.trk_cands[unasso_trkc_idx]

                # Update
                unasso_trk_cand.update(fidx=fidx)
                self.trk_cands[unasso_trkc_idx] = unasso_trk_cand
                del unasso_trk_cand

            # Generate New Tracklet Candidates with the Unassociated Detections
            for unasso_det_idx in unasso_det_indices:
                new_trk_cand = TrackletCandidate(
                    bbox=dets[unasso_det_idx], conf=confs[unasso_det_idx], label=labels[unasso_det_idx],
                    init_fidx=fidx
                )
                self.trk_cands.append(new_trk_cand)
                del new_trk_cand

    # Generate New Tracklets from Tracklet Candidates
    def gen_trks_from_trk_cands(self, sync_data_dict, new_trks, fidx):
        if len(self.trk_cands) != 0:
            # Associate Tracklet Candidates with Detection Associated Consecutively for < k > frames
            selected_trkc_indices = []
            for trkc_idx, trk_cand in enumerate(self.trk_cands):
                if snu_gfuncs.get_max_consecutive(trk_cand.is_associated, True) == \
                        self.opts.tracker.association["trk_init_age"]:
                    selected_trkc_indices.append(trkc_idx)
            sel_trk_cands = snu_gfuncs.select_from_list(self.trk_cands, selected_trkc_indices)

            # Initialize Tracklets
            for sel_trkc_idx, sel_trk_cand in enumerate(sel_trk_cands):
                selected_trkc_bbox, _ = snu_bbox.zx_to_bbox(sel_trk_cand.z[-1])

                # Get Initial Tracklet ID
                new_trk_id = (self.maximum_id+1 if self.maximum_id is not None else 1)

                # Get Initial Tracklet Depth
                # (initial value quickly calculated via disparity frame, histogram-based)
                disparity_patch = snu_patch.get_patch(
                    img=sync_data_dict["disparity"].frame, bbox=selected_trkc_bbox
                )
                disparity_hist, disparity_hist_idx = snu_hist.histogramize_patch(
                    sensor_patch=disparity_patch, dhist_bin=self.opts.tracker.disparity_params["hist_bin"],
                    min_value=self.opts.sensors.disparity["clip_distance"]["min"],
                    max_value=self.opts.sensors.disparity["clip_distance"]["max"]
                )

                # Generate Tracklet
                tracklet = Tracklet(
                    bbox=selected_trkc_bbox, conf=sel_trk_cand.asso_confs[-1], label=sel_trk_cand.label,
                    init_fidx=fidx, trk_id=new_trk_id
                )
                new_trks.append(tracklet)
                del tracklet
            del sel_trk_cands

            # Destroy Associated Tracklet Candidates
            self.trk_cands = snu_gfuncs.exclude_from_list(self.trk_cands, selected_trkc_indices)
        else:
            new_trks = []

        return new_trks

    # Multimodal Tracking Message
    def tracking_mesg(self, fidx):
        # Multimodal Tracking Message
        tracking_mesg = "Frame #[%08d] --> {Tracklets}: " % fidx
        tracklet_recursive_mesg = ""
        for trk_idx, trk in enumerate(self.trks):
            if trk_idx < len(self.trks) - 1:
                add_tracklet_mesg = "[%d]," % trk.id
            else:
                add_tracklet_mesg = "[%d]" % trk.id
            tracklet_recursive_mesg += add_tracklet_mesg
        mesg = tracking_mesg + tracklet_recursive_mesg
        print(mesg)

    # Wrap as ROS Message (in order to publish to ETRI)
    def wrap_tracks(self, odometry):
        out_tracks = []
        return out_tracks


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


# Standalone MOT Mode
def standalone_tracker():
    pass


# Namespace Function
if __name__ == "__main__":
    standalone_tracker()
    pass
