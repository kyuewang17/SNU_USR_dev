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
import copy
import cv2
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment as hungarian
#from scipy.optimize import linear_sum_assignment as hungarian

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
        #self.trk_bbox_size_limits = [8*8, 640*480*0.1]
        self.trk_bbox_size_limits = None

        # Set Timer Object
        self.test_timer = Timer(convert="FPS")

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
        #matched_indices = np.array(hungarian(-similarity_matrix))
        matched_indices = (hungarian(-similarity_matrix))

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
        # Unpack Detections and Concatenate Modalities
        dets, confs, labels, modals = [], [], [], []
        net_modals = list(set(detections.keys()))
        for modal, modal_detections in detections.items():
            if modal_detections["dets"].size != 0:
                dets.append(modal_detections["dets"])
                confs.append(modal_detections["confs"])
                labels.append(modal_detections["labels"])
                for _ in range(len(modal_detections["dets"])):
                    modals.append(modal)
        if len(dets) > 0:
            dets, confs, labels = \
                np.concatenate(dets, axis=0), np.concatenate(confs, axis=0), np.concatenate(labels, axis=0)

        # Initialize Similarity Matrix Variable
        similarity_matrix = np.zeros((len(dets), len(self.trks)), dtype=np.float32)

        # Get Key Modal Frames
        frame_objs = {}
        for modal in net_modals:
            sensor_data = sync_data_dict[modal]
            if sensor_data is not None:
                frame_objs[modal] = sync_data_dict[modal]
        frame_objs["disparity"] = sync_data_dict["disparity"]

        # Calculate Similarity Matrix
        for det_idx, det in enumerate(dets):
            for trk_idx, trk in enumerate(self.trks):
                # Check if Modality btw Detection and Trajectory Candidate is Equal (if not match then continue loop)
                if modals[det_idx] != trk.modal:
                    similarity_matrix[det_idx, trk_idx] = -1000.0
                    continue
                else:
                    modal = modals[det_idx]

                det_zx = snu_bbox.bbox_to_zx(det)

                # Get Predicted State of Trajectory
                trk_bbox, trk_velocity = snu_bbox.zx_to_bbox(trk.pred_states[-1])
                # self.test_timer.reset()
                # Get Appropriate Patches
                det_patch = frame_objs[modal].get_patch(bbox=det)
                trk_patch = frame_objs[modal].get_patch(bbox=trk_bbox)
                # print(self.test_timer.elapsed)
                patch_minmax = frame_objs[modal].get_type_minmax()

                # Skip Association Conditions
                if trk_patch.shape[0] <= 0 or trk_patch.shape[1] <= 0:
                    similarity_matrix[det_idx, trk_idx] = -1000.0
                    continue
                if trk.label != labels[det_idx]:
                    similarity_matrix[det_idx, trk_idx] = -1000.0
                    continue
                #if trk_patch.shape[0] >= 480*0.2 or trk_patch.shape[1] >= 640*0.2:
                #    similarity_matrix[det_idx, trk_idx] = -1000.0
                #    continue

                # Resize Patches
                resized_det_patch = cv2.resize(det_patch, dsize=(64, 64))
                resized_trk_patch = cv2.resize(trk_patch, dsize=(64, 64))

                # Get Histograms of Detection and Trajectory Patch
                det_hist, det_hist_idx = snu_hist.histogramize_patch(
                    sensor_patch=resized_det_patch, dhist_bin=128,
                    min_value=patch_minmax["min"], max_value=patch_minmax["max"], count_window=None
                )
                trk_hist, trk_hist_idx = snu_hist.histogramize_patch(
                    sensor_patch=resized_trk_patch, dhist_bin=128,
                    min_value=patch_minmax["min"], max_value=patch_minmax["max"], count_window=None
                )

                # [1] Get Histogram Similarity
                if len(det_hist) == 0 or len(trk_hist) == 0:
                    hist_similarity = 1.0
                else:
                    hist_product = np.matmul(det_hist.reshape(-1, 1).transpose(), trk_hist.reshape(-1, 1))
                    hist_similarity = np.sqrt(hist_product / (np.linalg.norm(det_hist) * np.linalg.norm(trk_hist)))
                    hist_similarity = hist_similarity[0, 0]
                # print(hist_similarity)

                # [2] Get IOU Similarity
                aug_LT_coord = trk_bbox[0:2] - trk_velocity*0.5
                aug_RB_coord = trk_bbox[2:4] + trk_velocity*1.5
                aug_trk_bbox = np.concatenate((aug_LT_coord, aug_RB_coord))
                # iou_similarity = 1.0 if snu_bbox.iou(det, aug_trk_bbox) > 0 else 0.0

                iou_similarity = snu_bbox.ioc(det, aug_trk_bbox, denom_comp=1)
                # iou_similarity = snu_bbox.iou(det, aug_trk_bbox)
                # iou_similarity = snu_bbox.iou(det, trk_bbox)
                # iou_similarity = 1.0

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
            matched_det, matched_conf, matched_label = dets[match[0]], confs[match[0]], labels[match[0]]

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

        # Collect Unassociated Detections
        res_dets, res_confs, res_labels = \
            np.empty((len(unmatched_det_indices), 4)), np.empty((len(unmatched_det_indices), 1)), np.empty((len(unmatched_det_indices), 1))
        res_modals = [None] * len(unmatched_det_indices)
        for res_det_idx, unasso_det_idx in enumerate(unmatched_det_indices):
            res_modals[res_det_idx] = modals[unasso_det_idx]
            res_dets[res_det_idx, :], res_confs[res_det_idx], res_labels[res_det_idx] = \
                dets[unasso_det_idx], confs[unasso_det_idx], labels[unasso_det_idx]

        # Convert to Multi-modal Detection Format
        detections = {}
        for modal in net_modals:
            detections[modal] = {
                "dets": np.array([], dtype=np.float32),
                "confs": np.array([], dtype=np.float32),
                "labels": np.array([], dtype=np.float32),
            }
        color_fr, thermal_fr = True, True
        for res_idx in range(res_dets.shape[0]):
            res_modal = res_modals[res_idx]
            res_det = res_dets[res_idx].reshape(1, 4)
            res_conf = res_confs[res_idx].reshape(1, 1)
            res_label = res_labels[res_idx].reshape(1, 1)

            if detections[res_modal]["dets"].size == 0:
                detections[res_modal]["dets"] = res_det
                detections[res_modal]["confs"] = res_conf
                detections[res_modal]["labels"] = res_label
            else:
                detections[res_modal]["dets"] = np.vstack((detections[res_modal]["dets"], res_det))
                detections[res_modal]["confs"] = np.vstack((detections[res_modal]["confs"], res_conf))
                detections[res_modal]["labels"] = np.vstack((detections[res_modal]["labels"], res_label))

        return detections

    # Associate Detections with Detections
    def associate_resdets_trkcands(self, sync_data_dict, residual_detections):
        # Unpack Residual Detections and Concatenate Modalities
        dets, confs, labels, modals = [], [], [], []
        for modal, modal_detections in residual_detections.items():
            if modal_detections["dets"].size != 0:
                dets.append(modal_detections["dets"])
                confs.append(modal_detections["confs"])
                labels.append(modal_detections["labels"])
                for _ in range(len(modal_detections["dets"])):
                    modals.append(modal)
        if len(dets) > 0:
            dets, confs, labels = \
                np.concatenate(dets, axis=0), np.concatenate(confs, axis=0), np.concatenate(labels, axis=0)

        # Initialize Similarity Matrix Variable
        similarity_matrix = np.zeros((len(dets), len(self.trk_cands)), dtype=np.float32)

        # Calculate Similarity Matrix
        for det_idx, det in enumerate(dets):
            for trk_cand_idx, trk_cand in enumerate(self.trk_cands):
                # Check if Modality btw Detection and Trajectory Candidate is Equal (if not match then continue loop)
                if modals[det_idx] != trk_cand.modal:
                    similarity_matrix[det_idx, trk_cand_idx] = -1000.0
                    continue
                else:
                    modal = modals[det_idx]

                # Get Similarity
                if trk_cand.z[-1] is None:
                    similarity = -1.0
                else:
                    det_zx = snu_bbox.bbox_to_zx(det)
                    trk_cand_bbox, trk_cand_vel = snu_bbox.zx_to_bbox(trk_cand.z[-1])

                    # [1] Get IOU Similarity w.r.t. SOT-predicted BBOX
                    predicted_bbox = trk_cand.predict(sync_data_dict[modal].get_data(), trk_cand_bbox)
                    iou_similarity = snu_bbox.iou(det, predicted_bbox)
                    # iou_similarity = snu_bbox.iou(det, trk_cand_bbox)

                    # IOC
                    #aug_LT_coord = trk_cand_bbox[0:2] - trk_cand_vel * 0.5
                    #aug_RB_coord = trk_cand_bbox[2:4] + trk_cand_vel * 1.5
                    #aug_trk_cand_bbox = np.concatenate((aug_LT_coord, aug_RB_coord))
                    # iou_similarity = 1.0 if snu_bbox.iou(det, aug_trk_bbox) > 0 else 0.0
                    # iou_similarity = snu_bbox.ioc(det, aug_trk_cand_bbox, denom_comp=1)

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
            matched_det, matched_conf, matched_label = dets[match[0]], confs[match[0]], labels[match[0]]
            matched_modal = modals[match[0]]

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
            modal = modals[unasso_det_idx]
            new_trk_cand = TrajectoryCandidate(
                frame=sync_data_dict[modal].get_data(), modal=modals[unasso_det_idx],
                bbox=dets[unasso_det_idx], conf=confs[unasso_det_idx], label=labels[unasso_det_idx],
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
        # NOTE: For Static Agent, use Color Modal Detection Results Only
        # if self.opts.agent_type == "static":
        #     trk_detections = {"color": detections["color"]}
        # else:
        #     if self.opts.time == "day":
        #         trk_detections = {"color": detections["color"]}
        #     else:
        #         trk_detections = {"thermal": detections["thermal"]}
        trk_detections = copy.deepcopy(detections)

        if self.trk_bbox_size_limits is None:
            _width = sync_data_dict["color"].get_data().shape[1]
            _height = sync_data_dict["color"].get_data().shape[0]

            size_min_limit = 10
            size_max_limit = _width*_height / 20.0
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
            trk_detections = self.associate_detections_with_trajectories(
                sync_data_dict=sync_data_dict, detections=trk_detections
            )

        # Associate Residual Detections with Trajectory Candidates
        if len(self.trk_cands) == 0:
            for modal, modal_detections in trk_detections.items():
                for det_idx, det in enumerate(modal_detections["dets"]):
                    # Initialize Trajectory Candidate
                    new_trk_cand = TrajectoryCandidate(
                        frame=sync_data_dict[modal].get_data(), modal=modal, bbox=det,
                        conf=modal_detections["confs"][det_idx], label=modal_detections["labels"][det_idx],
                        init_fidx=fidx, opts=self.opts
                    )
                    self.trk_cands.append(new_trk_cand)
                    del new_trk_cand
        else:
            self.associate_resdets_trkcands(
                sync_data_dict=sync_data_dict, residual_detections=trk_detections
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

        # Allocate Dictionary for Sensor Parameters
        trk_sensor_param_dict = {}
        for trk in self.trks:
            if trk.modal not in trk_sensor_param_dict.keys():
                trk_sensor_param_dict[trk.modal] = sync_data_dict[trk.modal].get_sensor_params()

        # Trajectory Prediction, Projection, and Message
        for trk_idx, trk in enumerate(self.trks):
            # Get Sensor Params
            sensor_params = trk_sensor_param_dict[trk.modal]

            # Get Pseudo-inverse of Projection Matrix
            Pinv = sensor_params.pinv_projection_matrix

            # Predict Trajectory States
            trk.predict()

            # Project Image Coordinate State (x3) to Camera Coordinate State (c3)
            if self.opts.agent_type == "dynamic":
                trk.img_coord_to_cam_coord(inverse_projection_matrix=Pinv, opts=self.opts)
            elif self.opts.agent_type == "static":
                trk.img_coord_to_ground_plane(sensor_params=sensor_params)

            # Compute RPY
            trk.compute_rpy(roll=0.0)

            # Adjust to Trajectory List
            self.trks[trk_idx] = trk
            del trk


if __name__ == "__main__":
    pass
