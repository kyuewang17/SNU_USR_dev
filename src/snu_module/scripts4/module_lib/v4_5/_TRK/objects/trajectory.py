"""
SNU Integrated Module v5.0
  - Code which defines Trajectory Object Class for Object Tracking

    - LATENT_TRAJECTORY : this is formerly named as "TRAJECTORY_CANDIDATE"
                          re-named for more intuitive-ness

    - TRAJECTORY

    - TRAJECTORIES

"""
import cv2
import copy
import math
import numpy as np

from module_lib.v4_5._TRK.objects import object_base, bbox, coordinates
from module_lib.v4_5._TRK.params.kalman_filter import KALMAN_FILTER
from module_lib.v4_5._TRK.utils.assignment import associate
from module_lib.v4_5._TRK.utils.histogram import histogramize_patch
from module_lib.v4_5._TRK.utils.depth import compute_depth
from module_lib.v4_5._TRK.utils.kcf import KCF_PREDICTOR


class LATENT_TRAJECTORY(object_base.object_instance):
    def __init__(self, frame, **kwargs):
        super(LATENT_TRAJECTORY, self).__init__(**kwargs)

        # Get Modal Information
        modal = kwargs.get("modal", "color")
        self.__modal = modal
        kwargs.pop("modal")

        # Get Tracker Options
        tracker_opts = kwargs.get("tracker_opts")
        assert tracker_opts is not None
        self.opts = tracker_opts
        kwargs.pop("tracker_opts")

        # Initialize Associated Detection List
        det_bbox, det_conf, is_associated = kwargs.get("det_bbox"), kwargs.get("det_conf"), kwargs.get("is_associated", True)
        assert isinstance(det_bbox, bbox.BBOX) and isinstance(is_associated, bool) and det_bbox.bbox_format == "LTRB"
        self.det_bboxes, self.det_confs, self.is_associated = [det_bbox], [det_conf], [is_associated]

        # Initialize Velocity Variable
        self.velocities = [np.array([0.0, 0.0])]

        # Initialize BBOX Predictor (KCF Module)
        self.BBOX_PREDICTOR = KCF_PREDICTOR(
            init_frame=frame, init_bbox=det_bbox, init_fidx=kwargs.get("init_fidx"),
            kcf_params=tracker_opts.latent_tracker["visual_tracker"]["kcf_params"]
        )

        # Set Iteration Counter
        self.__iter_counter = 0

    def __repr__(self):
        return "LATENT_TRK ID - [{}]".format(self.id)

    def __getitem__(self, idx):
        return {
            "id": self.id,
            "label": self.label,
            "asso_det_bbox": self.det_bboxes[idx],
            "asso_det_conf": self.det_confs[idx],
            "is_associated": self.is_associated[idx],
        }

    def next(self):
        try:
            iter_item = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return iter_item

    def get_modal(self):
        return self.__modal

    def get_depth(self, lidar_obj, pc_sampling_number):
        # Get LiDAR Object is None, set depth value as np.nan and Return Function
        if lidar_obj is None:
            return np.nan

        # TODO: Project LiDAR XYZ to UV-Coordinate, inside "z_bbox"
        uv_array, pc_distances = [], []

        # Compute Depth
        if len(uv_array) == 0:
            return np.nan
        else:
            depth_value = compute_depth(uv_array, pc_distances, self.det_bboxes[-1])

        # Append Depth
        return depth_value

    def get_intersection_bbox(self, other, **kwargs):
        raise NotImplementedError()

    def update(self, frame, **kwargs):
        # Get Frame Index and Append
        fidx = kwargs.get("fidx")
        assert isinstance(fidx, int) and fidx > self.frame_indices[-1]
        self.frame_indices.append(fidx)

        # Get Detection BBOX and Confidence, and get Observation Object
        prev_det_bbox = self.det_bboxes[-1]
        det_bbox, det_conf = kwargs.get("det_bbox"), kwargs.get("det_conf")
        if det_bbox is None and det_conf is None:
            # If Previous Detection Confidence is High Enough, then Predict BBOX via KCF
            # IDEA: Use Multi-modal Appearance for Visual Tracking Later on...?
            if self.det_confs[-1] > self.opts.latent_tracker["visual_tracker"]["activation_conf_thresh"]:

                # Predict BBOX
                pred_bbox, psr_value = self.BBOX_PREDICTOR.predict(frame=frame, roi_bbox=prev_det_bbox)

                # If PSR Value of Response is High Enough, then Update KCF Appearance Model
                if psr_value > self.opts.latent_tracker["visual_tracker"]["update_model_thresh"]:
                    self.BBOX_PREDICTOR.update(frame=frame, roi_bbox=pred_bbox)

                # Get Velocity
                velocity = pred_bbox - prev_det_bbox

                self.det_bboxes.append(pred_bbox)
                self.det_confs.append(psr_value)
                self.is_associated.append(True)
                self.velocities.append(velocity)

            else:
                self.det_bboxes.append(None)
                self.det_confs.append(None)
                self.is_associated.append(False)
                self.velocities.append(self.velocities[-1])

        elif det_bbox is not None and det_conf is not None:
            # NOTE: Calculate IOC between Detection BBOX and Latent Trajectory BBOX
            if prev_det_bbox is not None:
                # Convert Format
                prev_det_bbox.convert_bbox_fmt("LTRB")

                # Pre-compute LT and RB augmentation multiplier
                dx, dy = self.velocities[-1][0], self.velocities[-1][1]
                lt_x_aug_m = 0.5 if dx >= 0 else 1.5
                lt_y_aug_m = 0.5 if dy >= 0 else 1.5
                rb_x_aug_m, rb_y_aug_m = 2.0 - lt_x_aug_m, 2.0 - lt_y_aug_m

                # Augment Associated Detection BBOX
                aug_prev_det_bbox = copy.deepcopy(prev_det_bbox)
                aug_prev_det_bbox.lt_x = aug_prev_det_bbox.lt_x - abs(dx) * lt_x_aug_m
                aug_prev_det_bbox.lt_y = aug_prev_det_bbox.lt_y - abs(dy) * lt_y_aug_m
                aug_prev_det_bbox.rb_x = aug_prev_det_bbox.rb_x + abs(dx) * rb_x_aug_m
                aug_prev_det_bbox.rb_y = aug_prev_det_bbox.rb_y + abs(dy) * rb_y_aug_m
                aug_prev_det_bbox.adjust_coordinates()

                # Get Velocity-Augmented IOC
                ioc_value = aug_prev_det_bbox.get_ioc(det_bbox)
                if ioc_value > self.opts.latent_tracker["ioc_thresh"]:



            else:
                self.det_bboxes.append(None)
                self.det_confs.append(None)
                self.is_associated.append(False)
                self.velocities.append(self.velocities[-1])

                if ioc_value < 1e-3:
                    ioc_value = np.nan


            assert isinstance(det_bbox, bbox.BBOX)
            velocity = det_bbox - prev_det_bbox
            self.det_bboxes.append(det_bbox)
            self.det_confs.append(det_conf)
            self.is_associated.append(True)
            self.velocities.append(velocity)

        else:
            raise NotImplementedError()

    def update_bbox_predictor(self, frame, roi_bbox):
        assert isinstance(roi_bbox, bbox.BBOX)
        self.BBOX_PREDICTOR.update(frame=frame, roi_bbox=roi_bbox)

    def init_trk(self, lidar_obj, fidx, **kwargs):
        # Get Tracker Options
        tracker_opts = self.opts

        # Get New Trajectory ID
        trk_id = kwargs.get("trk_id")
        assert isinstance(trk_id, int)
        kwargs.pop("trk_id")

        # Get Depth
        pc_sampling_number = kwargs.get("pc_sampling_number")
        kwargs.pop("pc_sampling_number")
        init_depth = self.get_depth(
            lidar_obj=lidar_obj, pc_sampling_number=pc_sampling_number
        )

        # Initialize Trajectory
        trajectory = TRAJECTORY(
            label=self.label, id=trk_id, init_fidx=fidx,
            modal=self.get_modal(), tracker_opts=tracker_opts,
            det_bbox=self.det_bboxes[-1], det_conf=self.det_confs[-1],
            init_depth=init_depth, velocity=self.velocities[-1]
        )

        return trajectory


class TRAJECTORY(object_base.object_instance):
    def __init__(self, **kwargs):
        super(TRAJECTORY, self).__init__(**kwargs)

        # Get Modal Information
        modal = kwargs.get("modal", "color")
        self.__modal = modal
        kwargs.pop("modal")

        # Get Tracker Options
        tracker_opts = kwargs.get("tracker_opts")
        assert tracker_opts is not None
        self.opts = tracker_opts
        kwargs.pop("tracker_opts")

        # Load Kalman Filter Parameters (tentative)
        self.KALMAN_FILTER = KALMAN_FILTER(**kwargs)

        # Initialize Associated Detection List
        det_bbox, det_conf, is_associated = kwargs.get("det_bbox"), kwargs.get("det_conf"), kwargs.get("is_associated", True)
        assert isinstance(det_bbox, bbox.BBOX) and isinstance(is_associated, bool) and det_bbox.bbox_format == "LTRB"
        self.det_bboxes, self.det_confs, self.is_associated = [det_bbox], [det_conf], [is_associated]

        # Get Trajectory Depth Observation
        init_depth = kwargs.get("init_depth", 0.0)
        assert init_depth >= 0
        self.z_depths = [init_depth]

        # Get Velocity
        velocity = kwargs.get("velocity", np.array([0.0, 0.0]))
        assert isinstance(velocity, (np.ndarray, list, tuple))
        if isinstance(velocity, (list, tuple)):
            assert len(velocity) == 2
        else:
            assert velocity.size == 2
            if len(velocity.shape) > 1:
                velocity = velocity.reshape(-1)

        # Get Initial Observation Coordinate
        init_z = coordinates.OBSERVATION_COORD(
            bbox_object=det_bbox, dx=velocity[0], dy=velocity[1], depth=init_depth
        )

        # Initialize State and State Prediction
        self.x3 = init_z.to_state()
        x3p = self.KALMAN_FILTER.predict(self.x3.numpify())
        self.x3p = coordinates.STATE_IMAGE_COORD(input_arr=x3p)
        self.states, self.pred_states = [self.x3], [self.x3p]

        # Initialize Camera Coordinate States
        self.c3 = self.x3.to_camera_coord(**kwargs)
        self.cam_states = [self.c3]

        # Initialize Roll, Pitch, Yaw Variables
        self.roll, self.pitch, self.yaw = None, None, None

        # Initialize Action Classification Results
        self.pose = None
        self.pose_list = []

        # Set Kalman Filter Process Flag
        self.kalman_flag = {
            "update": False,
            "predict": False
        }

        # Set Iteration Counter
        self.__iter_counter = 0

    def __repr__(self):
        return "TRK ID - [{}]".format(self.id)

    def __add__(self, other):
        assert isinstance(other, (TRAJECTORY, TRAJECTORIES))
        if isinstance(other, TRAJECTORY):
            return TRAJECTORIES(objects=[self, other])
        else:
            return other + self

    def __getitem__(self, idx):
        return {
            "id": self.id,
            "label": self.label,
            "asso_det_bbox": self.det_bboxes[idx],
            "is_associated": self.is_associated[idx],
            "state": self.states[idx],
            "pred_state": self.pred_states[idx],
            "cam_state": self.cam_states[idx],
        }

    def next(self):
        try:
            iter_item = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return iter_item

    # NOTE: Experimental Functions
    def init_kalman_params(self):
        raise NotImplementedError()

    def reset_kalman_flags(self):
        self.kalman_flag["update"] = False
        self.kalman_flag["predict"] = False

    def get_modal(self):
        return self.__modal

    def get_size(self, **kwargs):
        # Get State Method
        state_method = kwargs.get("state_method", "x3")
        assert state_method in ["x3", "x3p"]

        # Return
        state_obj = getattr(self, state_method)
        return state_obj.get_size()

    def get_fidx_data(self, fidx):
        try:
            idx = self.frame_indices.index(fidx)
        except ValueError:
            return None
        return self[idx]

    def get_depth(self, **kwargs):
        # Get Depth Method
        depth_method = kwargs.get("depth_method", "x3")
        assert depth_method in ["x3", "x3p", "z"]

        # Get Frame Index
        fidx = kwargs.get("fidx")
        if fidx is not None:
            assert fidx in self.frame_indices
            idx = self.frame_indices.index(fidx)
        else:
            idx = len(self) - 1

        # for depth method,
        if depth_method == "x3":
            return self.states[idx].depth
        elif depth_method == "x3p":
            return self.pred_states[idx].depth
        else:
            return self.z_depths[idx]

    def compute_z_depth(self, lidar_obj, **kwargs):
        # Get LiDAR Object is None, set depth value as np.nan and Return Function
        if lidar_obj is None:
            self.z_depths.append(np.nan)
            return

        # If Detection(observation) BBOX is None, set predicted depth state as z_depth value and Return Function
        det_bbox = self.det_bboxes[-1]
        if det_bbox is None:
            self.z_depths.append(self.x3p.depth)
            return

        # Get LiDAR Sampling Number
        pc_sampling_number = kwargs.get("pc_sampling_number")

        # TODO: Project LiDAR XYZ to UV-Coordinate, inside "z_bbox"
        uv_array, pc_distances = [], []

        # Compute Depth
        if len(uv_array) == 0:
            depth_value = self.z_depths[-1]
        else:
            depth_value = compute_depth(uv_array, pc_distances, det_bbox)

        # Append Depth
        self.z_depths.append(depth_value)

    def update(self, lidar_obj, **kwargs):
        # Get LiDAR Sampling Number
        pc_sampling_number = kwargs.get("pc_sampling_number")
        kwargs.pop("pc_sampling_number")

        # Get Frame Index and Append
        fidx = kwargs.get("fidx")
        assert isinstance(fidx, int) and fidx > self.frame_indices[-1]
        self.frame_indices.append(fidx)

        # Get Detection BBOX and Confidence, and get Observation Object
        det_bbox, det_conf = kwargs.get("det_bbox"), kwargs.get("det_conf")
        if det_bbox is None and det_conf is None:
            self.det_bboxes.append(None)
            self.det_confs.append(None)
            self.is_associated.append(False)

            # Compute Depth from Detection(observation)
            self.compute_z_depth(lidar_obj=lidar_obj, pc_sampling_number=pc_sampling_number)

            # Think State Prediction as Observation
            z = self.x3p.to_observation_coord()

        elif det_bbox is not None and det_conf is not None:
            assert isinstance(det_bbox, bbox.BBOX)
            self.det_bboxes.append(det_bbox)
            self.det_confs.append(det_conf)
            self.is_associated.append(True)

            # Compute Depth from Detection(observation)
            self.compute_z_depth(lidar_obj=lidar_obj, pc_sampling_number=pc_sampling_number)

            # Get Velocity (detection - previous associated detection)
            # -> if previous association is None, use previous state
            if self.det_bboxes[-2] is not None:
                vel_arr = det_bbox - self.det_bboxes[-2]
            else:
                vel_arr = det_bbox - self.x3.to_bbox()

            # Get Observation Coordinate
            z = coordinates.OBSERVATION_COORD(bbox_object=det_bbox, dx=vel_arr[0], dy=vel_arr[1], depth=self.z_depths[-1])

        else:
            raise AssertionError("Input Error")

        # Pop KWARGS
        kwargs.pop("fidx"), kwargs.pop("det_bbox"), kwargs.pop("det_conf")

        # Kalman Update
        x3 = self.KALMAN_FILTER.update(self.x3p.numpify(), z.numpify())

        # Transform into State Image Coordinate
        self.x3 = coordinates.STATE_IMAGE_COORD(input_arr=x3)
        self.states.append(self.x3)

        # Update Camera Coordinate States
        self.c3 = self.x3.to_camera_coord(**kwargs)
        self.cam_states.append(self.c3)

        # Set Kalman Filter Update Flag as True
        self.kalman_flag["update"] = True

    def predict(self):
        # Kalman Prediction
        x3p = self.KALMAN_FILTER.predict(self.x3.numpify())

        # Transform into State Image Coordinate
        self.x3p = coordinates.STATE_IMAGE_COORD(input_arr=x3p)
        self.pred_states.append(self.x3p)

        # Set Kalman Filter Predict Flag as True
        self.kalman_flag["predict"] = True

    def compute_rpy(self, roll=0.0):
        c3_np = self.c3.numpify()
        direction_vec = c3_np[3:6].reshape(3, 1)

        # Roll needs additional information
        self.roll = roll

        # Pitch
        denum = np.sqrt(direction_vec[0][0] ** 2 + direction_vec[1][0] ** 2)
        self.pitch = math.atan2(direction_vec[2][0], denum)

        # Yaw
        self.yaw = math.atan2(direction_vec[1][0], direction_vec[0][0])

    def get_intersection_bbox(self, other, **kwargs):
        valid_types = (
            bbox.BBOX,
            coordinates.OBSERVATION_COORD, coordinates.STATE_IMAGE_COORD,
            TRAJECTORY
        )
        assert isinstance(other, valid_types)

        # Get Return Format
        return_fmt = kwargs.get("return_fmt", "LTRB")
        assert return_fmt in ["LTRB", "LTWH", "XYWH"]

        # Get State Method
        state_method = kwargs.get("state_method", "x3p")
        assert state_method in ["x3", "x3p"]

        # Get State
        state_obj = getattr(self, state_method)

        # for input object type,
        if isinstance(other, TRAJECTORY):
            other_state_method = kwargs.get("other_state_method", state_method)
            assert other_state_method in ["x3", "x3p"]
            other_state_obj = getattr(other, other_state_method)

            # Get Intersection BBOX
            return state_obj.get_intersection_bbox(other=other_state_obj, return_fmt=return_fmt)

        else:
            # Get Intersection BBOX
            return state_obj.get_intersection_bbox(other=other, return_fmt=return_fmt)

    def compute_association_cost(self, frame, det_bbox, **kwargs):
        assert isinstance(det_bbox, bbox.BBOX)

        # Get Frame dtype Info
        frame_dtype_info = np.iinfo(frame.dtype)

        # Get Patch Resize Size
        patch_resize_sz = kwargs.get("patch_resize_sz", (64, 64))
        assert isinstance(patch_resize_sz, tuple)

        # Get Histogram Bin Number
        dhist_bin = kwargs.get("dhist_bin", 32)
        assert isinstance(dhist_bin, int) and dhist_bin > 0

        # Get Predicted State of Trajectory, get its bbox version
        x3p = self.x3p
        x3p_bbox = x3p.to_bbox(conversion_fmt="LTRB")

        # Pre-compute LT and RB augmentation multiplier
        lt_x_aug_m = 0.5 if x3p.dx >= 0 else 1.5
        lt_y_aug_m = 0.5 if x3p.dy >= 0 else 1.5
        rb_x_aug_m, rb_y_aug_m = 2.0 - lt_x_aug_m, 2.0 - lt_y_aug_m

        # Velocity-based BBOX Augmentation (using predicted state)
        aug_x3p_bbox = copy.deepcopy(x3p_bbox)
        aug_x3p_bbox.lt_x = aug_x3p_bbox.lt_x - abs(x3p.dx) * lt_x_aug_m
        aug_x3p_bbox.lt_y = aug_x3p_bbox.lt_y - abs(x3p.dy) * lt_y_aug_m
        aug_x3p_bbox.rb_x = aug_x3p_bbox.rb_x + abs(x3p.dx) * rb_x_aug_m
        aug_x3p_bbox.rb_y = aug_x3p_bbox.rb_y + abs(x3p.dy) * rb_y_aug_m
        aug_x3p_bbox.adjust_coordinates()

        # (1) Get Velocity-Augmented IOC
        ioc_similarity = aug_x3p_bbox.get_ioc(det_bbox, denom_comp="other")
        if ioc_similarity < 1e-3:
            return np.nan

        # Get Detection and Trajectory Patch
        # NOTE: If "get_patch" is slow, change code to get patch w.r.t. frame image
        det_patch, trk_patch = det_bbox.get_patch(frame=frame), x3p.get_patch(frame=frame)
        resized_det_patch = cv2.resize(det_patch, dsize=patch_resize_sz, interpolation=cv2.INTER_NEAREST)
        resized_trk_patch = cv2.resize(trk_patch, dsize=patch_resize_sz, interpolation=cv2.INTER_NEAREST)

        # Get Histograms of Resized Detection and Trajectory Patch
        det_hist, det_hist_idx = histogramize_patch(
            resized_det_patch, dhist_bin, frame_dtype_info.min, frame_dtype_info.max,
            count_window=None, linearize=True
        )
        trk_hist, trk_hist_idx = histogramize_patch(
            resized_trk_patch, dhist_bin, frame_dtype_info.min, frame_dtype_info.max,
            count_window=None, linearize=True
        )

        # (2) Get Histogram Similarity (cosine metric)
        if len(det_hist) == 0 or len(trk_hist) == 0:
            hist_similarity = 1.0
        else:
            hist_product = np.matmul(det_hist.reshape(-1, 1).transpose(), trk_hist.reshape(-1, 1))
            hist_similarity = np.sqrt(hist_product / (np.linalg.norm(det_hist) * np.linalg.norm(trk_hist)))
            hist_similarity = hist_similarity[0, 0]

        # (3) Get Distance Similarity
        l2_distance = np.linalg.norm(x3p_bbox - det_bbox)
        dist_similarity = np.exp(-l2_distance)[0]

        # Get Similarity Weights
        s_w_dict = self.opts.association["trk"]["similarity_weights"]
        s_w = np.array([s_w_dict["intersection"], s_w_dict["histogram"], s_w_dict["distance"]]).reshape(3, 1)

        # Weighted-Sum of Each Similarities
        similarity = np.matmul(s_w.T, np.array([ioc_similarity, hist_similarity, dist_similarity]).reshape(3, 1))

        # Cost is negative similarity
        cost = -similarity

        # Return
        return cost

    def classify_pose(self):
        raise NotImplementedError()


class TRAJECTORIES(object_base.object_instances):
    def __init__(self, **kwargs):
        super(TRAJECTORIES, self).__init__(**kwargs)

        # Set Trajectories
        self.trajectories = self.objects

        # Set Trajectory ID List
        self.ids = self.object_ids

        # Iteration Counter
        self.__iteration_counter = 0

        # Set Kalman Filter Process Flag
        self.kalman_flag = {
            "update": False,
            "predict": False
        }

    def __add__(self, other):
        self.append(other=other)
        return self

    def __getitem__(self, idx):
        return self.trajectories[idx]

    def __setitem__(self, idx, val):
        assert isinstance(val, TRAJECTORY)
        self.trajectories[idx] = val

    def __iter__(self):
        return self

    def next(self):
        try:
            return_value = self[self.__iteration_counter]
        except IndexError:
            self.__iteration_counter = 0
            raise StopIteration
        self.__iteration_counter += 1
        return return_value

    def append(self, other):
        assert isinstance(other, (TRAJECTORY, TRAJECTORIES))
        if isinstance(other, TRAJECTORY):
            self.trajectories.append(other)
            self.ids.append(other.id)
        else:
            for other_object in other:
                self.trajectories.append(other_object)
                self.ids.append(other_object.id)
        self.arrange_objects(arrange_method="id")

    def destroy_trajectories(self, **kwargs):
        raise NotImplementedError()

    def get_maximum_id(self):
        if len(self.object_ids) == 0:
            return None
        else:
            return max(self.object_ids)

    def reset_kalman_flags(self):
        for trk in self:
            trk.reset_kalman_flags()
        self.kalman_flag["update"] = False
        self.kalman_flag["predict"] = False

    def update(self, frame, lidar_obj, detections, fidx, **kwargs):
        if len(self) == 0:
            return detections

        # Change Association Activation Flag
        self.__is_updated = True

        # Get Cost Threshold
        cost_thresh = kwargs.get("cost_thresh")
        if cost_thresh is not None:
            assert 0 <= cost_thresh <= 1

        # Unpack Detections
        dets, confs, labels = detections["dets"], detections["confs"], detections["labels"]

        # Initialize Cost Matrix
        cost_matrix = np.zeros((len(dets), len(self)), dtype=np.float32)

        # Compute Cost Matrix (for every detection BBOX, for every trajectory objects)
        det_bboxes = []
        for det_idx, det in enumerate(dets):
            det_bbox = bbox.BBOX(bbox_format="LTRB", lt_x=det[0], lt_y=det[1], rb_x=det[2], rb_y=det[3])
            det_bboxes.append(det_bbox)
            det_conf, det_label = confs[det_idx], labels[det_idx]

            for trk_idx, trk in enumerate(self):
                # Ignore if Label does not match
                if trk.label != det_label:
                    cost_matrix[det_idx, trk_idx] = np.nan
                    continue

                # Compute Cost
                cost_matrix[det_idx, trk_idx] = \
                    trk.compute_association_cost(frame=frame, det_bbox=det_bbox)

        # Associate Using Hungarian Algorithm
        matches, unmatched_det_indices, unmatched_trk_indices = \
            associate(cost_matrix=cost_matrix, cost_thresh=cost_thresh)

        # Initialize Kalman Update Flag List
        kalman_update_flags = []

        # Update Associated Trajectories
        for match in matches:
            matched_det_bbox, matched_conf, matched_label = \
                det_bboxes[match[0]], confs[match[0]], labels[match[0]]
            matched_trk = self[match[1]]

            # Update Associated Trajectory
            matched_trk.update(
                lidar_obj=lidar_obj, det_bbox=matched_det_bbox, det_conf=matched_conf,
                fidx=fidx, **kwargs
            )
            kalman_update_flags.append(matched_trk.kalman_flag["update"])

            # Replace Trajectory
            self[match[1]] = matched_trk
            del matched_trk

        # Update Unassociated Trajectories
        for unmatched_trk_idx in unmatched_trk_indices:
            unmatched_trk = self[unmatched_trk_idx]

            # Update Unassociated Trajectory
            unmatched_trk.update(
                lidar_obj=lidar_obj, fidx=fidx, **kwargs
            )
            kalman_update_flags.append(unmatched_trk.kalman_flag["update"])

            # Replace Trajectory
            self[unmatched_trk_idx] = unmatched_trk
            del unmatched_trk

        # If all gathered kalman update flags are True, then Set Kalman Filter Update Flag as True
        if all(kalman_update_flags):
            self.kalman_flag["update"] = True

        # Collect Unassociated Detections (return as Residual Detections)
        res_dets, res_confs, res_labels = \
            np.empty((len(unmatched_det_indices), 4)), np.empty((len(unmatched_det_indices), 1)), np.empty((len(unmatched_det_indices), 1))
        for res_det_idx, unmatched_det_idx in enumerate(unmatched_det_indices):
            res_dets[res_det_idx, :], res_confs[res_det_idx], res_labels[res_det_idx] = \
                dets[unmatched_det_idx], confs[unmatched_det_idx], labels[unmatched_det_idx]

        # Pack Residual Detection as Dictionary and Return
        return {
            "dets": np.array([], dtype=np.float32) if len(unmatched_det_indices) == 0 else res_dets,
            "confs": np.array([], dtype=np.float32) if len(unmatched_det_indices) == 0 else res_confs,
            "labels": np.array([], dtype=np.float32) if len(unmatched_det_indices) == 0 else res_labels,
        }

    def predict(self):
        # Initialize Kalman Predict Flag List
        kalman_predict_flags = []

        # For each trajectory, predict
        for trk in self:
            trk.predict()
            kalman_predict_flags.append(trk.kalman_flag["predict"])

        # If all gathered kalman predict flags are True, then Set Kalman Filter Predict Flag as True
        if all(kalman_predict_flags):
            self.kalman_flag["predict"] = True


class test_A(object):
    def __init__(self):
        self.jkjk = [1,2,3,4,{"hello": "world"}]
        self.pp1 = 123
        self.pp2 = 455
        self.pp3 = 999
        self.pp4 = 911

        # Iteration Counter
        self.__iteration_counter = 0

    def __add__(self, other):
        self.jkjk.append(other)
        return self

    def __iter__(self):
        return self

    def next(self):
        try:
            return_value = self[self.__iteration_counter]
        except IndexError:
            self.__iteration_counter = 0
            raise StopIteration
        self.__iteration_counter += 1
        return return_value

    def __getitem__(self, idx):
        # return self.jkjk[idx]
        _tmp_ = ["pp1", "pp2", "pp3", "pp4"]
        if isinstance(idx, slice):
            slice_start, slice_end = idx.start, idx.stop
            slice_result_list = []
            for j in range(slice_start, slice_end):
                slice_result_list.append(getattr(self, _tmp_[j]))

            return slice_result_list

        elif isinstance(idx, int):
            return getattr(self, _tmp_[idx])

    def __setitem__(self, idx, value):
        self.jkjk[idx] = value


class test_A_sub(test_A):
    def __init__(self):
        super(test_A_sub, self).__init__()

        self.opop = self.jkjk

        # # Iteration Counter
        # self.__iteration_counter = 0

    def __add__(self, other):
        self.opop.append(other)
        return self




if __name__ == "__main__":
    IU = test_A_sub()
    for test_val in IU:
        print(test_val)

    for test_val in IU:
        if isinstance(test_val, dict):
            test_val["hello"] = "KYLE"

    pass
