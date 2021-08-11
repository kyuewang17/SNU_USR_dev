"""
SNU Integrated Module v5.0
  - Code which defines Trajectory Object Class for Object Tracking

"""
import cv2
import copy
import math
import numpy as np

from module_lib.v4_5._TRK.objects import object_base, bbox, coordinates
from module_lib.v4_5._TRK.params.kalman_filter import KALMAN_FILTER
from module_lib.v4_5._TRK.utils.assignment import associate
from module_lib.v4_5._TRK.utils.histogram import histogramize_patch


class TRAJECTORY(object_base.object_instance):
    def __init__(self, **kwargs):
        super(TRAJECTORY, self).__init__(**kwargs)

        # Load Kalman Filter Parameters (tentative)
        self.KALMAN_FILTER = KALMAN_FILTER(**kwargs)

        # Initialize Associated Detection List
        det_bbox, det_conf, is_associated = kwargs.get("det_bbox"), kwargs.get("det_conf"), kwargs.get("is_associated")
        assert isinstance(det_bbox, bbox.BBOX) and isinstance(is_associated, bool)
        self.det_bboxes, self.det_confs, self.is_associated = [det_bbox], [det_conf], [is_associated]

        # Get Trajectory Depth
        depth = kwargs.get("depth", 0.0)
        assert depth >= 0
        self.depths = [depth]

        # Initialize Observation Coordinate
        curr_asso_det_bbox = self.det_bboxes[-1]
        assert isinstance(curr_asso_det_bbox, bbox.BBOX)

        # Get Velocity
        velocity = kwargs.get("velocity", np.array([0.0, 0.0]))
        assert isinstance(velocity, (np.ndarray, list, tuple))
        if isinstance(velocity, (list, tuple)):
            assert len(velocity) == 2
        else:
            assert velocity.size == 2
            if len(velocity.shape) > 1:
                velocity = velocity.reshape(-1)

        # Get Initial Observation
        init_z = coordinates.OBSERVATION_COORD(
            bbox_object=curr_asso_det_bbox, dx=velocity[0], dy=velocity[1]
        )

        # Initialize State and State Prediction
        self.x3 = init_z.to_state(depth=depth, d_depth=0.0)
        x3p = self.KALMAN_FILTER.predict(self.x3.numpify())
        self.x3p = coordinates.STATE_IMAGE_COORD(input_arr=x3p)
        self.states, self.pred_states = [self.x3], [self.x3p]

        # Initialize Camera Coordinate States
        self.c3 = self.x3.to_camera_coord()
        self.cam_states = [self.c3]

        # Initialize Roll, Pitch, Yaw Variables
        self.roll, self.pitch, self.yaw = None, None, None

        # Initialize Action Classification Results
        self.pose = None
        self.pose_list = []

        # Set Iteration Counter
        self.__iter_counter = 0

    def __repr__(self):
        return "TRK ID - [{}]".format(self.id)

    def __add__(self, other):
        assert isinstance(other, (TRAJECTORY, TRAJECTORIES))
        if isinstance(other, TRAJECTORY):
            raise NotImplementedError()
        else:
            return other + self

    def __getitem__(self, idx):
        assert isinstance(idx, int) and 0 <= idx <= len(self) - 1
        return {
            "id": self.id,
            "label": self.label,
            "state": self.states[idx],
            "depth": self.depths[idx],
            "cam_state": self.cam_states[idx],
            "is_associated": self.is_associated[idx]
        }

    def next(self):
        try:
            iter_item = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return iter_item

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

    # NOTE: Experimental Functions
    def init_kalman_params(self):
        raise NotImplementedError()

    def update(self, **kwargs):
        # Get Frame Index and Append
        fidx = kwargs.get("fidx")
        assert isinstance(fidx, int) and fidx > self.frame_indices[-1]
        self.frame_indices.append(fidx)

        # Get Depth
        depth = kwargs.get("depth", self.depths[-1])
        self.depths.append(depth)

        # Get Detection BBOX and Confidence, and get Observation Object
        det_bbox, det_conf = kwargs.get("det_bbox"), kwargs.get("det_conf")
        if det_bbox is None and det_conf is None:
            self.det_bboxes.append(None)
            self.det_confs.append(None)
            self.is_associated.append(False)

            # Think State Prediction as Observation
            z = self.x3p.to_observation_coord()

        elif det_bbox is not None and det_conf is not None:
            assert isinstance(det_bbox, bbox.BBOX)
            self.det_bboxes.append(det_bbox)
            self.det_confs.append(det_conf)
            self.is_associated.append(True)

            # Get Velocity (detection - previous state)
            vel_arr = det_bbox - self.x3.to_bbox()

            # Get Observation Coordinate
            z = coordinates.OBSERVATION_COORD(bbox_object=det_bbox, dx=vel_arr[0], dy=vel_arr[1])

        else:
            raise AssertionError("Input Error")

        # Kalman Update
        x3 = self.KALMAN_FILTER.update(self.x3p.numpify(), z.numpify())

        # Transform into State Image Coordinate
        self.x3 = coordinates.STATE_IMAGE_COORD(input_arr=x3)
        self.states.append(self.x3)

        # Update Camera Coordinate States
        self.c3 = self.x3.to_camera_coord(**kwargs)
        self.cam_states.append(self.c3)

    def predict(self):
        # Kalman Prediction
        x3p = self.KALMAN_FILTER.predict(self.x3.numpify())

        # Transform into State Image Coordinate
        self.x3p = coordinates.STATE_IMAGE_COORD(input_arr=x3p)
        self.pred_states.append(self.x3p)

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

    # def get_intersection_ratio(self, other, method="IOU", **kwargs):
    #     valid_types = (
    #         bbox.BBOX,
    #         coordinates.OBSERVATION_COORD, coordinates.STATE_IMAGE_COORD,
    #         TRAJECTORY
    #     )
    #     assert isinstance(other, valid_types)
    #
    #     # Assertion for Method
    #     assert method in ["IOU", "IOC"]
    #
    #     # Get State Method
    #     state_method = kwargs.get("state_method", "x3p")
    #     assert state_method in ["x3", "x3p"]
    #     kwargs.pop("state_method")
    #
    #     # Get Intersection BBOX
    #     common_bbox = self.get_intersection_bbox(
    #         other=other, state_method=state_method, other_state_method=state_method, **kwargs
    #     )
    #
    #     # Get Intersection Area
    #     common_area = common_bbox.get_size()
    #
    #     # for methods,
    #     if method == "IOU":
    #         if isinstance(other, TRAJECTORY):
    #             union_area = \
    #                 self.get_size(state_method=state_method) + other.get_size(state_method=state_method) - common_area
    #         else:
    #             union_area = \
    #                 self.get_size(state_method=state_method) + other.get_size() - common_area
    #
    #         # Return
    #         if union_area == 0:
    #             return np.nan
    #         else:
    #             return float(common_area) / float(union_area)
    #
    #     elif method == "IOC":
    #         # Get Denominator Component
    #         denom_comp = kwargs.get("denom_comp", "other")
    #         assert denom_comp in ["self", "other"]
    #
    #         # Get Denominator Area
    #         if denom_comp == "other":
    #             if isinstance(other, TRAJECTORY):
    #                 denom_area = other.get_size(state_method=state_method)
    #             else:
    #                 denom_area = other.get_size()
    #         else:
    #             denom_area = self.get_size(state_method=state_method)
    #
    #         # Return
    #         return float(common_area) / float(denom_area)

    def compute_association_cost(self, frame, det_bbox, **kwargs):
        assert isinstance(det_bbox, bbox.BBOX)

        # Get Frame dtype Info
        frame_dtype_info = np.iinfo(frame.dtype)

        # Get Patch Resize Size
        patch_resize_sz = kwargs.get("patch_resize_sz", (64, 64))
        assert isinstance(patch_resize_sz, tuple)
        kwargs.pop("patch_resize_sz")

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
        det_patch, trk_patch = det_bbox.get_patch(frame=frame), x3p.get_patch(frame=frame)
        resized_det_patch = cv2.resize(det_patch, dsize=patch_resize_sz, interpolation=cv2.INTER_NEAREST)
        resized_trk_patch = cv2.resize(trk_patch, dsize=patch_resize_sz, interpolation=cv2.INTER_NEAREST)

        # Get Histograms of Resized Detection and Trajectory Patch
        dhist_bin = 32
        det_hist, det_hist_idx = histogramize_patch(
            resized_det_patch, dhist_bin, (frame_dtype_info.min, frame_dtype_info.max),
            count_window=None, linearize=True
        )
        trk_hist, trk_hist_idx = histogramize_patch(
            resized_trk_patch, dhist_bin, (frame_dtype_info.min, frame_dtype_info.max),
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

        # NOTE: Three Similarities are all implemented
        # NOTE: Now, Let's implement Similarity Merging (weighted-sum) to get total similarity and cost value
        # NOTE: Cost Value = -Similarity


class TRAJECTORIES(object_base.object_instances):
    def __init__(self, **kwargs):
        super(TRAJECTORIES, self).__init__(**kwargs)

        # Set Trajectories
        self.trajectories = self.objects

        # Iteration Counter
        self.__iteration_counter = 0

        # Association Activation Flag
        self.__is_asso_activated = False

    def next(self):
        try:
            return_value = self[self.__iteration_counter]
        except IndexError:
            self.__iteration_counter = 0
            raise StopIteration
        self.__iteration_counter += 1
        return return_value

    def associate(self, **kwargs):
        pass

    def update(self):
        assert self.__is_asso_activated
        raise NotImplementedError()

    def destroy_trajectories(self, **kwargs):
        raise NotImplementedError()

    def get_maximum_id(self):
        if len(self.object_ids) == 0:
            return None
        else:
            return max(self.object_ids)


class test_A(object):
    def __init__(self):
        self.jkjk = [1,2,3,4]

        # Iteration Counter
        self.__iteration_counter = 0

    def __add__(self, other):
        self.jkjk.append(other)
        return self

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        return self.jkjk[idx]


class test_A_sub(test_A):
    def __init__(self):
        super(test_A_sub, self).__init__()

        self.opop = self.jkjk

        # # Iteration Counter
        # self.__iteration_counter = 0

    def __add__(self, other):
        self.opop.append(other)
        return self

    def next(self):
        try:
            return_value = self[self.__iteration_counter]
        except IndexError:
            self.__iteration_counter = 0
            raise StopIteration
        self.__iteration_counter += 1
        return return_value


if __name__ == "__main__":
    tt = np.array([1,2,3])

    IU = test_A_sub()

    type_tuple = (list, tuple, np.ndarray)
    assert isinstance(tt, type_tuple)

    for test_val in IU:
        print(test_val)

    pass
