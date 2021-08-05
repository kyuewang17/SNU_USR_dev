"""
SNU Integrated Module v5.0
  - Code which defines Trajectory Object Class for Object Tracking

"""
import numpy as np

from module_lib.v4_5._TRK.objects import object_base, bbox, coordinates
from module_lib.v4_5._TRK.params.kalman_filter import KALMAN_FILTER


class TRAJECTORY(object_base.object_instance):
    def __init__(self, **kwargs):
        super(TRAJECTORY, self).__init__(**kwargs)

        # Load Kalman Filter Parameters (tentative)
        self.KALMAN_FILTER = KALMAN_FILTER()

        # Initialize Associated Detection List
        self.det_bboxes = kwargs.get("det_bboxes")
        self.det_confs = kwargs.get("det_confs")
        self.is_associated = kwargs.get("is_associated")

        # Get Trajectory Depth
        asso_depths = kwargs.get("asso_depths")
        assert isinstance(asso_depths, list) and len(asso_depths) > 0
        if len(asso_depths) == 1:
            asso_depths.insert(0, 0.0)
        self.depths = asso_depths
        self.depth = asso_depths[-1]

        # Initialize Observation Coordinate
        curr_asso_det_bbox = self.det_bboxes[-1]
        assert isinstance(curr_asso_det_bbox, bbox.BBOX)
        if len(self.det_bboxes) > 1:
            curr_asso_vel = curr_asso_det_bbox - self.det_bboxes[-2]
        else:
            curr_asso_vel = np.array([0.0, 0.0])

        # Get Initial Observation
        init_z = coordinates.OBSERVATION_COORD(
            bbox_object=curr_asso_det_bbox, dx=curr_asso_vel[0], dy=curr_asso_vel[1]
        )

        # Initialize State and State Prediction
        self.x3 = init_z.to_state(depth=asso_depths[-1], d_depth=asso_depths[-1] - asso_depths[-2])
        x3p = self.KALMAN_FILTER.predict(self.x3)
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

    def get_fidx_data(self, fidx):
        try:
            idx = self.frame_indices.index(fidx)
        except ValueError:
            return None
        return self[idx]

    def update(self, *args, **kwargs):
        pass

    def predict(self):
        x3p = self.KALMAN_FILTER.predict(self.x3)
        self.x3p = coordinates.STATE_IMAGE_COORD(input_arr=x3p)


class TRAJECTORIES(object_base.object_instances):
    def __init__(self, **kwargs):
        super(TRAJECTORIES, self).__init__(**kwargs)

        # Set Trajectories
        self.trajectories = self.objects

    def associate(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    tt = np.array([1,2,3])
    pass
