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

        # Get Initial Observation
        init_z = coordinates.OBSERVATION_COORD(
            bbox_object=curr_asso_det_bbox, dx=0.0, dy=0.0
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

    def get_fidx_data(self, fidx):
        try:
            idx = self.frame_indices.index(fidx)
        except ValueError:
            return None
        return self[idx]

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



    def predict(self):
        # Kalman Prediction
        x3p = self.KALMAN_FILTER.predict(self.x3.numpify())

        # Transform into State Image Coordinate
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
