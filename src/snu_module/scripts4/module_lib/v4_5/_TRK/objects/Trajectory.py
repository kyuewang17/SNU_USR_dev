"""
SNU Integrated Module v5.0
  - Code which defines Trajectory Object Class for Object Tracking

"""
import numpy as np

from module_lib.v4_5._TRK.objects import base, bbox, coordinates
from module_lib.v4_5._TRK.params.kalman_filter import KALMAN_FILTER


class TRAJECTORY(base.object_instance):
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

        # Initialize Kalman State
        self.x3 = init_z.to_state(depth=asso_depths[-1], d_depth=asso_depths[-1]-asso_depths[-2])
        x3p = self.KALMAN_FILTER.predict(self.x3)
        self.x3p = coordinates.STATE_IMAGE_COORD(input_arr=x3p)

        #




























if __name__ == "__main__":
    pass
