"""
SNU Integrated Module v5.0
  - Code which defines Trajectory Candidate Object Class for Object Tracking

"""
import cv2
import copy
import math
import numpy as np

from module_lib.v4_5._TRK.objects import object_base, bbox, coordinates
from module_lib.v4_5._TRK.utils import old_kcf


# NOTE: Deprecated!!!!!!!!!!!!!!!!!!!!!!!!
class TRAJECTORY_CANDIDATE(object_base.object_instance):
    def __init__(self, frame, **kwargs):
        super(TRAJECTORY_CANDIDATE, self).__init__(**kwargs)

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
        assert isinstance(det_bbox, bbox.BBOX) and isinstance(is_associated, bool)
        self.det_bboxes, self.det_confs, self.is_associated = [det_bbox], [det_conf], [is_associated]

        # Initialize

        # Initialize BBOX Predictor (KCF Module)
        # TODO: Need to Modify KCF Predictor (modify to handle custom "BBOX" type)
        # self.BBOX_PREDICTOR = kcf.KCF_PREDICTOR(
        #     init_frame=frame,
        # )

        # Set Iteration Counter
        self.__iter_counter = 0

    def __repr__(self):
        return "TRK-CAND ID - [{}]".format(self.id)

    def __add__(self, other):
        assert isinstance(other, (TRAJECTORY_CANDIDATE, list))
        if isinstance(other, TRAJECTORY_CANDIDATE):
            return [self, other]
        else:
            return other.append(self)

    def __getitem__(self, idx):
        return {
            "id": self.id,
            "label": self.label,
            "asso_det_bbox": self.det_bboxes[idx],
            "asso_det_conf": self.det_confs[idx],
            "is_associated": self.is_associated[idx],
        }

    def next(self):
        pass

    def update(self, *args, **kwargs):
        pass







if __name__ == "__main__":
    pass
