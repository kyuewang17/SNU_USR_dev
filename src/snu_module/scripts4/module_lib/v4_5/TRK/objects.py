"""
SNU Integrated Module v5.0
    - Code which defines object classes of Tracking Module

"""
# Import Modules
import math
import numpy as np
import filterpy.kalman.kalman_filter as kalmanfilter

# Import Custom Modules
import utils.bounding_box as snu_bbox
from utils.kcf import KCF_PREDICTOR


# Associated Detection Class
class ASSO_DET(object):
    def __init__(self, det_bbox, det_conf, label, init_fidx):
        # Assertion
        assert det_bbox is not None or det_conf is not None or label is not None

        # Detection Information
        self.bboxes = [det_bbox]
        self.confs = [det_conf]
        self.label = label

        # Frame Information
        self.fidxs = [init_fidx]

    def get_label(self):
        return self.label

    def update(self, det_bbox, det_conf):
        self.bboxes.append(det_bbox)
        self.confs.append(det_conf)
        self.fidxs.append(self.fidxs[-1]+1)

    def get_bbox(self, **kwargs):
        # Get Frame Index Information
        fidx = kwargs.get("fidx")
        assert fidx in self.fidxs
        return self.bboxes[self.fidxs.index(fidx)]

    def is_associated(self, **kwargs):
        # Association Check Frame Index
        fidx = kwargs.get("fidx", self.fidxs[-1])

        # Check
        if self.bboxes[self.fidxs.index(fidx)] is not None:
            return True
        else:
            return False


# Object Base Class
class BASE_OBJECT(object):
    def __init__(self, **kwargs):
        # Initialize Frame Index
        init_fidx = kwargs.get("init_fidx")
        assert init_fidx is not None
        self.fidxs = [init_fidx]

        # Get Object Modal
        self.modal = kwargs.get("modal")

        # Get Object Type
        self.type = kwargs.get("type")

        # Get Object ID
        id = kwargs.get("id")
        assert isinstance(id, int)
        self.id = id

    def __repr__(self):
        return self.type

    def __len__(self):
        return len(self.fidxs)

    def __getitem__(self, idx):
        raise NotImplementedError()

    def update(self, *args, **kwargs):
        raise NotImplementedError()


# Trajectory Candidate Class
class TrajectoryCandidate(BASE_OBJECT):
    def __init__(self, **kwargs):
        super(TrajectoryCandidate, self).__init__(type="TrajectoryCandidate", **kwargs)

        # Initialize Associated Detection Results
        det_bbox, det_conf, det_label = kwargs.get("det_bbox"), kwargs.get("det_conf"), kwargs.get("det_label")
        self.associated_detections = ASSO_DET(
            det_bbox=det_bbox, det_conf=det_conf, label=det_label, init_fidx=kwargs.get("init_fidx")
        )

        # Initialize Observation (z) - bbox type: {u, v, du, dv, w, h}
        self.z = [snu_bbox.bbox_to_zx(det_bbox, np.zeros(2))]



    def associate_detection(self, det_bbox, det_conf):
        self.associated_detections.update(det_bbox=det_bbox, det_conf=det_conf)

    def is_associated(self, **kwargs):
        # Association Check Frame Index
        fidx = kwargs.get("fidx", self.fidxs[-1])
        return self.associated_detections.is_associated(fidx=fidx)

    def get_label(self):
        return self.associated_detections.get_label()



















































if __name__ == "__main__":
    test = TrajectoryCandidate(init_fidx=1, id=1, det_bbox=[1, 1, 10, 10], det_conf=0.777, det_label=33)
    pass
