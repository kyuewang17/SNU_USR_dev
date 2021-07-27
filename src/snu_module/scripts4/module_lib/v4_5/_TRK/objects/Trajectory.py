"""
SNU Integrated Module v5.0
  - Code which defines Trajectory Object Class for Object Tracking

"""
import numpy as np
from module_lib.v4_5._TRK.objects.base import *


class TRAJECTORY(object_instance):
    def __init__(self, **kwargs):
        super(TRAJECTORY, self).__init__(**kwargs)

        # Initialize Associated Detection List
        self.asso_dets = kwargs.get("asso_dets")
        self.asso_confs = kwargs.get("asso_confs")
        self.asso_flags = kwargs.get("asso_flags")

        # Trajectory


if __name__ == "__main__":
    pass
