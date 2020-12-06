#!/usr/bin/env python
"""
- Object Class Python Script for Camera Environment (especially Camera Parameters)

"""
import cv2
import numpy as np


class BASE_CAMERA_PARAMS_OBJ(object):
    def __init__(self, param_precision):
        # Parameter Precision
        self.param_precision = param_precision

        # Initialize Projection Matrix and its Pseudo-inverse
        self.P = None
        self.Pinv = None

    def update_params(self, param_argument):
        raise NotImplementedError()


class ROS_CAMERA_PARAMS_OBJ(BASE_CAMERA_PARAMS_OBJ):
    def __init__(self, param_precision=np.float32):
        super(ROS_CAMERA_PARAMS_OBJ, self).__init__(param_precision)

        """ Initialize Camera Parameter Matrices
        ----------------------------------------
        D: Distortion Matrix (5x1)
        K: Intrinsic Matrix (3x3)
        R: Rotation Matrix (3x3)
        P: Projection Matrix (3x4)
        ----------------------------------------
        """
        self.D, self.K, self.R, self.P = None, None, None, None

    def update_params(self, msg):
        self.D = np.asarray(msg.D).reshape((5, 1))  # Distortion Matrix
        self.K = np.asarray(msg.K).reshape((3, 3))  # Intrinsic Matrix
        self.R = np.asarray(msg.R).reshape((3, 3))  # Rotation Matrix
        self.P = np.asarray(msg.P).reshape((3, 4))  # Projection Matrix

        self.Pinv = np.linalg.pinv(self.P)


class FILE_CAMERA_PARAMS_OBJ(BASE_CAMERA_PARAMS_OBJ):
    """ Used for Static Camera (YAML-based Camera Parameter) """
    def __init__(self, param_precision=np.float32):
        super(FILE_CAMERA_PARAMS_OBJ, self).__init__(param_precision)
















if __name__ == "__main__":
    pass
