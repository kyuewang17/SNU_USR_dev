"""
Object Recognition Module (SNU Integrated Module)
for Outdoor Surveillance Robots

    - Configuration File for ROS Bag File
    - Environment
        - Dynamic Agent (Moving Agent)

"""
# Import Module
import os
import numpy as np

# Import General Configuration Module
from cfg_general import __C, KPARAMS, CN

# Load this Variable when importing this file as a module
cfg = __C

# Agent Type Marker
__AGENT_TYPE__ = "dynamic"

# Agent Environment
__C.agent.type = __AGENT_TYPE__

# ROS Sensors Configuration
__C.ROS.SENSORS.color.is_valid = True
__C.ROS.SENSORS.depth.is_valid = True
__C.ROS.SENSORS.thermal.is_valid = True
__C.ROS.SENSORS.infrared.is_valid = True
__C.ROS.SENSORS.nightvision.is_valid = True
__C.ROS.SENSORS.lidar.is_valid = True

# Segmentation Module Configurations
# N/A

# Detection Module Configurations
# __C.MODULE.DET.model_base_path = None

# Tracking Module Configurations
KPARAMS_DICT = KPARAMS(agent_type=__AGENT_TYPE__, return_type="list")

__C.MODULE.TRK.KALMAN_PARAMS = CN(new_allowed=False)
__C.MODULE.TRK.KALMAN_PARAMS.A = KPARAMS_DICT.A
__C.MODULE.TRK.KALMAN_PARAMS.P = KPARAMS_DICT.P
__C.MODULE.TRK.KALMAN_PARAMS.Q = KPARAMS_DICT.Q
__C.MODULE.TRK.KALMAN_PARAMS.R = KPARAMS_DICT.R
__C.MODULE.TRK.KALMAN_PARAMS.K = KPARAMS_DICT.K

__C.MODULE.TRK.ASSOCIATION.TRJ_CAND.init_age = 2
__C.MODULE.TRK.ASSOCIATION.TRJ_CAND.destroy_age = 3
__C.MODULE.TRK.ASSOCIATION.TRJ_CAND.SIMILARITY.threshold = 0.6

__C.MODULE.TRK.ASSOCIATION.TRJ.init_age = 3
__C.MODULE.TRK.ASSOCIATION.TRJ.destroy_age = 3
__C.MODULE.TRK.ASSOCIATION.TRJ.SIMILARITY.threshold = 0.1

# Action Classification Module Configurations
# __C.MODULE.ACL.model_base_path = None


if __name__ == "__main__":
    opop = cfg
    pass
