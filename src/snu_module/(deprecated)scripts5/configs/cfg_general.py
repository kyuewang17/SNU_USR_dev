"""
Object Recognition Module (SNU Integrated Module)
for Outdoor Surveillance Robots

    - Configuration File for ROS Bag File
    - Environment
        - General Code

"""
# Import Module
import os
import numpy as np

# Import Kalman Filter Parameters
from kalman_params import KALMAN_PARAMS

# Import Configuration Module
from yacs.config import CfgNode as CN

# Get NN Model Base Path
model_base_path = os.path.join(
    (os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models"
)
assert os.path.isdir(model_base_path), "PATH [{}] does not exist...!".format(model_base_path)

# Initialize Kalman Filter Parameters
KPARAMS = KALMAN_PARAMS()

# Config Class Initialization
__C = CN(new_allowed=False)

# Load this Variable when importing this file as a module
cfg = __C

# Agent Environment
__C.agent = CN(new_allowed=True)
__C.agent.type = "NULL"
__C.agent.id = -1

# General ROS Topic Names
__C.ROS = CN(new_allowed=True)
__C.ROS.TOPICS = CN(new_allowed=True)
__C.ROS.TOPICS.tracks = "/osr/tracks"
__C.ROS.TOPICS.odometry = "/robot_odom"

# ----------------------------- #
# ROS Multimodal Sensor Options #
# ----------------------------- #

# ROS Sensor Option Initialization
__C.ROS.SENSORS = CN(new_allowed=False)

# ROS Sensors Initialization
__C.ROS.SENSORS.color = CN(new_allowed=True)
__C.ROS.SENSORS.color.is_valid = True
__C.ROS.SENSORS.color.encoding = "8UC3"
__C.ROS.SENSORS.color.rostopic_name = "/osr/image_color"
__C.ROS.SENSORS.color.camerainfo_rostopic_name = "/osr/image_color_camerainfo"

__C.ROS.SENSORS.depth = CN(new_allowed=True)
__C.ROS.SENSORS.depth.is_valid = True
__C.ROS.SENSORS.depth.encoding = "16UC1"
__C.ROS.SENSORS.depth.rostopic_name = "/osr/image_aligned_depth"
__C.ROS.SENSORS.depth.camerainfo_rostopic_name = "/osr/image_depth_camerainfo"
__C.ROS.SENSORS.depth.CLIP = CN(new_allowed=False)
__C.ROS.SENSORS.depth.CLIP.value = -1
__C.ROS.SENSORS.depth.CLIP.min_distance = 1000
__C.ROS.SENSORS.depth.CLIP.max_distance = 15000

__C.ROS.SENSORS.thermal = CN(new_allowed=True)
__C.ROS.SENSORS.thermal.is_valid = True
__C.ROS.SENSORS.thermal.encoding = "16UC1"
__C.ROS.SENSORS.thermal.rostopic_name = "/osr/image_thermal"
__C.ROS.SENSORS.thermal.camerainfo_rostopic_name = "/osr/image_thermal_camerainfo"

__C.ROS.SENSORS.infrared = CN(new_allowed=True)
__C.ROS.SENSORS.infrared.is_valid = True
__C.ROS.SENSORS.infrared.encoding = "8UC1"
__C.ROS.SENSORS.infrared.rostopic_name = "/osr/image_ir"
__C.ROS.SENSORS.infrared.camerainfo_rostopic_name = "NULL"

__C.ROS.SENSORS.nightvision = CN(new_allowed=True)
__C.ROS.SENSORS.nightvision.is_valid = True
__C.ROS.SENSORS.nightvision.encoding = "8UC3"
__C.ROS.SENSORS.nightvision.rostopic_name = "/osr/image_nv1"
__C.ROS.SENSORS.nightvision.camerainfo_rostopic_name = "NULL"

__C.ROS.SENSORS.lidar = CN(new_allowed=True)
__C.ROS.SENSORS.lidar.is_valid = True
__C.ROS.SENSORS.lidar.encoding = "NULL"
__C.ROS.SENSORS.lidar.rostopic_name = "/osr/lidar_pointcloud"
__C.ROS.SENSORS.lidar.camerainfo_rostopic_name = "NULL"

# -------------- #
# Module Options #
# -------------- #

# Initialize Module Option
__C.MODULE = CN(new_allowed=False)

# Segmentation Options
__C.MODULE.SEG = CN(new_allowed=True)
__C.MODULE.SEG.name = "DeepLabv3"
__C.MODULE.SEG.device = 0
__C.MODULE.SEG.activate = True

__C.MODULE.SEG.SENSORS = CN(new_allowed=False)
__C.MODULE.SEG.SENSORS.color = True
__C.MODULE.SEG.SENSORS.depth = False
__C.MODULE.SEG.SENSORS.thermal = False
__C.MODULE.SEG.SENSORS.infrared = False
__C.MODULE.SEG.SENSORS.nightvision = False
__C.MODULE.SEG.SENSORS.lidar = False

__C.MODULE.SEG.VISUALIZATION = CN(new_allowed=True)
__C.MODULE.SEG.VISUALIZATION.is_draw = True
__C.MODULE.SEG.VISUALIZATION.is_show = True
__C.MODULE.SEG.VISUALIZATION.auto_save = False
__C.MODULE.SEG.VISUALIZATION.is_result_publish = False
__C.MODULE.SEG.VISUALIZATION.result_rostopic_name = "/osr/recognition_seg_result_image"

__C.MODULE.SEG.MISC = CN(new_allowed=True)
__C.MODULE.SEG.MISC.attnet = CN(new_allowed=True)
__C.MODULE.SEG.MISC.attnet.device = 0
__C.MODULE.SEG.MISC.attnet.activate = True

# Detector Options
__C.MODULE.DET = CN(new_allowed=True)
__C.MODULE.DET.name = "YOLOv4"
__C.MODULE.DET.device = 0
__C.MODULE.DET.model_base_path = os.path.join(model_base_path, "detector")

__C.MODULE.DET.SENSORS = CN(new_allowed=False)
__C.MODULE.DET.SENSORS.color = True
__C.MODULE.DET.SENSORS.depth = False
__C.MODULE.DET.SENSORS.thermal = False
__C.MODULE.DET.SENSORS.infrared = False
__C.MODULE.DET.SENSORS.nightvision = False
__C.MODULE.DET.SENSORS.lidar = False

__C.MODULE.DET.VISUALIZATION = CN(new_allowed=True)
__C.MODULE.DET.VISUALIZATION.is_draw = True
__C.MODULE.DET.VISUALIZATION.is_show = True
__C.MODULE.DET.VISUALIZATION.auto_save = False
__C.MODULE.DET.VISUALIZATION.bbox_color = (255, 0, 0)
__C.MODULE.DET.VISUALIZATION.is_result_publish = False
__C.MODULE.DET.VISUALIZATION.result_rostopic_name = "/osr/recognition_det_result_image"

# Tracker Options (TRK and ACL have each own options, but algorithms are unified)
__C.MODULE.TRK = CN(new_allowed=True)
__C.MODULE.TRK.name = "Custom"
__C.MODULE.TRK.device = 0

__C.MODULE.TRK.SENSORS = CN(new_allowed=False)
__C.MODULE.TRK.SENSORS.color = True
__C.MODULE.TRK.SENSORS.depth = True
__C.MODULE.TRK.SENSORS.thermal = False
__C.MODULE.TRK.SENSORS.infrared = True
__C.MODULE.TRK.SENSORS.nightvision = False
__C.MODULE.TRK.SENSORS.lidar = True

# __C.MODULE.TRK.KALMAN_PARAMS =
__C.MODULE.TRK.ASSOCIATION = CN(new_allowed=True)

__C.MODULE.TRK.ASSOCIATION.TRJ_CAND = CN(new_allowed=True)
__C.MODULE.TRK.ASSOCIATION.TRJ_CAND.init_age = 2
__C.MODULE.TRK.ASSOCIATION.TRJ_CAND.destroy_age = 3
__C.MODULE.TRK.ASSOCIATION.TRJ_CAND.SIMILARITY = CN(new_allowed=True)
__C.MODULE.TRK.ASSOCIATION.TRJ_CAND.SIMILARITY.threshold = 0.6
__C.MODULE.TRK.ASSOCIATION.TRJ_CAND.SIMILARITY.weight_iou = 1.0 / 2.0
__C.MODULE.TRK.ASSOCIATION.TRJ_CAND.SIMILARITY.weight_distance = 1.0 / 2.0

__C.MODULE.TRK.ASSOCIATION.TRJ = CN(new_allowed=True)
__C.MODULE.TRK.ASSOCIATION.TRJ.init_age = 3
__C.MODULE.TRK.ASSOCIATION.TRJ.destroy_age = 3
__C.MODULE.TRK.ASSOCIATION.TRJ.SIMILARITY = CN(new_allowed=True)
__C.MODULE.TRK.ASSOCIATION.TRJ.SIMILARITY.threshold = 0.1
__C.MODULE.TRK.ASSOCIATION.TRJ.SIMILARITY.weight_iou = 1.0 / 3.0
__C.MODULE.TRK.ASSOCIATION.TRJ.SIMILARITY.weight_distance = 1.0 / 3.0
__C.MODULE.TRK.ASSOCIATION.TRJ.SIMILARITY.weight_histogram = 1.0 / 3.0

__C.MODULE.TRK.VISUALIZATION = CN(new_allowed=True)
__C.MODULE.TRK.VISUALIZATION.is_draw = True
__C.MODULE.TRK.VISUALIZATION.is_show = True
__C.MODULE.TRK.VISUALIZATION.auto_save = False
__C.MODULE.TRK.VISUALIZATION.bbox_color = (0, 0, 255)
__C.MODULE.TRK.VISUALIZATION.is_result_publish = True
__C.MODULE.TRK.VISUALIZATION.result_rostopic_name = "/osr/snu_trajectory_result_image"

# Action Classification Options
__C.MODULE.ACL = CN(new_allowed=True)
__C.MODULE.ACL.name = "Custom"
__C.MODULE.ACL.device = 0
__C.MODULE.ACL.model_base_path = os.path.join(model_base_path, "aclassifier")

__C.MODULE.ACL.SENSORS = CN(new_allowed=False)
__C.MODULE.ACL.SENSORS.color = True
__C.MODULE.ACL.SENSORS.depth = False
__C.MODULE.ACL.SENSORS.thermal = True
__C.MODULE.ACL.SENSORS.infrared = False
__C.MODULE.ACL.SENSORS.nightvision = False
__C.MODULE.ACL.SENSORS.lidar = False

__C.MODULE.ACL.VISUALIZATION = CN(new_allowed=True)
__C.MODULE.ACL.VISUALIZATION.is_draw = True
__C.MODULE.ACL.VISUALIZATION.is_show = True


if __name__ == "__main__":
    pass
