"""
SNU Integrated Module v2.05
  - Option File for Unified SNU Framework

"""
# Import Modules
import cv2
import os
from data_struct import STRUCT
import numpy as np


# Current File Path
curr_file_path = os.path.dirname(__file__)

# Model Base Directory Path
model_base_path = os.path.join(os.path.dirname(curr_file_path), "model")

# Manual Camera Parameter Path for Static Camera
static_cam_param_path = os.path.join(os.path.dirname(curr_file_path), "sensor_params")


# Detector Option Class
class detector_options(object):
    def __init__(self, device=0):
        # day-rgb: thermal=False, detection_model_day_rgb
        # night-thermal: thermal=True, detection_model_night_thermal
        self.thermal = True
        self.model_dir = os.path.join(model_base_path, "detection_model_day_rgb")
        self.device = device
        self.tiny_area_threshold = 10

        self.detection_args = {
            'n_classes': 3,
            'input_h': 320, 'input_w': 448
        }

        self.backbone_args = {
            'name': 'res34level4',
            'pretrained': False
        }

        self.detector_args = {
            'is_bnorm': False, 'tcb_ch': 256,
            'fmap_chs': [128, 256, 512, 128],
            'fmap_sizes': [40, 20, 10, 5], 'fmap_steps': [8, 16, 32, 64],
            'anch_scale': [0.1, 0.2], 'anch_min_sizes': [32, 64, 128, 256],
            'anch_max_sizes': [], 'anch_aspect_ratios': [[2], [2], [2], [2]],
            'n_boxes_list': [3, 3, 3, 3], 'is_anch_clip': True
        }

        self.postproc_args = {
            'n_infer_rois': 300, 'device': 1, 'only_infer': True,
            # 'conf_thresh' ==>> Classification(2nd threshold)
            # 'nms_thresh': 0.45, 'conf_thresh': 0.3,
            'nms_thresh': 0.5, 'conf_thresh': 0.2,
            # 'pos_anchor_threshold ==>> Objectness(1st threshold)
            'max_boxes': 200, 'pos_anchor_threshold': 0.2,
            # 'max_boxes': 200, 'pos_anchor_threshold': 0.0001,
            'anch_scale': [0.1, 0.2]
        }


# Tracker Option Class
class tracker_options(object):
    def __init__(self):
        self.association = {
            # Tracklet Candidate to Tracklet Association age (for Tracklet Initialization)
            'trk_init_age': 3,
            # 'trk_init_age': 1,

            # Destroy Unassociated Tracklets with this amount of continuous unassociation
            # 'trk_destroy_age': 4,
            'trk_destroy_age': 4,

            # Destroy Unassociated Tracklet Candidates with this amount of continuous unassociation
            'trkc_destroy_age': 10,

            # Association Cost Threshold
            # [1] DETECTION to TRACKLET
            'cost_thresh_d2trk': 0.3,
            # [2] DETECTION to TRACKLET CANDIDATE
            'cost_thresh_d2trkc': 0.6
        }

        self.depth_params = {
            # histogram bin number
            'hist_bin': 500,

            # histogram count window gaussian weight map parameters
            'hist_gaussian_mean': 0,
            'hist_gaussian_stdev': 0.6,
        }

        self.tracklet_colors = np.random.rand(32, 3)


# Action Classifier Option Class
class aclassifier_options(object):
    def __init__(self, device=0):
        self.model_dir = os.path.join(model_base_path, "aclassifier_model/model_test.pt")
        self.device = device

        self.params = {}


# Visualizer Option Class
class visualizer_options(object):
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.font_size = 1.2

        self.pad_pixels, self.info_interval = 2, 4

        self.detection = {
            "is_draw": None,

            "is_show_label": None,
            "is_show_score": None,
            "is_show_fps": None,

            "bbox_color": (0, 0, 255),

            "linewidth": 2,
        }

        self.tracking = {
            "is_draw": True,

            "is_show_id": None,
            "is_show_3d_coord": None,
            "is_show_depth": None,
            "is_show_fps": None,

            "bbox_color": (0, 0, 255),

            "linewidth": 2,
        }

        self.aclassifier = {
            "is_draw": True,
        }


# Sensor Option Class
class sensor_options(object):
    def __init__(self):
        self.rgb = {
            "imgmsg_to_cv2_encoding": "8UC3",
            "rostopic_name": "/camera/color/image_raw",
            "camerainfo_rostopic_name": "/camera/color/camera_info"
        }
        self.depth = {
            "imgmsg_to_cv2_encoding": "16UC1",
            "rostopic_name": "/camera/aligned_depth_to_color/image_raw",
            "camerainfo_rostopic_name": "/camera/aligned_depth_to_color/camera_info",

            # Depth Image Clip Value
            "clip_value": -1,

            # Depth Clip Distance (in "millimeters")
            "clip_distance": {
                "min": 1000,
                "max": 20000
            },
        }
        self.lidar1 = {
            "imgmsg_to_cv2_encoding": "8UC3",
            "rostopic_name": "/camera_lidar",

            # LIDAR Image Clip Value
            "clip_value": -2,

            # LIDAR Scaling Factor
            "scaling_factor": float(50) / float(255),
        }
        self.lidar2 = {
            "imgmsg_to_cv2_encoding": "8UC3",
            "rostopic_name": "/camera_lidar2",

            # LIDAR Image Clip Value
            "clip_value": -2,

            # LIDAR Scaling Factor
            "scaling_factor": float(50) / float(255),
        }
        self.lidar_pc = {
            "msg_encoding": None,
            "rostopic_name": "/osr/lidar_pointcloud"
        }
        self.infrared = {
            "imgmsg_to_cv2_encoding": "8UC1",
            "rostopic_name": "/osr/image_ir",
        }
        self.thermal = {
            "imgmsg_to_cv2_encoding": "8UC1",
            "rostopic_name": "/osr/image_thermal",
        }
        self.nightvision = {
            "imgmsg_to_cv2_encoding": "8UC3",
            "rostopic_name": "/osr/image_nv1",
        }
        self.odometry = {
            "rostopic_name": "/robot_odom"
        }


# Define Option Class
class option_class(object):
    def __init__(self, agent_type=None, agent_name=None):
        # Agent Type
        if agent_type is None:
            print("[WARNING] Agent Type is NOT DEFINED: setting it to 'ambiguous type'!")
            self.agent_type = "ambiguous"
        elif agent_type in ["static", "dynamic"]:
            self.agent_type = agent_type
        else:
            assert 0, "UNDEFINED AGENT TYPE!"

        # Agent Name
        self.agent_name = agent_name

        # Paths (e.g. models, parameters, etc.)
        self.paths = {
            'curr_file_path': curr_file_path,
            'model_base_path': model_base_path,
            'static_cam_param_path': os.path.join(static_cam_param_path, "CamParam.yml"),
        }

        # Detector Options
        self.detector = detector_options()

        # Tracker Options
        self.tracker = tracker_options()

        # Action Classifier Options
        self.aclassifier = aclassifier_options()

        # Visualizer Options
        self.visualization = visualizer_options()

        # Sensor Options
        self.sensors = sensor_options()

        # Rostopic Message for Publisher
        self.publish_mesg = {
            "tracks": "/osr/tracks",
            "result_image": "/osr/snu_result_image"
        }


# # Option Struct Initialization
# opts = STRUCT()
# opts.agent_type = "dynamic"
# opts.detector, opts.tracker, opts.aclassifier = STRUCT(), STRUCT(), STRUCT()
# opts.sensors = STRUCT()
# opts.visualization = STRUCT()
#
# ######## Detector Options ########
# opts.detector.





















