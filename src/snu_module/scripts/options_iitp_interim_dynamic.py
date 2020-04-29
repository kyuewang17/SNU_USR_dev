"""
SNU Integrated Module v2.05
  - Option File for Unified SNU Framework

"""
# Import Modules
import cv2
import os
from data_struct import STRUCT
import numpy as np

from snu_utils.general_functions import colormap


# Current File Path
curr_file_path = os.path.dirname(__file__)

# Model Base Directory Path
model_base_path = os.path.join(os.path.dirname(curr_file_path), "model")

# Manual Camera Parameter Path for Static Camera
static_cam_param_path = os.path.join(os.path.dirname(curr_file_path), "sensor_params")


# Detector Option Class
class detector_options(object):
    def __init__(self, device=0):
        self.model_dir = os.path.join(model_base_path, "detection_model")
        self.device = device
        self.tiny_area_threshold = 10

        self.detection_args = {
            'n_classes': 3,
            'input_h': 320, 'input_w': 320
        }

        self.backbone_args = {
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
            # 'nms_thresh': 0.5, 'conf_thresh': 0.83,
            'nms_thresh': 0.5, 'conf_thresh': 0.6,
            # 'nms_thresh': 0.5, 'conf_thresh': 0.6,
            # 'pos_anchor_threshold ==>> Objectness(1st threshold)
            'max_boxes': 200, 'pos_anchor_threshold': 0.2,
            # 'max_boxes': 200, 'pos_anchor_threshold': 0.0001,
            'anch_scale': [0.1, 0.2],
            # dynamic edit (191016)!
            'max_w': 1000,
        }


# Tracker Option Class
class tracker_options(object):
    def __init__(self):
        self.association = {
            # Tracklet Candidate to Tracklet Association age (for Tracklet Initialization)
            'trk_init_age': 5,
            # 'trk_init_age': 1,

            # Destroy Unassociated Tracklets with this amount of continuous unassociation
            # 'trk_destroy_age': 4,
            'trk_destroy_age': 15,

            # Destroy Unassociated Tracklet Candidates with this amount of continuous unassociation
            'trkc_destroy_age': 10,

            # Association Cost Threshold
            # [1] DETECTION to TRACKLET
            'cost_thresh_d2trk': 0.0,
            # [2] DETECTION to TRACKLET CANDIDATE (d2d)
            'cost_thresh_d2trkc': 0.2
        }

        self.depth_params = {
            # histogram bin number
            'hist_bin': 100,

            # histogram count window gaussian weight map parameters
            'hist_gaussian_mean': 0,
            'hist_gaussian_stdev': 0.1,

            # [TEST]
            'spatch_len_factor': 0.5,
            'lpatch_len_factor': 0.8,

            # Moving Average for Depth Filtering
            'depth_update_weights': [1, 4, 9, 16, 25],

            # Destroy too far away Tracklets (in meters)
            'max_depth_threshold': 15,
        }

        # self.tracklet_colors = np.random.rand(32, 3)
        self.trk_color_refresh_period = 16
        self.tracklet_colors = np.array(colormap(self.trk_color_refresh_period))


# Action Classifier Option Class
class aclassifier_options(object):
    def __init__(self, device=0):
        self.model_dir = os.path.join(model_base_path, "aclassifier_model/model_test_RoboWorld2.pt")
        self.device = device

        self.params = {}


# Visualizer Option Class
class visualizer_options(object):
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.font_size = 1.2

        self.pad_pixels, self.info_interval = 2, 4

        self.detection = {
            "is_draw": True,

            "is_show_label": None,
            "is_show_score": None,
            "is_show_fps": None,

            # <R, G, B> in our setting
            "bbox_color": (255, 0, 0),

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
            "rostopic_name": "/osr/image_color",
            "camerainfo_rostopic_name": "/osr/image_color_camerainfo",
        }
        self.depth = {
            "imgmsg_to_cv2_encoding": "16UC1",
            "rostopic_name": "/osr/image_aligned_depth",
            "camerainfo_rostopic_name": "/osr/__image_depth_camerainfo",

            # Depth Image Clip Value
            "clip_value": -1,

            # Depth Clip Distance (in "millimeters")
            "clip_distance": {
                "min": 1000,
                "max": 30000,
            },
        }
        self.lidar1 = {
            "imgmsg_to_cv2_encoding": "8UC3",
            "rostopic_name": "/osr/image_lidar",

            # LIDAR Image Clip Value
            "clip_value": -2,

            # LIDAR Scaling Factor
            "scaling_factor": float(50) / float(255),
        }
        self.lidar2 = {
            "imgmsg_to_cv2_encoding": "8UC3",
            "rostopic_name": "/__camera_lidar2",

            # LIDAR Image Clip Value
            "clip_value": -2,

            # LIDAR Scaling Factor
            "scaling_factor": float(50) / float(255),
        }
        self.lidar_pc = {
            "msg_encoding": None,
            "rostopic_name": "/osr/__lidar_pointcloud"
        }
        self.infrared = {
            "imgmsg_to_cv2_encoding": "8UC1",
            "rostopic_name": "/osr/image_ir",
            "camerainfo_rostopic_name": "/osr/__image_ir_camerainfo",
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
        self.sensor_frame_rate = 10

        # Rostopic Message for Publisher
        self.publish_mesg = {
            "tracks": "/osr/tracks",
            "result_image": "/osr/snu_result_image_1"
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





















