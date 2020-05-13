"""
SNU Integrated Module v3.0
    - Option Class
        - Module Options
            - Object Detection
            - Multiple Target Tracking
            - Action Classification
        - Sensor Options
            - [color] D435i RGB Camera
            - [disparity] D435i Depth Camera
            - [thermal] Thermal Camera (TODO: Find Camera Model)
            - [infrared] Infrared Camera (TODO: Find Camera Model)
            - [Nightvision] NightVision Camera (TODO: Find Camera Model)
            - [LIDAR] Velodyne LiDAR Sensor (TODO: Check Sensor Model and Type)



"""

# Import Modules
import cv2
import os
import numpy as np

# Import Colormap
from snu_utils.general_functions import colormap

# Current File Path
curr_file_path = os.path.dirname(__file__)

# Model Base Directory Path
model_base_path = os.path.join(os.path.dirname(curr_file_path), "model")

# Manual Multimodal Sensor Parameter Path
# (might or might not be deleted later on....)
static_cam_param_path = os.path.join(os.path.dirname(curr_file_path), "sensor_params")


# Define Option Class
class snu_option_class(object):
    def __init__(self, agent_type=None, agent_name=None):
        # Agent Type
        if agent_type is None:
            print("[WARNING] Agent Type is NOT DEFINED: setting it to 'ambiguous type'!")
            self.agent_type = "ambiguous"
            self.modal_switch_dict = {
                "color": True,
                "disparity": True,
                "thermal": False,
                "infrared": False,
                "nightvision": False,
                "lidar": False,
            }
        elif agent_type in ["static", "dynamic"]:
            self.agent_type = agent_type
            self.modal_switch_dict = {
                "color": True,
                "disparity": True,
                "thermal": (True if agent_type == "dynamic" else False),
                "infrared": True,
                "nightvision": True,
                "lidar": True,
            }
        elif agent_type == "rosbagfile":
            self.agent_type = agent_type
            self.modal_switch_dict = {
                "color": True,
                "disparity": True,
                "thermal": False,
                "infrared": False,
                "nightvision": False,
                "lidar": True,
            }
        else:
            assert 0, "UNDEFINED AGENT TYPE!"

        # Agent Name
        self.agent_name = agent_name

        # ROS Node Sleep Time for Sensor Synchronization
        self.node_sleep_time_for_sensor_sync = 0.05

        # Paths (e.g. models, parameters, etc.)
        self.paths = {
            "curr_file_path": curr_file_path,
            "model_base_path": model_base_path,
            "static_cam_param_path": os.path.join(static_cam_param_path, "CamParam.yml"),
        }

        # Detector Options
        self.detector = detector_options(modal_switch_dict=self.modal_switch_dict, device=0)

        # Tracker Options
        self.tracker = tracker_options()

        # Action Classifier Options
        self.aclassifier = aclassifier_options(modal_switch_dict=self.modal_switch_dict, device=0)

        # Visualizer Options
        self.visualization = visualizer_options(is_draw_detection=True,
                                                is_draw_tracking=True,
                                                is_draw_aclassification=False)

        # Sensor Options
        self.sensors = sensor_options()
        self.sensor_frame_rate = 10

        # Rostopic Message for Publisher
        self.publish_mesg = {
            "tracks": "/osr/tracks",
            "result_image": "/osr/snu_result_image",

            "detection_result": "/osr/snu_detection_result_image",
            "tracking_result": "/osr/snu_tracking_result_image",
            "aclassification_result": "/osr/snu_aclassification_result_image",
        }


# Sensor Option Class (ROS message)
class sensor_options(object):
    def __init__(self):
        # D435i RGB Camera
        self.color = {
            # ROS Message
            "imgmsg_to_cv2_encoding": "8UC3",
            "rostopic_name": "/osr/image_color",
            "camerainfo_rostopic_name": "/camera/color/camera_info",

            # Calibrated to Camera
            "calib_obj_cam": "color",
        }

        # D435i Depth Camera
        self.disparity = {
            # ROS Message
            "imgmsg_to_cv2_encoding": "16UC1",
            # "rostopic_name": "/osr/image_aligned_depth",
            "rostopic_name": "/osr/image_depth",
            "camerainfo_rostopic_name": "/camera/depth/camera_info",

            # Calibrated to Camera
            "calib_obj_cam": "color",

            # Disparity Image Clip Value
            "clip_value": -1,

            # Disparity Image Clip Distance (in "millimeters")
            "clip_distance": {
                "min": 1000,
                "max": 15000,
            }
        }

        # Thermal Camera
        self.thermal = {
            # ROS Message
            "imgmsg_to_cv2_encoding": "16UC1",
            "rostopic_name": "/osr/image_thermal",
            "camerainfo_rostopic_name": None,

            # Calibrated to Camera
            "calib_obj_cam": "color",
        }

        # Infrared Camera
        self.infrared = {
            # ROS Message
            "imgmsg_to_cv2_encoding": "8UC1",
            "rostopic_name": "/osr/image_ir",
            "camerainfo_rostopic_name": None,

            # Calibrated to Camera
            "calib_obj_cam": "color",
        }

        # NightVision Camera
        self.nightvision = {
            # ROS Message
            "imgmsg_to_cv2_encoding": "8UC3",
            "rostopic_name": "/osr/image_nv1",
            "camerainfo_rostopic_name": None,

            # Calibrated to Camera
            "calib_obj_cam": "color",
        }

        # LIDAR Point-cloud
        self.lidar = {
            # ROS Message
            # "rostopic_name": "/velodyne_points",
            "rostopic_name": "/osr/lidar_pointcloud",

            # Calibrated to Camera
            "calib_obj_cam": "color",
        }

        # Odometry Message (Pass-through SNU module to ETRI module)
        self.odometry = {
            "rostopic_name": "/robot_odom"
        }


# Detector Option Class
class detector_options(object):
    def __init__(self, modal_switch_dict, device=0):
        # Assertion
        assert (type(modal_switch_dict) == dict), "Argument 'modal_switch_dict' must be a <dict> type!"

        # Load Detection Model Path regarding the Trained Modalities
        if modal_switch_dict["thermal"] is True:
            # Get Model path of RGB-T model for now...(on night)
            self.model_dir = os.path.join(model_base_path, "detection_model_night_thermal")

            # Set Actually Using Sensor Modalities
            self.sensor_dict = {
                "color": True,
                "disparity": False,
                "thermal": True,
                "infrared": False,
                "nightvision": False,
                "lidar": False,
            }
        else:
            # Get Model path of RGB model for now...(on day)
            self.model_dir = os.path.join(model_base_path, "detection_model_day_rgb")

            # Set Actually Using Sensor Modalities
            self.sensor_dict = {
                "color": True,
                "disparity": True,
                "thermal": False,
                "infrared": False,
                "nightvision": False,
                "lidar": False,
            }

        # GPU-device
        self.device = device

        # Tiny Area Threshold
        self.tiny_area_threshold = 10

        # Detection Arguments
        self.detection_args = {
            "n_classes": 3,
            "input_h": 320, "input_w": 448,
        }

        # Backbone Arguments
        self.backbone_args = {
            "name": "res34level4",
            "pretrained": False,
        }

        # Detector Arguments
        # ( what is the difference between 'detection_args' ?!?! )
        self.detector_args = {
            "is_bnorm": False, "tcb_ch": 256,
            'fmap_chs': [128, 256, 512, 128],
            'fmap_sizes': [40, 20, 10, 5], 'fmap_steps': [8, 16, 32, 64],
            'anch_scale': [0.1, 0.2], 'anch_min_sizes': [32, 64, 128, 256],
            'anch_max_sizes': [], 'anch_aspect_ratios': [[2], [2], [2], [2]],
            'n_boxes_list': [3, 3, 3, 3], 'is_anch_clip': True,
        }

        # Post-processing Arguments
        self.postproc_args = {
            'n_infer_rois': 300, 'device': 0, 'only_infer': True,
            # 'conf_thresh' ==>> Classification(2nd threshold)
            # 'nms_thresh': 0.45, 'conf_thresh': 0.3,
            # 'nms_thresh': 0.5, 'conf_thresh': 0.83,
            'nms_thresh': 0.5, 'conf_thresh': 0.4,
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
        # Set Actually Using Sensor Modalities
        self.sensor_dict = {
            "color": True,
            "disparity": True,
            "thermal": False,
            "infrared": False,
            "nightvision": False,
            "lidar": True,
        }

        # Association-related Options
        self.association = {
            # Tracklet Candidate to Tracklet Association age (for Tracklet Initialization)
            # bug when set to even number
            'trk_init_age': 4,
            # 'trk_init_age': 1,

            # Destroy Unassociated Tracklets with this amount of continuous unassociation
            # 'trk_destroy_age': 4,
            'trk_destroy_age': 7,

            # Destroy Unassociated Tracklet Candidates with this amount of continuous unassociation
            'trkc_destroy_age': 5,

            # Association Cost Threshold
            # [1] DETECTION to TRACKLET
            'cost_thresh_d2trk': 0.01,
            # 'cost_thresh_d2trk': 0.,

            # [2] DETECTION to TRACKLET CANDIDATE (d2d)
            # 'cost_thresh_d2trkc': 0.35,
            'cost_thresh_d2trkc': 0.
        }

        # Disparity Modality Parameters
        self.disparity_params = {
            # Extraction Rate for Disparity Patch
            "extraction_roi_rate": 0.65,

            # Histogram Bin for Rough Depth Computation
            "rough_hist_bin": 25,

            # Histogram Bin Number
            "hist_bin": 100,

            # Histogram Count Window Gaussian Weight Map Parameters
            "hist_gaussian_mean": 0,
            "hist_gaussian_stdev": 0.1,
        }

        # Tracklet Color Options
        self.trk_color_refresh_period = 16
        self.tracklet_colors = np.array(colormap(self.trk_color_refresh_period))


# Action Classifier Option Class
class aclassifier_options(object):
    def __init__(self, modal_switch_dict, device=0):
        # Assertion
        assert (type(modal_switch_dict) == dict), "Argument 'modal_switch_dict' must be a <dict> type!"

        # Load Detection Model Path regarding the Trained Modalities
        if modal_switch_dict["thermal"] is True:
            # Get Model path of RGB-T model for now...(on night)
            self.model_dir = os.path.join(model_base_path, "aclassifier_model/model_thermal_1ch_input.pt")

            # Set Actually Using Sensor Modalities
            self.sensor_dict = {
                "color": True,
                "disparity": False,
                "thermal": True,
                "infrared": False,
                "nightvision": False,
                "lidar": False,
            }
        else:
            # Get Model path of RGB model for now...(on day)
            self.model_dir = os.path.join(model_base_path, "aclassifier_model/model_test_RoboWorld2.pt")

            # Set Actually Using Sensor Modalities
            self.sensor_dict = {
                "color": True,
                "disparity": False,
                "thermal": False,
                "infrared": False,
                "nightvision": False,
                "lidar": False,
            }

        # GPU-device for Action Classification
        self.device = device

        # Miscellaneous Parameters (for future usage)
        self.params = {}


# Visualizer Option Class
class visualizer_options(object):
    def __init__(self, is_draw_detection, is_draw_tracking, is_draw_aclassification):
        assert ((type(is_draw_detection) == bool) and
                (type(is_draw_tracking) == bool) and
                (type(is_draw_aclassification) == bool)), "Arguments must be Boolean (True/False)!"

        self.font = cv2.FONT_HERSHEY_PLAIN
        self.font_size = 1.2

        self.pad_pixels = 2
        self.info_interval = 4

        self.detection = {
            "is_draw": is_draw_detection,

            "is_show_label": None,
            "is_show_score": None,
            "is_show_fps": None,

            # (RGB) in our setting
            "bbox_color": (255, 0, 0),

            # Line-width
            "linewidth": 2,
        }

        self.tracking = {
            "is_draw": is_draw_tracking,

            "is_show_id": None,
            "is_show_3d_coord": None,
            "is_show_depth": None,
            "is_show_fps": None,

            "bbox_color": (0, 0, 255),

            "linewidth": 2,
        }

        self.aclassifier = {
            "is_draw": is_draw_aclassification,
        }
