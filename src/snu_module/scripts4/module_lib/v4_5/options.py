"""
SNU Integrated Module v4.5
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
import copy
import numpy as np

# Import Colormap, Screen Geometry
from utils.general_functions import colormap, get_screen_geometry

# Current File Path
curr_file_path = os.path.dirname(__file__)


# Kalman Filter Parameters
class kalman_params(object):
    def __init__(self):
        # State Transition Matrix
        self.A = None

        # Unit Transformation Matrix
        self.H = None

        # Error Covariance Matrix
        self.P = None

        # State Covariance Matrix
        self.Q = None

        # Measurement Covariance Matrix
        self.R = None

        # Kalman Gain (for initialization)
        self.K = None

    def get_params_from_cfg(self, kparams_cfg):
        self.A = np.asarray(kparams_cfg.A).astype(np.float32)
        self.H = np.asarray(kparams_cfg.H).astype(np.float32)
        self.P = np.asarray(kparams_cfg.P).astype(np.float32)
        self.Q = np.asarray(kparams_cfg.Q).astype(np.float32)
        self.R = np.asarray(kparams_cfg.R).astype(np.float32)
        self.K = np.asarray(kparams_cfg.K).astype(np.float32)


# Define Option Class
class snu_option_class(object):
    def __init__(self, cfg, dev_version=4.5):
        # Image Environment Type
        env_type = cfg.env.type
        self.env_type = env_type

        # Agent Type
        if env_type == "agent":
            self.agent_type, self.agent_id = env_type, cfg.env.id if hasattr(cfg.env, "id") else None
        else:
            self.agent_type, self.agent_id = None, None

        # Development Version
        self.dev_version = dev_version

        # Modal Switch Dictionary
        self.modal_switch_dict = {
            "color": cfg.sensors.color.is_valid,
            "disparity": cfg.sensors.disparity.is_valid,
            "thermal": cfg.sensors.thermal.is_valid,
            "infrared": cfg.sensors.infrared.is_valid,
            "nightvision": cfg.sensors.nightvision.is_valid,
            "lidar": cfg.sensors.lidar.is_valid,
        }

        # Machine Name
        self.machine_name = cfg.machine_name

        # Screen Compensation
        if cfg.screen_compensate is True:
            screen_geometry_dict = get_screen_geometry()
            self.screen_imshow_x = int(screen_geometry_dict["margin_length"]["left_pixels"] * 1.1)
            self.screen_imshow_y = int(screen_geometry_dict["margin_length"]["top_pixels"] * 1.1)
        else:
            self.screen_imshow_x, self.screen_imshow_y = None, None

        # Detector Options
        self.detector = detector_options(cfg=cfg)

        # Tracker Options
        self.tracker = tracker_options(cfg=cfg)

        # Action Classifier Options
        self.aclassifier = aclassifier_options(cfg=cfg)

        # Visualizer Options
        self.visualization = visualizer_options(cfg=cfg)

        # Sensor Options
        self.sensors = sensor_options(cfg=cfg)
        self.sensor_frame_rate = 10

        # Rostopic Message for Publisher
        self.publish_mesg = {
            "tracks": cfg.publisher.tracks,

            "det_result_rostopic_name": cfg.detector.visualization.result_rostopic_name,
            "trk_acl_result_rostopic_name": cfg.tracker.visualization.result_rostopic_name,

            "trk_top_view_rostopic_name": cfg.tracker.visualization.top_view.result_rostopic_name,
        }

    def __repr__(self):
        return "v{}-[{}]".format(self.dev_version, self.env_type)


# Sensor Option Class (ROS message)
class sensor_options(object):
    def __init__(self, cfg):
        # Initialize Sensor Image Width and Height
        self.width, self.height = None, None

        # Sensor Frequency (for Image Sequence)
        self.frequency = cfg.sensors.frequency if hasattr(cfg.sensors, "frequency") else None

        # D435i RGB Camera
        self.color = {
            # Valid Flag
            "is_valid": cfg.sensors.color.is_valid,

            # ROS Message
            "imgmsg_to_cv2_encoding": cfg.sensors.color.encoding,
            "rostopic_name": cfg.sensors.color.rostopic_name,
            "camerainfo_rostopic_name": cfg.sensors.color.camerainfo_rostopic_name,
        }

        # D435i Depth Camera
        self.disparity = {
            # Valid Flag
            "is_valid": cfg.sensors.disparity.is_valid,

            # ROS Message
            "imgmsg_to_cv2_encoding": cfg.sensors.disparity.encoding,
            "rostopic_name": cfg.sensors.disparity.rostopic_name,
            "camerainfo_rostopic_name": cfg.sensors.disparity.camerainfo_rostopic_name,

            # Disparity Image Clip Value
            "clip_value": cfg.sensors.disparity.clip.value,

            # Disparity Image Clip Distance (in "millimeters")
            "clip_distance": {
                "min": cfg.sensors.disparity.clip.min_distance,
                "max": cfg.sensors.disparity.clip.max_distance,
            }
        }

        # Thermal Camera
        self.thermal = {
            # Valid Flag
            "is_valid": cfg.sensors.thermal.is_valid,

            # ROS Message
            "imgmsg_to_cv2_encoding": cfg.sensors.thermal.encoding,
            "rostopic_name": cfg.sensors.thermal.rostopic_name,
            "camerainfo_rostopic_name": cfg.sensors.thermal.camerainfo_rostopic_name,
        }

        # Infrared Camera
        self.infrared = {
            # Valid Flag
            "is_valid": cfg.sensors.infrared.is_valid,

            # ROS Message
            "imgmsg_to_cv2_encoding": cfg.sensors.infrared.encoding,
            "rostopic_name": cfg.sensors.infrared.rostopic_name,
            "camerainfo_rostopic_name": cfg.sensors.infrared.camerainfo_rostopic_name,
        }

        # NightVision Camera
        self.nightvision = {
            # Valid Flag
            "is_valid": cfg.sensors.nightvision.is_valid,

            # ROS Message
            "imgmsg_to_cv2_encoding": cfg.sensors.nightvision.encoding,
            "rostopic_name": cfg.sensors.nightvision.rostopic_name,
            "camerainfo_rostopic_name": cfg.sensors.nightvision.camerainfo_rostopic_name,
        }

        # LIDAR Point-cloud Image
        self.lidar = {
            # Valid Flag
            "is_valid": cfg.sensors.lidar.is_valid,

            # ROS Message
            "imgmsg_to_cv2_encoding": cfg.sensors.lidar.encoding,
            "rostopic_name": cfg.sensors.lidar.rostopic_name,
        }

        # Odometry Message (Pass-through SNU module to ETRI module)
        self.odometry = {
            "rostopic_name": cfg.odometry_rostopic_name
        }

    def update_sensor_image_size(self, frame):
        self.width, self.height = frame.shape[1], frame.shape[0]


# Detector Option Class
class detector_options(object):
    def __init__(self, cfg):
        # Get Model Path for Detector
        self.model_dir = \
            os.path.join(cfg.detector.model_base_path, cfg.detector.name, cfg.env.time)

        # Set Actually Using Sensor Modalities
        self.sensor_dict = {
            "color": cfg.detector.sensors.color,
            "disparity": cfg.detector.sensors.disparity,
            "thermal": cfg.detector.sensors.thermal,
            "infrared": cfg.detector.sensors.infrared,
            "nightvision": cfg.detector.sensors.nightvision,
            "lidar": cfg.detector.sensors.lidar,
        }

        # GPU-device
        self.device = cfg.detector.device

        # Result Image Publish Flag
        self.is_result_publish = cfg.detector.visualization.is_result_publish

        # Tiny Area Threshold (remove bboxes with size under this amount)
        self.tiny_area_threshold = cfg.detector.tiny_area_threshold

        # Detection Arguments
        self.detection_args = {
            "n_classes": 81,
            # "input_h": 320, "input_w": 448,
            "input_h": 512, "input_w": 512,
        }

        # Backbone Arguments
        self.backbone_args = {
            "name": "res34level4",
            "pretrained": False,
        }

        # Detector Arguments
        # ( what is the difference between 'detection_args' ?!?! )
        # Detector Arguments
        # ( what is the difference between 'detection_args' ?!?! )
        self.detector_args = {
            "name": "yolov4",
            # "net_width": 448, "net_height": 320,
            "net_width": 512, "net_height": 512,
            # "thresh": 0.5, "hier_thresh": 0.5,
            "thresh": 0.65, "hier_thresh": 0.5,
            "nms_thresh": 0.45,
            "meta_path": os.path.join(
                os.path.dirname(os.path.dirname(curr_file_path)),
                "detection_lib", "darknet", "cfg", "coco.data"
            ),
            # "meta_path": "{}/detection_lib/darknet/cfg/coco.data".format(curr_file_path),

            # "name": "refinedet",
            # "is_bnorm": False, "tcb_ch": 256,
            # 'fmap_chs': [128, 256, 512, 128],
            # "fmap_sizes": [40, 20, 10, 5], "fmap_steps": [8, 16, 32, 64],
            # "anch_scale": [0.1, 0.2], "anch_min_sizes": [32, 64, 128, 256],
            # "anch_max_sizes": [], "anch_aspect_ratios": [[2], [2], [2], [2]],
            # "n_boxes_list": [3, 3, 3, 3], "is_anch_clip": True,
            # "backbone_args": {
            #     "name": "res34level4",
            #     "pretrained": False,
            # },
            # "postproc_args": {
            #     "name": "refinedet",
            #     'n_infer_rois': 300, 'device': 0, 'only_infer': True,
            #     # 'conf_thresh' ==>> Classification(2nd threshold)
            #     # 'nms_thresh': 0.45, 'conf_thresh': 0.3,
            #     # 'nms_thresh': 0.5, 'conf_thresh': 0.83,
            #     'nms_thresh': 0.5, 'conf_thresh': 0.4,
            #     # 'nms_thresh': 0.5, 'conf_thresh': 0.6,
            #     # 'pos_anchor_threshold ==>> Objectness(1st threshold)
            #     'max_boxes': 200, 'pos_anchor_threshold': 0.2,
            #     # 'max_boxes': 200, 'pos_anchor_threshold': 0.0001,
            #     'anch_scale': [0.1, 0.2],
            #     # dynamic edit (191016)!
            #     'max_w': 1000,
            # }
        }


# Tracker Option Class
class tracker_options(object):
    def __init__(self, cfg):
        # Set Device for Tracking
        self.device = cfg.tracker.device

        # Set Actually Using Sensor Modalities
        self.sensor_dict = {
            "color": cfg.tracker.sensors.color,
            "disparity": cfg.tracker.sensors.disparity,
            "thermal": cfg.tracker.sensors.thermal,
            "infrared": cfg.tracker.sensors.infrared,
            "nightvision": cfg.tracker.sensors.nightvision,
            "lidar": cfg.tracker.sensors.lidar,
        }

        # Result Image Publish Flag
        self.is_result_publish = cfg.tracker.visualization.is_result_publish

        # Set base Kalman Parameters
        self.kparams = kalman_params()
        self.kparams.get_params_from_cfg(kparams_cfg=cfg.tracker.kalman_params)

        # Association-related Parameters
        self.association = {
            # Trajectory Parameters
            "trk": {
                # Trajectory Initialization Age (from Trajectory Candidate)
                "init_age": cfg.tracker.association.trk.init_age,

                # Trajectory Destroy Age (destroy Trajectory with this amount of continuous unassociation)
                "destroy_age": cfg.tracker.association.trk.destroy_age,

                # Similarity Weights
                "similarity_weights": {
                    "histogram": cfg.tracker.association.trk.similarity_weights.histogram,
                    "iou": cfg.tracker.association.trk.similarity_weights.iou,
                    "distance": cfg.tracker.association.trk.similarity_weights.distance,
                },

                # Association Similarity Threshold (Trajectory to Detection: T2D)
                "similarity_thresh": cfg.tracker.association.trk.similarity_thresh,
            },

            # Trajectory Candidate Parameters
            "trk_cand": {
                # Trajectory Candidate Destroy Age
                "destroy_age": cfg.tracker.association.trk_cand.destroy_age,

                # Similarity Weights
                "similarity_weights": {
                    "iou": cfg.tracker.association.trk_cand.similarity_weights.iou,
                    "distance": cfg.tracker.association.trk_cand.similarity_weights.distance,
                },

                # Association Similarity Threshold (Detection to Detection: D2D)
                "similarity_thresh": cfg.tracker.association.trk_cand.similarity_thresh,
            },
        }

        # Disparity Modality Parameters
        self.disparity_params = {
            # Extraction Rate for Disparity Patch
            "extraction_roi_rate": 0.8,

            # Histogram Bin for Rough Depth Computation
            "rough_hist_bin": 100,
        }

        # LiDAR Modality Parameters
        self.lidar_params = {
            # LiDAR Random Sampling Number of Tracklet BBOX
            "sampling_number": 50,

            # LiDAR Window size
            "lidar_window_size": 4,
        }

        # KCF Params
        self.kcf_params = {
            # Regularization
            "lambda": 1e-4,

            # Padding
            "padding": 0.5,

            # Kernel Sigma
            "sigma": 0.5,

            # Output Sigma Factor
            "output_sigma_factor": 0.1,

            # Interpolation Factor
            "interp_factor": 0.075,

            # Resize Patch
            "resize": {
                "flag": True,
                "size": [64, 64],
            },

            # Cell Size (# of pixels per cell)
            "cell_size": 1,

            # Cosine Window Flag
            "is_cos_window": True,
        }

        # Trajectory Color Options
        self.trk_color_refresh_period = 16
        self.trajectory_colors = np.array(colormap(self.trk_color_refresh_period))


# Action Classifier Option Class
class aclassifier_options(object):
    def __init__(self, cfg):
        # Get Model Path for Action Classifier
        self.model_dir = \
            os.path.join(cfg.aclassifier.model_base_path, cfg.aclassifier.name, cfg.env.time)

        # Set Actually Using Sensor Modalities
        self.sensor_dict = {
            "color": cfg.aclassifier.sensors.color,
            "disparity": cfg.aclassifier.sensors.disparity,
            "thermal": cfg.aclassifier.sensors.thermal,
            "infrared": cfg.aclassifier.sensors.infrared,
            "nightvision": cfg.aclassifier.sensors.nightvision,
            "lidar": cfg.aclassifier.sensors.lidar,
        }

        # GPU-device for Action Classification
        self.device = cfg.aclassifier.device

        # Miscellaneous Parameters (for future usage)
        self.params = {}


# Visualizer Option Class
class visualizer_options(object):
    def __init__(self, cfg):
        # Set Fonts and Image Intervals, etc.
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.font_size = 1.2

        self.pad_pixels = 2
        self.info_interval = 4

        self.detection = {
            "is_draw": cfg.detector.visualization.is_draw,
            "is_show": cfg.detector.visualization.is_show,
            "auto_save": cfg.detector.visualization.auto_save,

            "is_show_label": None,
            "is_show_score": None,
            "is_show_fps": None,

            # (RGB) in our setting
            "bbox_color": cfg.detector.visualization.bbox_color,

            # Line-width
            "linewidth": 2,
        }

        self.tracking = {
            "is_draw": cfg.tracker.visualization.is_draw,
            "is_show": cfg.tracker.visualization.is_show,
            "auto_save": cfg.tracker.visualization.auto_save,

            "is_show_id": None,
            "is_show_3d_coord": None,
            "is_show_depth": None,
            "is_show_fps": None,

            "linewidth": 2,
        }

        # TODO: Change this Working Method
        if self.detection["auto_save"] is True or self.tracking["auto_save"] is True:
            sample_result_base_dir = \
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_results")
            if os.path.isdir(sample_result_base_dir) is False:
                os.mkdir(sample_result_base_dir)

        self.aclassifier = {
            "is_draw": cfg.aclassifier.visualization.is_draw,
            "is_show": cfg.aclassifier.visualization.is_show,
        }

        self.top_view = {
            "is_draw": cfg.tracker.visualization.top_view.is_draw,
            "is_show": cfg.tracker.visualization.top_view.is_show,

            "map_size": cfg.tracker.visualization.top_view.map_size,

            "trk_radius": cfg.tracker.visualization.top_view.trk_radius,
        }

    def correct_flag_options(self):
        if self.detection["is_draw"] is False:
            self.detection["is_show"], self.detection["auto_save"] = False, False

        if self.tracking["is_draw"] is False:
            self.tracking["is_show"], self.tracking["auto_save"] = False, False

        if self.top_view["is_draw"] is False:
            self.top_view["is_show"] = False
