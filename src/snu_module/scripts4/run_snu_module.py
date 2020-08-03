#!/usr/bin/env python
"""
SNU Integrated Module v4.5

- Code Massively Re-arranged
    - Module Segmentation
    - Multimodal Sensor Synchronization Source Code Customized (original from KIRO)
    - LiDAR Point-Cloud Projection Optimized
        - Do Projection on Trajectory BBOX
    - Changed Mis-used Terminologies
        - (ex) Tracklet -> Trajectory
    - Optimize Trivial Things...

"""
import os
import argparse
import time
import rospy
import logging
import tf2_ros

import snu_visualizer
from utils.ros.coverage import coverage
from utils.ros.sensors import snu_SyncSubscriber
from snu_algorithms_v4 import snu_algorithms
from utils.profiling import Timer

from module_detection import load_model as load_det_model
from module_action import load_model as load_acl_model

from config import cfg


# Argument Parser
parser = argparse.ArgumentParser(
    prog="SNU-Integrated-v4.5", description="SNU Integrated Algorithm"
)
parser.add_argument(
    "--config", "-C",
    default=os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
)
args = parser.parse_args()


# Set Logger
def set_logger(logging_level=logging.INFO):
    # Define Logger
    logger = logging.getLogger()

    # Set Logger Display Level
    logger.setLevel(level=logging_level)

    # Set Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("[%(levelname)s] | %(asctime)s : %(message)s")
    )
    logger.addHandler(stream_handler)

    return logger


# Define SNU Module Class
class snu_module(coverage):
    def __init__(self, opts):
        super(snu_module, self).__init__(
            opts=opts, is_sensor_param_file=self.sensor_param_file_check()
        )

        # Initialize Logger Variable
        self.logger = set_logger(logging_level=logging.INFO)

        # Initialize Frame Index
        self.fidx = 0

        # Synchronized Timestamp of Multimodal Sensors
        self.sync_stamp = None

        # Declare SNU Visualizer
        self.visualizer = snu_visualizer.visualizer(opts=opts)

        # Declare ROS Synchronization Switch Dictionary
        self.ros_sync_switch_dict = {
            "color": True,
            "disparity": False, "aligned_disparity": True,
            "thermal": True,
            "infrared": True,
            "nightvision": True,
        }

    @staticmethod
    def sensor_param_file_check():
        return False

    def gather_all_sensor_params_via_files(self):
        raise NotImplementedError()

    # Call as Function
    def __call__(self, module_name):
        # Load Detection and Action Classification Models
        frameworks = {
            "det": load_det_model(opts=self.opts),
            "acl": load_acl_model(opts=self.opts),
        }
        self.logger.info("Detector and Action Classifier Neural Network Model Loaded...!")
        time.sleep(0.01)

        # Initialize SNU Algorithm Class
        snu_usr = snu_algorithms(frameworks=frameworks, opts=self.opts)
        self.logger.info("SNU Algorithm Loaded...!")
        time.sleep(0.01)

        # ROS Node Initialization
        self.logger.info("ROS Node Initialization")
        rospy.init_node(name=module_name, anonymous=True)

        # Subscribe for tf_static
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(buffer=tf_buffer)

        # Iterate Loop until "tf_static is heard"
        while self.tf_transform is None:
            try:
                self.tf_transform = tf_buffer.lookup_transform(
                    "rgb_frame", 'velodyne_frame_from_rgb', rospy.Time(0)
                )
            except:
                rospy.logwarn("SNU-MODULE : TF_STATIC Transform Unreadable...!")

        # Load ROS Synchronized Subscriber
        rospy.loginfo("Load ROS Synchronized Subscriber...!")
        sync_ss = snu_SyncSubscriber(
            ros_sync_switch_dict=self.ros_sync_switch_dict, options=self.opts
        )

        # ROS Loop Starts
















































if __name__ == "__main__":
    pass

