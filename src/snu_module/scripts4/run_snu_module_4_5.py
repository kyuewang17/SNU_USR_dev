#!/usr/bin/env python
"""
SNU Integrated Module 4.5

"""
import os
import time
import argparse
import rospy
import logging
import tf2_ros

from utils.ros import coverage, sensors, wrapper
from snu_visualizer import visualizer
from snu_algorithms_v4 import snu_algorithms

from module_detection import load_model as load_det_model
from module_action import load_model as load_acl_model


# Argument Parser
parser = argparse.ArgumentParser(
    description="SNU Integrated Algorithm", prog="SNU-Integrated-4.5"
)
parser.add_argument(
    "--config", "-C",
    default=os.path.join(os.path.dirname(__file__), "config", "curr_agent", "config.yaml"),
    type=str, help="Configuration YAML File Path"
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
class snu_module(coverage.coverage):
    def __init__(self, opts):
        super(snu_module, self).__init__(opts=opts)

        # Initialize Logger Variable
        self.logger = set_logger(logging_level=logging.INFO)

        # Initialize Frame Index
        self.fidx = 0

        # Initialize Synchronized Timestamp of Multimodal Sensors
        self.sync_stamp = None

        # Declare Visualizer
        self.visualizer = visualizer(opts=opts)

        # Declare ROS Synchronization Switch for Target Sensors
        self.ros_sync_switch_dict = {
            "color": True,
            "disparity": False,
            "aligned_disparity": True,
            "thermal": True,
            "infrared": True,
            "nightvision": True,
        }

    # Call Module as Function
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

























