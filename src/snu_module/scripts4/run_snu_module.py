#!/usr/bin/env python
"""

WRITE COMMENTS


"""

import cv2
import copy
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg
from osr_msgs.msg import Tracks

import options
import ros_utils
import snu_algorithms
import snu_visualizer

from module_detection import load_model as load_det_model
from module_action import load_model as load_acl_model


# Get Agent Type and Agent Name
agent_type = "rosbagfile"
agent_name = "snu_osr"


# Define SNU Module Class
class snu_module(ros_utils.ros_multimodal_subscriber):
    def __init__(self, opts):
        super(snu_module, self).__init__(opts)

        # Initialize Tracklet and Tracklet Candidates
        self.trks, self.trk_cands = [], []

        # Initialize Frame Index
        self.fidx = 0

        # Declare SNU Visualizer
        self.visualizer = snu_visualizer.visualizer(opts=opts)

        # Sensor Synchronization Flag
        self.is_all_synchronous = None

        # ROS Publisher
        self.tracks_pub = rospy.Publisher(
            opts.publish_mesg["tracks"], Tracks, queue_size=1
        )

    def synchronize_sensor_msg(self, sync_target_modal_list):
        self.is_all_synchronous = None

        # ROS Target Sensors Synchronization
        while self.is_all_synchronous is not True:
            sync_modal_list = []
            msg_flag_dict = self.collect_all_messages()
            for modal, modal_msg in msg_flag_dict.items():
                if modal_msg is not None:
                    sync_modal_list.append(modal)
                if modal in sync_modal_list:
                    self.is_all_synchronous = False
                    continue
            # Check for Sub-list Relation
            if all((k in sync_modal_list for k in sync_target_modal_list)):
                self.is_all_synchronous = True

    def publish_tracks(self, tracklets):
        pass

    def __call__(self, module_name):
        # Load Detection and Action Classification Models
        frameworks = {
            "det": load_det_model(self.opts),
            "acl": load_acl_model(self.opts),
        }
        print("Loading Detection and Action Classification Models...!")

        # Initialize SNU Algorithm Class
        snu_usr = snu_algorithms.snu_algorithms(frameworks=frameworks)
        print("Loading SNU Algorithm...!")

        # ROS Node Initialization
        rospy.init_node(module_name, anonymous=True)

        # ROS Sensor Modal Synchronization Target List
        # sync_target_modal_list = ["color", "disparity", "thermal"]
        sync_target_modal_list = ["color", "disparity", "thermal", "infrared", "nightvision", "lidar"]

        while not rospy.is_shutdown():
            rospy.sleep(self.opts.node_sleep_time_for_sensor_sync)

            # ROS Sensor Message Synchronization
            self.synchronize_sensor_msg(sync_target_modal_list=sync_target_modal_list)

            # Get Current ROS Timestamp
            curr_timestamp = rospy.Time.now()

            # Convert Messages to Sensor Data
            self.update_all_modals(null_timestamp=curr_timestamp)

            # Increase Frame Index
            self.fidx += 1

            # SNU USR Algorithm
            tracklets, detections, module_time_dict = snu_usr(
                sync_data_dict=self.collect_all_sensors(),
                fidx=self.fidx, opts=self.opts
            )

            # # Draw Color Image Sequence
            # self.visualizer.visualize_modal_frames(self.color)

            # Draw Results
            self.visualizer(
                sensor_data=self.color, tracklets=tracklets, detections=detections
            )


def main():
    # Load Options
    opts = options.snu_option_class(agent_type=agent_type, agent_name=agent_name)

    # Initialize SNU Module
    snu_usr = snu_module(opts=opts)

    # Run SNU Module
    snu_usr(module_name="snu_module")


if __name__ == "__main__":
    main()
