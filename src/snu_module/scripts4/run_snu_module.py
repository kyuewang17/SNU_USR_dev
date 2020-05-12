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

        # Asynchronous Sensor Modals
        self.is_all_synchronous = None
        self.async_modals = []

        # ROS Publisher
        self.tracks_pub = rospy.Publisher(
            opts.publish_mesg["tracks"], Tracks, queue_size=1
        )

    def notify_asynchronous_sensors(self):
        async_modals = []
        sensor_flag_dict = self.collect_all_sensor_flags()

        if not all(sensor_flag_dict.values()):
            for idx, (modal, sensor_flag) in enumerate(sensor_flag_dict.items()):
                if sensor_flag is not True:
                    async_modals.append(modal)
            print(async_modals)
            self.is_all_synchronous = False
        else:
            self.is_all_synchronous = True

        self.async_modals = async_modals

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

        while not rospy.is_shutdown():
            # Mandatory ROS Node Sleep
            rospy.sleep(self.opts.node_sleep_time_for_sensor_sync)

            # Notify Asynchronous Sensors
            self.notify_asynchronous_sensors()

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

            # Draw Color Image Sequence
            self.visualizer.visualize_modal_frames(self.color)

            # # Draw Detection Results
            # self.visualizer(
            #     sensor_data=self.color, tracklets=tracklets, detections=detections
            # )


def main():
    # Load Options
    opts = options.snu_option_class(agent_type=agent_type, agent_name=agent_name)

    # Initialize SNU Module
    snu_usr = snu_module(opts=opts)

    # Run SNU Module
    snu_usr(module_name="snu_module")


if __name__ == "__main__":
    main()
