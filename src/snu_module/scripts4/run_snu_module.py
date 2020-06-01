#!/usr/bin/env python
"""

WRITE COMMENTS


"""
import time
import rospy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from osr_msgs.msg import Tracks

import options
import ros_utils
import snu_algorithms
import snu_visualizer

from module_detection import load_model as load_det_model
from module_action import load_model as load_acl_model


# Get Agent Type and Agent Name
agent_type = "dynamic"
agent_name = "snu-dynamic-3"


# Define SNU Module Class
class snu_module(ros_utils.ros_multimodal_subscriber):
    def __init__(self, opts):
        super(snu_module, self).__init__(opts)

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
        self.det_result_pub = rospy.Publisher(
            opts.publish_mesg["det_result_rostopic_name"], Image, queue_size=1
        )
        self.trk_acl_result_pub = rospy.Publisher(
            opts.publish_mesg["trk_acl_result_rostopic_name"], Image, queue_size=1
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
        # Wrap Tracklets into ROS Topic Type
        out_tracks = ros_utils.wrap_tracks(tracklets, self.odometry_msg)
        self.tracks_pub.publish(out_tracks)

    def publish_snu_result_image(self, result_frame_dict):
        for module, result_frame in result_frame_dict.items():
            if result_frame is not None:
                if module == "det":
                    self.det_result_pub.publish(
                        self.pub_bridge.cv2_to_imgmsg(
                            result_frame, "rgb8"
                        )
                    )
                elif module == "trk_acl":
                    self.trk_acl_result_pub.publish(
                        self.pub_bridge.cv2_to_imgmsg(
                            result_frame, "rgb8"
                        )
                    )
                else:
                    assert 0, "Undefined module!"

    def __call__(self, module_name):
        # Load Detection and Action Classification Models
        frameworks = {
            "det": load_det_model(self.opts),
            "acl": load_acl_model(self.opts),
        }
        print("[SNU Algorithm Flow #01] Loading Detection and Action Classification Models...!")
        time.sleep(1)

        # Initialize SNU Algorithm Class
        snu_usr = snu_algorithms.snu_algorithms(frameworks=frameworks)
        print("[SNU Algorithm Flow #02] Loading SNU Algorithm...!")
        time.sleep(1)

        # Attempt to Gather all Sensor Parameters by File
        self.gather_all_sensor_parameters()

        # Time Sleep
        time.sleep(1)

        # ROS Node Initialization
        rospy.init_node(module_name, anonymous=True)

        # ROS Sensor Modal Synchronization Target List
        sync_target_modal_list = ["color", "disparity", "thermal"]  # 190823_kiro_lidar_camera_calib.bag
        # sync_target_modal_list = ["color", "disparity", "thermal", "lidar"]  # kiro_rosbag.bag
        # sync_target_modal_list = ["color", "disparity", "thermal", "infrared", "nightvision"]

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
            result_frame_dict = self.visualizer(
                sensor_data=self.color, tracklets=tracklets, detections=detections
            )

            # # Top-view Result
            # self.visualizer.visualize_top_view_tracklets(tracklets=tracklets)

            # Publish Tracks
            self.publish_tracks(tracklets=tracklets)

            # Publish SNU Result Image Results
            self.publish_snu_result_image(result_frame_dict=result_frame_dict)


def main():
    # Load Options
    opts = options.snu_option_class(agent_type=agent_type, agent_name=agent_name)

    # Initialize SNU Module
    snu_usr = snu_module(opts=opts)

    # Run SNU Module
    snu_usr(module_name="snu_module")


if __name__ == "__main__":
    main()
