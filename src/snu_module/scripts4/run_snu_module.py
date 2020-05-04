#!/usr/bin/env python

"""
SNU Integrated Module v3.0
    - Main Execution Code (Run Code)
    - ROS embedded version
        - we now develop only ROS-embedded version only
        - test for image sequences can be done in individual module files
          (using namespace functionality of Python, will be implemented)
    - Synchronize Multimodal Sensors via "Approximate Time Synchronizer"

    - < Project Information >
        - "Development of multimodal sensor-based intelligent systems for outdoor surveillance robots"
        - Project Total Period : 2017-04-01 ~ 2012-12-31
        - Current Year : 2020-01-01 ~ 2020-12-31 (4/5)

    - < Institutions & Researchers >
        - Seoul National University
            - Perception and Intelligence Laboratory (PIL)
                [1] Kyuewang Lee (kyuewang@snu.ac.kr)
                [2] Daeho Um (umdaeho1@gmail.com)
            - Machine Intelligence and Pattern Recognition Laboratory (MIPAL)
                [1] Jae-Young Yoo (yoojy31@snu.ac.kr)
                [2] Jee-soo Kim (kimjiss0305@snu.ac.kr)
                [3] Hojun Lee (hojun815@snu.ac.kr)
                [4] Inseop Chung (jis3613@snu.ac.kr)

    - < Updates >
        - Initial Version (started writing code on 2020-04-30)
            - Code Adapted for Anaconda envs
            - TBA

    - < Dependencies >
        - PyTorch == 1.1.0
            - torchvision == 0.3.0
        - CUDA == 10.0
            - cuDNN == 7.5.0
        - ROS-kinetics
            - need rospkg inside the Anaconda Virtual Environment
            - download rospkg via pip
        - opencv-python (download using pip)
        - empy (pip download)
        - yaml (download "pyyaml")
        - cython (install via conda) : ????!?!?!?!?!

        - numpy, numba, scipy, FilterPy, sklearn, yacs

    - < Some Memos >
        - Watch out for OpenCV Image Formatting
            - openCV basically takes RGB input image as BGR image format
            - RGB image is shown in BGR format

        - Notes about ROS RGB image input
            - rosbag file from Pohang agents
                - BGR format
            - rostopic from D435i RGBD-camera
                - RGB format

"""

# Import Modules
import cv2
import socket
import datetime
import logging

import copy
import numpy as np
import rospy
import message_filters
from rostopic import get_topic_type
from cv_bridge import CvBridge, CvBridgeError

# Import ROS Messages
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg

# Import Custom Modules
import options
import ros_utils
import snu_utils.general_functions as gfuncs
import snu_algorithms
import snu_visualizer

# Get Computer Screen Geometry
# screen_geometry_dict = gfuncs.get_screen_geometry()

# Get Agent Type and Agent Name
# agent_type = "rosbagfile"
agent_type = "dynamic"
agent_name = "snu_osr"

# Initialize SNU Module Options
opts = options.snu_option_class(agent_type=agent_type, agent_name=agent_name)


# Define SNU Module for ROS Embedding
class snu_module_ros(object):
    def __init__(self, subscribe_rate=10):
        # Initialize Tracklets and Tracklet Candidates (Lists for Objects)
        self.trackers, self.tracker_cands = [], []

        # Declare SNU Visualizer
        self.visualizer = snu_visualizer.visualizer(opts=opts)

        # Frame Count Index
        self.fidx = 0

        # Sensor Subscribe Rate (in hertz [Hz])
        self.subscribe_rate = subscribe_rate

        # [1] Color Modality
        self.color_msg = None
        self.color = ros_utils.ros_sensor_image(modal_type="color")

        # [2] Disparity Modality (Aligned)
        self.disparity_msg = None
        self.disparity = ros_utils.ros_sensor_image(modal_type="disparity")

        # [3] Thermal Modality
        self.thermal_msg = None
        self.thermal = ros_utils.ros_sensor_image(modal_type="thermal")

        # [4] Infrared Modality
        self.infrared_msg = None
        self.infrared = ros_utils.ros_sensor_image(modal_type="infrared")

        # [5] NightVision Modality
        self.nightvision_msg = None
        self.nightvision = ros_utils.ros_sensor_image(modal_type="nightvision")

        # [6] LiDAR Modality
        self.lidar_msg = None
        self.lidar = ros_utils.ros_sensor_lidar()

        # Odometry Message Variable (Pass-through)
        self.odometry_msg = None

        # CvBridges (for both Subscribe and Publish)
        self.sub_bridge, self.pub_bridge = CvBridge(), CvBridge()

        # imshow window position
        # self.imshow_window_x = int(screen_geometry_dict["margin_length"]["left_pixels"] * 1.05)
        # self.imshow_window_y = int(screen_geometry_dict["margin_length"]["top_pixels"] * 1.05)

        # Synchronized Subscribers (use Approximate Synchronized Subscribe)
        self.color_sub = message_filters.Subscriber(opts.sensors.color["rostopic_name"], Image)
        self.disparity_sub = message_filters.Subscriber(opts.sensors.disparity["rostopic_name"], Image)
        self.thermal_sub = message_filters.Subscriber(opts.sensors.thermal["rostopic_name"], Image)
        self.infrared_sub = message_filters.Subscriber(opts.sensors.infrared["rostopic_name"], Image)
        self.nightvision_sub = message_filters.Subscriber(opts.sensors.nightvision["rostopic_name"], Image)
        self.lidar_sub = message_filters.Subscriber(opts.sensors.lidar["rostopic_name"], PointCloud2)

        # Camerainfo Subscribers (individual subscribe)
        self.color_camerainfo_sub = rospy.Subscriber(
            opts.sensors.color["camerainfo_rostopic_name"], numpy_msg(CameraInfo),
            self.color_cam_params_callback
        )
        self.disparity_camerainfo_sub = rospy.Subscriber(
            opts.sensors.disparity["camerainfo_rostopic_name"], numpy_msg(CameraInfo),
            self.disparity_cam_params_callback
        )

        # Subscribe Odometry
        self.odometry_sub = rospy.Subscriber(
            opts.sensors.odometry["rostopic_name"], Odometry, self.odometry_callback
        )

        # Define Synchronous Subscriber Dictionary for Existing Multimodal Sensors in current Agent
        self.synchronous_sub_node_dict = self.select_sensors(opts.modal_switch_dict)

        # Approximate Time Synchronizer
        # slop: time window for synchronous subscribe
        # slop = (1.0 / subscribe_rate) * 0.5
        slop = 0.04
        self.ats = message_filters.ApproximateTimeSynchronizer(
            self.synchronous_sub_node_dict.values(),
            queue_size=1, slop=slop, allow_headerless=True
        )

    # Color Modal Camerainfo Callback Function
    def color_cam_params_callback(self, msg):
        self.color.update_cam_params(msg)

    # Disparity Modal Camerainfo Callback Function
    def disparity_cam_params_callback(self, msg):
        self.disparity.update_cam_params(msg)

    # Odometry Callback Function
    def odometry_callback(self, msg):
        self.odometry_msg = msg

    # Select Sensors to Subscribe Synchronously
    def select_sensors(self, modal_switch_dict):
        assert (type(modal_switch_dict) == dict), "Input Argument Must be type <dict>!"

        # Define Synchronous Subscriber Dictionary for Multimodal Sensors
        synchronous_sub_node_dict = {
            "color": self.color_sub,
            "disparity": self.disparity_sub,
            "thermal": self.thermal_sub,
            "infrared": self.infrared_sub,
            "nightvision": self.nightvision_sub,
            "lidar": self.lidar_sub,
        }

        # Eliminate from Dictionary if Modal Does not Exist in the Agent
        for modal, switch in modal_switch_dict.items():
            if switch is False:
                del synchronous_sub_node_dict[modal]

        return synchronous_sub_node_dict

    # Image Message to OpenCV Image
    def imgmsg_to_cv2(self, img_msg, msg_encode_type):
        return self.sub_bridge.imgmsg_to_cv2(img_msg, msg_encode_type)

    # Custom Synchronous Subscribe
    def multimodal_callback(self, _check_run_time=False):
        callback_msg_dict, callback_msg_list = {}, []

        # Parse Approximate Time Synchronizer
        for queue_node_dict in self.ats.queues:
            queue_node_list = queue_node_dict.items()
            queued_msg_list = []
            for msg_tuple in queue_node_list:
                queued_msg_list.append(msg_tuple[1])
            callback_msg_list.append(queued_msg_list)

        # Initialize Multimodal Sensor Message Dictionary
        keys = self.synchronous_sub_node_dict.keys()
        if len(keys) == len(callback_msg_list):
            for key_idx, key in enumerate(keys):
                callback_msg_dict[key] = callback_msg_list[key_idx]
        else:
            for key_idx, key in enumerate(keys):
                callback_msg_dict[key] = callback_msg_list

        # Parse-out Sensor Messages
        for modal_type, sensor_msg in callback_msg_dict.items():
            if len(sensor_msg) > 2:
                assert 0, "More than 1 messages subscribed for modal [%s]!" % modal_type
            elif len(sensor_msg) == 0:
                # print("(FIDX:%08d) [WARNING] No Sensor Messages are Retrieved Synchronously" % self.fidx)
                pass
            else:
                # print("(FIDX:%08d) [SYNC SUCCESS]" % self.fidx)
                # Convert to OpenCV ndarray image w.r.t. Sensor Image Modal Type
                if modal_type == "color":
                    self.color_msg = sensor_msg[0]
                elif modal_type == "disparity":
                    self.disparity_msg = sensor_msg[0]
                elif modal_type == "thermal":
                    self.thermal_msg = sensor_msg[0]
                elif modal_type == "infrared":
                    self.infrared_msg = sensor_msg[0]
                elif modal_type == "nightvision":
                    self.nightvision_msg = sensor_msg[0]
                elif modal_type == "lidar":
                    self.lidar_msg = sensor_msg[0]
                else:
                    assert 0, "Current modal type is not defined!"

    # Update Multimodal Sensor Data
    def update_sensor_data(self, sensor_opts, _check_run_time=False):
        # For Null Header
        null_header = Header()
        null_header.stamp = rospy.Time.now()

        # ROS Message Decoding for Color Modality
        if self.color_msg is not None:
            # For BGR format
            if self.color_msg.encoding.__contains__("bgr") is True:
                # Convert BGR to RGB
                color_frame = cv2.cvtColor(
                    self.imgmsg_to_cv2(self.color_msg, sensor_opts.color["imgmsg_to_cv2_encoding"]),
                    cv2.COLOR_BGR2RGB
                )
            # For RGB format
            elif self.color_msg.encoding.__contains__("rgb") is True:
                color_frame = self.imgmsg_to_cv2(self.color_msg, sensor_opts.color["imgmsg_to_cv2_encoding"])
            # For GrayScale Color Camera Image
            elif self.color_msg.encoding.__contains__("mono") is True:
                color_frame = self.imgmsg_to_cv2(self.color_msg, "8UC1")
            else:
                assert 0, "Current Encoding Type is not Defined!"

            # Update Color Frame
            self.color.update(color_frame, self.color_msg.header)

        else:
            self.color.update(None, null_header)

        # ROS Message Decoding for Disparity Modality
        if self.disparity_msg is not None:
            self.disparity.update(
                self.imgmsg_to_cv2(self.disparity_msg, sensor_opts.disparity["imgmsg_to_cv2_encoding"]),
                self.disparity_msg.header
            )
        else:
            self.disparity.update(None, null_header)

        # ROS Message Decoding for Thermal Modality
        if self.thermal_msg is not None:
            self.thermal.update(
                self.imgmsg_to_cv2(self.thermal_msg, sensor_opts.thermal["imgmsg_to_cv2_encoding"]),
                self.thermal_msg.header
            )
        else:
            self.thermal.update(None, null_header)

        # ROS Message Decoding for Infrared Modality
        if self.infrared_msg is not None:
            self.infrared.update(
                self.imgmsg_to_cv2(self.infrared_msg, sensor_opts.infrared["imgmsg_to_cv2_encoding"]),
                self.infrared_msg.header
            )
        else:
            self.infrared.update(None, null_header)

        # ROS Message Decoding for Nightvision Modality
        if self.nightvision_msg is not None:
            self.nightvision.update(
                self.imgmsg_to_cv2(self.nightvision_msg, sensor_opts.nightvision["imgmsg_to_cv2_encoding"]),
                self.nightvision_msg.header
            )
        else:
            self.nightvision.update(None, null_header)

        # ROS Message Decoding for LiDAR Modality
        if self.lidar_msg is not None:
            self.lidar.update(self.lidar_msg, self.lidar_msg.header)
        else:
            self.lidar.update(None, null_header)

    # Switch Back New Sensor Data Flags to False (flush new data flag)
    def flush_new_data_flags(self):
        # Loop through all data and switch back new sensor data flags
        for modal in self.synchronous_sub_node_dict.keys():
            modal_data_object = getattr(self, modal)
            modal_data_object.flush_new_data_flag()

    # Pack Synchronized Sensor Data Object as a Dictionary
    def pack_sync_sensor_data(self, _check_run_time=False):
        sync_data_object = {}
        none_value_sensors = 0

        # Loop through all synchronous sensor data modalities and gather the sensor objects
        for modal in self.synchronous_sub_node_dict.keys():
            sensor_data = getattr(self, modal)

            # Check for "None" Sensor Values
            if hasattr(sensor_data, "frame") is True:
                if sensor_data.frame is None:
                    none_value_sensors += 1
            elif hasattr(sensor_data, "pc2_msg") is True:
                if sensor_data.pc2_msg is None:
                    none_value_sensors += 1

            sync_data_object[modal] = sensor_data

        # Check if all sensor values are "None"
        if none_value_sensors == 0:
            all_synced = True
        else:
            all_synced = False

        return sync_data_object, all_synced

    # Run Algorithm
    def run_algorithm(self):
        r = rospy.Rate(self.subscribe_rate)
        while not rospy.is_shutdown():
            # Multimodal Synchronous Sensor Dictionary Callback (check elapsed time)
            self.multimodal_callback()

            # Update Multimodal Sensor Data
            self.update_sensor_data(opts.sensors)

            # Pack Synchronized Sensor Data as a Dictionary
            sync_data_dict, all_synced = self.pack_sync_sensor_data()

            # Wait SNU Algorithm Until All Target Sensors are Synchronized
            if all_synced is True:
                # Increase Frame Index Count
                # (NOTE): this frame index runs in a slightly different manner, compared to the
                #         frame index in the sensor_data object("ros_sensor_xxx" class)
                # -> Ordinarily, use this frame index as the representative frame index
                self.fidx += 1

                # SNU Integrated Algorithm
                self.trackers, self.tracker_cands, detections, algorithm_time_dict = \
                    snu_algorithms.usr_integrated_snu(
                        fidx=self.fidx,
                        sync_data_dict=sync_data_dict,
                        tracklets=self.trackers, tracklet_cands=self.tracker_cands,
                        opts=opts
                    )

                # Select Visualization Frame
                vis_data = copy.deepcopy(sync_data_dict["color"])

                # Run Visualizer
                self.visualizer(sensor_data=vis_data, tracklets=self.trackers, detections=detections)

                # Publish ROS Message
                # TODO: Write a Publisher Code

            # Switch Back New Sensor Data Flags
            self.flush_new_data_flags()


# Main Code
def main():
    # ROS Node Initialization
    rospy.init_node("snu_module", anonymous=True)

    # ROS Class Initialization
    ros_embedded_snu_module = snu_module_ros(subscribe_rate=10)

    # Run Algorithm
    ros_embedded_snu_module.run_algorithm()

    # Spin
    rospy.spin()


# Main Namespace
if __name__ == "__main__":
    main()
