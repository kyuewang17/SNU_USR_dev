#!/usr/bin/env python
"""
SNU Integrated Module V3.0 Candidate
"""
import copy
import datetime
import numpy as np
import cv2
import rospy
import message_filters
from rostopic import get_topic_type
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from rospy.numpy_msg import numpy_msg

import snu_utils.general_functions as gfuncs


# Get Computer Screen Geometry
screen_geometry_dict = gfuncs.get_screen_geometry()


# Topic Dictionary
topic_dict = {
    # D435i Camera
    "d435i": {
        # RGB (color)
        "rgb": {
            "raw": "/osr/image_color",
            "camera_info": "/camera/color/image_color_camerainfo",
        },

        # Depth (stereo)
        "depth": {
            "raw": "/osr/image_depth",
            "aligned": "/osr/image_aligned_depth",
            "camera_info": "/camera/depth/image_depth_camerainfo",
        },
    },
}


# Class for ROS Sensor Images
class ros_sensor_image(object):
    def __init__(self):
        self.frame = None
        self.seq, self.stamp = None, None

        # for Camera Projection Matrix
        self.cam_params = None

        # New Frame Flag
        self.is_new_frame = False

    # Update
    def update(self, frame, msg_header):
        self.frame = frame
        self.seq, self.stamp = msg_header.seq, msg_header.stamp
        self.is_new_frame = True

    # Update Camera Parameters
    def update_cam_param(self, msg):
        self.cam_params = {
            "D": msg.D.reshape((5, 1)),  # Distortion Matrix
            "K": msg.K.reshape((3, 3)),  # Intrinsic Matrix
            "R": msg.R.reshape((3, 3)),  # Rotation Matrix
            "P": msg.P.reshape((3, 4)),  # Projection Matrix
        }


class ros_embedder(object):
    def __init__(self, subscribe_rate=10):
        # Frame Count Index
        self.fidx = 0

        # Sensor Subscribe Rate
        self.subscribe_rate = subscribe_rate

        # [1] RGB Sensor Image
        self.rgb_msg = None
        self.rgb = ros_sensor_image()

        # [2] Depth Sensor Image
        self.raw_depth_msg = None

        # [3] Aligned Depth Sensor Image
        self.depth_msg = None
        self.depth = ros_sensor_image()

        # [4] LIDAR Pointcloud
        self.lidar_msg, self.is_lidar_new = None, False

        # CvBridge
        self.bridge = CvBridge()

        # imshow window position
        self.imshow_window_x = int(screen_geometry_dict["margin_length"]["left_pixels"] * 1.05)
        self.imshow_window_y = int(screen_geometry_dict["margin_length"]["top_pixels"] * 1.05)

        # Initialize Subscribers for each modal sensors (for Approximate Synchronous Subscribe)
        # self.rgb_sub = message_filters.Subscriber("/osr/image_color", Image)
        # self.depth_sub = message_filters.Subscriber("/osr/image_aligned_depth", Image)

        # self.rgb_sub = \
        #     message_filters.Subscriber(topic_dict["d435i"]["rgb"]["raw"])

        self.rgb_sub = message_filters.Subscriber("/osr/image_color", Image)
        self.depth_sub = message_filters.Subscriber("/osr/image_depth", Image)

        # Subscribe these independently
        self.rgb_cam_params_sub = rospy.Subscriber("/camera/color/image_color_camerainfo", numpy_msg(CameraInfo), self.rgb_cam_params_callback)
        self.depth_cam_params_sub = rospy.Subscriber("/camera/depth/image_color_camerainfo", numpy_msg(CameraInfo), self.depth_cam_params_callback)

        # Approximate Time Synchronizer (slop ==> time window for synchronous subscribe)
        self.synchronous_sub_node_dict = {
            "rgb": self.rgb_sub,
            "depth": self.depth_sub,
        }
        self.ats = message_filters.ApproximateTimeSynchronizer(
                self.synchronous_sub_node_dict.values(), queue_size=1, slop=0.01, allow_headerless=False)

    # Image Message to OpenCV Image (Separate This from Callback Function)
    def imgmsg_to_cv2(self, img_msg, msg_encode_type):
        return self.bridge.imgmsg_to_cv2(img_msg, msg_encode_type)

    # Custom Synchronous Subscribe
    def multimodal_callback(self):
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

        for modal_type, sensor_msg in callback_msg_dict.items():
            if len(sensor_msg) > 2:
                assert 0, "More than 1 messages subscribed for modal [%s]!" % modal_type
            elif len(sensor_msg) == 0:
                pass
            else:
                # Convert to openCV ndarray image w.r.t. Sensor Image Modal Type
                if modal_type == "rgb":
                    self.rgb_msg = sensor_msg[0]
                elif modal_type == "depth":
                    self.depth_msg = sensor_msg[0]
                else:
                    assert 0, "UNDEFINED modal [%s]" % modal_type

    # Raw Depth Callback
    def raw_depth_msg_callback(self, msg):
        self.raw_depth_msg = msg

    # RGB Camera Parameters Callback
    def rgb_cam_params_callback(self, msg):
        self.rgb.update_cam_param(msg)

    # Depth Camera Parameters Callback
    def depth_cam_params_callback(self, msg):
        self.depth.update_cam_param(msg)

    # Update Multimodal Sensor Images
    def update_sensor_images(self):
        # for Null Header
        null_header = Header()
        null_header.stamp = rospy.Time.now()

        if self.rgb_msg is not None:
            if self.rgb_msg.encoding.__contains__("bgr") is True:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(self.imgmsg_to_cv2(self.rgb_msg, "8UC3"),
                                         cv2.COLOR_BGR2RGB)
            elif self.rgb_msg.encoding.__contains__("rgb") is True:
                rgb_frame = self.imgmsg_to_cv2(self.rgb_msg, "8UC3")
            else:
                assert 0, "Unexpected Image Message Encoding for Color Image"

            self.rgb.update(rgb_frame, self.rgb_msg.header)
        else:
            self.rgb.update(None, null_header)

        if self.depth_msg is not None:
            self.depth.update(self.imgmsg_to_cv2(self.depth_msg, "16UC1"), self.depth_msg.header)
        else:
            self.depth.update(None, null_header)

    # Switch back New Sensor Flags
    def falsify_new_sensor_flags(self):
        self.rgb.is_new_frame, self.depth.is_new_frame = False, False

    # Run Algorithm
    def run_algorithm(self):
        r = rospy.Rate(self.subscribe_rate)
        while not rospy.is_shutdown():
            # Increase Frame Index Count
            self.fidx += 1

            # Multimodal Synchronous Sensor Dictionary Callback (dtime: around 0.00005 secs)
            self.multimodal_callback()

            # Update Multimodal Sensor Images
            self.update_sensor_images()

            # Visualize
            if self.rgb.frame is not None and self.depth.frame is not None:
                # Depth Weight for Interpolation
                gamma_depth = 0.8

                grayscale_frame = cv2.cvtColor(self.rgb.frame, cv2.COLOR_BGR2GRAY)
                depth_frame_scaled = cv2.convertScaleAbs(self.depth.frame, alpha=(255.0/65535.0))
                mixed_frame = ((1-gamma_depth)*grayscale_frame + gamma_depth*depth_frame_scaled).astype(np.uint8)

                concat_frame = np.concatenate((grayscale_frame, depth_frame_scaled), axis=1)

                # OpenCV imshow visualizes "BGR" image into "RGB"
                bgr_frame = cv2.cvtColor(copy.deepcopy(self.rgb.frame), cv2.COLOR_RGB2BGR)

                winname = "RGB Image + Depth Image"
                cv2.namedWindow(winname)
                cv2.moveWindow(winname, self.imshow_window_x, self.imshow_window_y)

                # cv2.imshow(winname, bgr_frame)
                cv2.imshow(winname, concat_frame)
                cv2.waitKey(1)
            else:
                # print("RGB Frame is None")
                pass

            # Falsify New Sensor Image Flags
            self.falsify_new_sensor_flags()

            # ROS Subscribe Rate
            # r.sleep()


# Main
def main():
    # ROS Node Initialization
    rospy.init_node("snu_module", anonymous=True)

    # ROS Class Initialization
    ros_embedded_module = ros_embedder(subscribe_rate=10)

    # Run Algorithm
    ros_embedded_module.run_algorithm()

    # Spin
    rospy.spin()


# Main Namespace
if __name__ == '__main__':
    main()































