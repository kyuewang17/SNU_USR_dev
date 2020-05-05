#!/usr/bin/env python
"""
Subscribe ROS topics (for D435)

"""
import numpy as np

# ROS Modules
import rospy
import message_filters
from cv_bridge import CvBridge

# ROS Message Types
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from rospy.numpy_msg import numpy_msg

# Import Modules
import snu_utils.general_functions as gfuncs


# Get Computer Screen Geometry
screen_geometry_dict = gfuncs.get_screen_geometry()


# D435 Topic Dictionary
d435_topic_dict = {
    """
    To Be Added
    """
}


# Class for D435 Images
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


# Class for D435 Rostopic Subscriber
class d435_subscriber(object):
    def __init__(self, subscribe_rate=10):
        # Frame Count Index
        self.fidx = 0

        # Sensor Subscribe Rate
        self.subscribe_rate = subscribe_rate

        # [1] RGB Sensor Image
        self.rgb_msg = None
        self.rgb = ros_sensor_image()














def main():
    pass


if __name__ == '__main__':
    main()