import rospy
import numpy as np
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge, CvBridgeError

import cv2

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField

import snu_utils.general_functions as gfuncs


# Get Computer Screen Geometry
screen_geometry_dict = gfuncs.get_screen_geometry()


# Initialize Global Variables
d435_rgb_image, d435_aligned_depth_image, d435_depth_image = None, None, None
d435_rgb_seq, d435_rgb_stamp = None, None
d435_depth_seq, d435_depth_stamp = None, None

velodyne_pointcloud = None
velodyne_pointcloud_seq, velodyne_pointcloud_stamp = None, None

d435_rgb_cam_P, d435_depth_cam_P = None, None

# CvBridge
bridge = CvBridge()


############################
# Callback Functions (ROS) #
############################
# [1] D435 RGB
def d435_rgb_callback(msg):
    global d435_rgb_image, d435_rgb_seq, d435_rgb_stamp

    if msg is None:
        print("message is None!")

    d435_rgb_seq, d435_rgb_stamp = msg.header.seq, msg.header.stamp
    d435_rgb_image = bridge.imgmsg_to_cv2(msg, "8UC3")
    # Wait for Message
    if d435_rgb_image is None:
        msg = rospy.wait_for_message("/osr/image_color", Image)
        d435_rgb_seq, d435_rgb_stamp = msg.header.seq, msg.header.stamp
        d435_rgb_image = bridge.imgmsg_to_cv2(msg, "8UC3")


# [2] D435 Depth
def d435_depth_callback(msg):
    global d435_depth_image, d435_depth_seq, d435_depth_stamp

    # Wait for Message
    if msg is None:
        pass

    d435_depth_seq, d435_depth_stamp = msg.header.seq, msg.header.stamp
    d435_depth_image = bridge.imgmsg_to_cv2(msg, "16UC1")


# [3] D435 Aligned Depth
def d435_aligned_depth_callback(msg):
    global d435_aligned_depth_image

    # Wait for Message
    if msg is None:
        pass

    d435_aligned_depth_image = bridge.imgmsg_to_cv2(msg, "16UC1")


# [4] LIDAR Pointcloud
def lidar_pointcloud_callback(msg):
    global velodyne_pointcloud, velodyne_pointcloud_seq, velodyne_pointcloud_stamp

    # Wait for Message
    if msg is None:
        pass

    velodyne_pointcloud_seq, velodyne_pointcloud_stamp = msg.header.seq, msg.header.stamp


# D435 RGB Camera Parameter Callback
def d435_rgb_cam_param_callback(msg):
    global d435_rgb_cam_P

    # Wait for Message
    if msg is None:
        pass

    d435_rgb_cam_P = msg.P.reshape((3, 4))


# D435 Depth Camera Parameter Callback
def d435_depth_cam_param_callback(msg):
    global d435_depth_cam_P

    # Wait for Message
    if msg is None:
        pass

    d435_depth_cam_P = msg.P.reshape((3, 4))


# ROS Initialize Node
rospy.init_node("snu_module", anonymous=True)

# Subscribers
rgb_sub = rospy.Subscriber("/osr/image_color", Image, d435_rgb_callback)
# depth_sub = rospy.Subscriber("/osr/image_depth", Image, d435_depth_callback)
# aligned_depth_sub = rospy.Subscriber("/osr/image_aligned_depth", Image, d435_aligned_depth_callback)
# lidar_pointcloud_sub = rospy.Subscriber("/osr/lidar_pointcloud", Image, lidar_pointcloud_callback)
# rgb_cam_param_sub = rospy.Subscriber("/osr/image_color_camerainfo", Image, d435_rgb_cam_param_callback)
# depth_cam_param_sub = rospy.Subscriber("/osr/image_depth_camerainfo", Image, d435_depth_cam_param_callback)

# Initialize Seq List
rgb_seq_list, depth_seq_list = [], []

# Frame Index Init
fidx = 0

# Set OpenCV Imshow Window Position
margin_factors = [1.05, 1.05]
width_offset = int(screen_geometry_dict["margin_length"]["left_pixels"] * margin_factors[0])
height_offset = int(screen_geometry_dict["margin_length"]["top_pixels"] * margin_factors[1])

# Loop
while not rospy.is_shutdown():
    fidx += 1
    if d435_rgb_image is not None:
        print("RGB is not None (fidx: %s)" % str(fidx))
        winname = "RGB Image"
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, width_offset, height_offset)
        cv2.imshow(winname, d435_rgb_image)
        cv2.waitKey(1)
    else:
        print("RGB is None (fidx: %s)" % str(fidx))







