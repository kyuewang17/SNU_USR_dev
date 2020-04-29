#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import time
import threading
import multiprocessing

import cv2
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PKG = 'auro_calibration'
import roslib; roslib.load_manifest(PKG)
import rosbag
import rospy
import tf2_ros
import ros_numpy
import image_geometry
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_matrix
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

import yaml
import sensor_msgs.msg
import argparse

# Global variables
PAUSE = False
FIRST_TIME = True
KEY_LOCK = threading.Lock()
TF_BUFFER = None
TF_LISTENER = None
CV_BRIDGE = CvBridge()
COLOR_CAMERA_MODEL = image_geometry.PinholeCameraModel()
THERMAL_CAMERA_MODEL = image_geometry.PinholeCameraModel()

# Global paths
PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CALIB_PATH = 'calibration_data/lidar_camera_calibration'

def handle_keyboard():
    global KEY_LOCK, PAUSE
    key = raw_input('Press [ENTER] to pause and pick points\n')
    with KEY_LOCK: PAUSE = True

def start_keyboard_handler():
    keyboard_t = threading.Thread(target=handle_keyboard)
    keyboard_t.daemon = True
    keyboard_t.start()

def start_thread():
    project_t = threading.Thread(target=test)
    project_t.daemon = True
    project_t.start()

def test(points3D,img_color):
    fx=COLOR_CAMERA_MODEL.fx()
    fy=COLOR_CAMERA_MODEL.fy()
    cx=COLOR_CAMERA_MODEL.cx()
    cy=COLOR_CAMERA_MODEL.cy()
    Tx=COLOR_CAMERA_MODEL.Tx()
    Ty=COLOR_CAMERA_MODEL.Ty()
    px = (fx*points3D[:,0] + Tx) / points3D[:,2] + cx;
    py = (fy*points3D[:,1] + Ty) / points3D[:,2] + cy;
    points2D=np.column_stack((px,py))
    
    #points2D = np.asarray(points2D)
    inrange = np.where((points2D[:, 0] >= 0) &
                       (points2D[:, 1] >= 0) &
                       (points2D[:, 0] < img_color.shape[1]) &
                       (points2D[:, 1] < img_color.shape[0]))
    points2D = points2D[inrange[0]].round().astype('int')
    return points2D

def project_point_cloud(lidar_pointcloud, image_color,image_thermal, lidar_color_image_pub, lidar_thermal_image_pub):
    # Read image using CV bridge
    
    try:
        img_color = CV_BRIDGE.imgmsg_to_cv2(image_color, 'bgr8')
        img_thermal = CV_BRIDGE.imgmsg_to_cv2(image_thermal, 'bgr8')
    except CvBridgeError as e: 
        rospy.logerr(e)
        return

    # Transform the point cloud
    try:
        transform_rgb = TF_BUFFER.lookup_transform('rgb', 'velodyne_link_rgb', rospy.Time())
        transform_rgb_inv = TF_BUFFER.lookup_transform('velodyne_link_rgb', 'rgb', rospy.Time())
        transform_th = TF_BUFFER.lookup_transform('thermal', 'velodyne_link_thermal', rospy.Time())

        lidar_pointcloud = do_transform_cloud(lidar_pointcloud, transform_rgb)

    except tf2_ros.LookupException:
        pass
        
    points3D = ros_numpy.point_cloud2.pointcloud2_to_array(lidar_pointcloud)
    points3D = np.asarray(points3D.tolist())

    # Filter points in front of camera
    inrange = np.where((points3D[:, 2] > 0) &
                       (points3D[:, 2] < 25) &
                       (np.abs(points3D[:, 0]) < 10) &
                       (np.abs(points3D[:, 1]) < 10)&
                       (np.sqrt(points3D[:, 0]*points3D[:, 0]+points3D[:, 1]*points3D[:, 1]+points3D[:, 2]*points3D[:, 2])<25))
    max_intensity = np.max(points3D[:, -1])
    points3D = points3D[inrange[0]]

    pc_distance=np.sqrt(points3D[:, 0]*points3D[:, 0]+points3D[:, 1]*points3D[:, 1]+points3D[:, 2]*points3D[:, 2])

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('jet')
    #colors = cmap(points3D[:, -1] / max_intensity) * 255
    
    colors=pc_distance*255/25
    
    points2D = test(points3D,img_color)
    
    # Draw the projected 2D points
    for i in range(len(points2D)):
        cv2.circle(img_color, tuple(points2D[i]), 1, (colors[i],colors[i],colors[i]), -1)
        #cv2.circle(img_color, tuple(points2D[i]), 1, colors[i], -1)
    # Publish the projected points image
    try:
        lidar_color_image_pub.publish(CV_BRIDGE.cv2_to_imgmsg(img_color, "bgr8"))
        lidar_thermal_image_pub.publish(CV_BRIDGE.cv2_to_imgmsg(img_thermal, "bgr8"))
    except CvBridgeError as e: 
        rospy.logerr(e)


def thermal_camera_info_callback(thermal_camera_info):
    global THERMAL_CAMERA_MODEL
    rospy.loginfo('Setting up thermal camera model')
    #print('Yaml file :', thermal_camera_info)
    THERMAL_CAMERA_MODEL.fromCameraInfo(thermal_camera_info)

def color_camera_info_callback(color_camera_info):
    global COLOR_CAMERA_MODEL
    rospy.loginfo('Setting up color camera model')
    #print('Yaml file :', color_camera_info)
    COLOR_CAMERA_MODEL.fromCameraInfo(color_camera_info)

def callback(lidar_pointcloud, image_color,image_thermal, lidar_color_image_pub=None, lidar_thermal_image_pub=None):
    global FIRST_TIME, PAUSE, TF_BUFFER, TF_LISTENER

    # Setup the pinhole camera model
    if FIRST_TIME:
        FIRST_TIME = False
        
        # TF listener
        rospy.loginfo('Setting up static transform listener')
        TF_BUFFER = tf2_ros.Buffer()
        TF_LISTENER = tf2_ros.TransformListener(TF_BUFFER)
    
    project_point_cloud(lidar_pointcloud, image_color,image_thermal, lidar_color_image_pub, lidar_thermal_image_pub)
        


def listener(color_camera_info, thermal_camera_info,image_color,image_thermal,lidar_color_image=None,lidar_thermal_image=None):
    # Start node
    rospy.init_node('kiro_image_recv', anonymous=True)
    rospy.loginfo('color_camera_info topic: %s' % color_camera_info)
    rospy.loginfo('thermal_camera_info topic: %s' % thermal_camera_info)
    rospy.loginfo('color_image topic: %s' % image_color)
    rospy.loginfo('thermal_image topic: %s' % image_thermal)
    rospy.loginfo('lidar_pointcloud topic: %s' % lidar_pointcloud)
    rospy.loginfo('lidar_color_image topic: %s' % lidar_color_image)
    rospy.loginfo('lidar_thermal_image topic: %s' % lidar_thermal_image)

    color_camera_info_sub = message_filters.Subscriber(color_camera_info, CameraInfo)
    thermal_camera_info_sub = message_filters.Subscriber(thermal_camera_info, CameraInfo)
    image_color_sub = message_filters.Subscriber(image_color, Image)
    image_thermal_sub = message_filters.Subscriber(image_thermal, Image)
    lidar_pointcloud_sub = message_filters.Subscriber(lidar_pointcloud, PointCloud2)

    # Publish output topic
    lidar_color_image_pub = None
    lidar_thermal_image_pub = None
    if lidar_color_image: lidar_color_image_pub = rospy.Publisher(lidar_color_image, Image, queue_size=5)
    if lidar_thermal_image: lidar_thermal_image_pub = rospy.Publisher(lidar_thermal_image, Image, queue_size=5)
    color_camera_info_sub.registerCallback(color_camera_info_callback)
    thermal_camera_info_sub.registerCallback(thermal_camera_info_callback)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [lidar_pointcloud_sub,image_color_sub,image_thermal_sub], queue_size=5, slop=0.1)
    ats.registerCallback(callback, lidar_color_image_pub,lidar_thermal_image_pub)
     
    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':
    color_camera_info = rospy.get_param('color_camera_info_topic')
    thermal_camera_info = rospy.get_param('thermal_camera_info_topic')
    image_color = rospy.get_param('image_color_topic')
    image_thermal = rospy.get_param('image_thermal_topic')
    lidar_pointcloud = rospy.get_param('lidar_pointcloud_topic')
    lidar_color_image = rospy.get_param('lidar_color_image_topic')
    lidar_thermal_image = rospy.get_param('lidar_thermal_image_topic')

    # Start subscriber
    listener(color_camera_info, thermal_camera_info,image_color,image_thermal,lidar_color_image,lidar_thermal_image)
