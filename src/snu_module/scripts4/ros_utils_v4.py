"""
SNU Integrated Module v4.0
    - Classes for ROS Multimodal Sensors, with Synchronization (sync part from KIRO)
    - Message Wrapper for Publishing Message to ETRI Agent

"""
# Import Modules
import cv2
import os
import yaml
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
import threading

# Import ROS Messages
from osr_msgs.msg import Track, Tracks, BoundingBox
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg
from tf.transformations import quaternion_from_euler

# Import KIRO's Synchronized Subscriber
from snu_utils.sync_subscriber import SyncSubscriber


""" Sensor Parameter Class """


# Sensor Parameter Base Class
class sensor_params(object):
    def __init__(self, param_precision):
        # Set Parameter Precision
        self.param_precision = param_precision

        # Set Projection Matrix and its Pseudo-inverse Matrix
        self.projection_matrix = None
        self.pinv_projection_matrix = None

    # Update Parameters
    def update_params(self, param_argument):
        raise NotImplementedError()


# Sensor Parameter Class (Rostopic)
class sensor_params_rostopic(sensor_params):
    def __init__(self, param_precision=np.float32):
        super(sensor_params_rostopic, self).__init__(param_precision)

        """ Initialize Camera Parameter Matrices
        ----------------------------------------
        D: Distortion Matrix (5x1)
        K: Intrinsic Matrix (3x3)
        R: Rotation Matrix (3x3)
        P: Projection Matrix (3x4)
        ----------------------------------------
        """
        self.D, self.K, self.R, self.P = None, None, None, None

    def update_params(self, msg):
        self.D = msg.D.reshape((5, 1))  # Distortion Matrix
        self.K = msg.K.reshape((3, 3))  # Intrinsic Matrix
        self.R = msg.R.reshape((3, 3))  # Rotation Matrix
        self.P = msg.P.reshape((3, 4))  # Projection Matrix

        self.projection_matrix = self.P
        self.pinv_projection_matrix = np.linalg.pinv(self.P)


# Sensor Parameter Class (File)
class sensor_params_file(sensor_params):
    def __init__(self, param_precision=np.float32):
        super(sensor_params_file, self).__init__(param_precision)

        # Initialize Intrinsic-related Variables
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        self.w = None

        # Initialize Translation-related Variables
        self.x, self.y, self.z = None, None, None

        # Initialize Pan(yaw) / Tilt(pitch) / Roll Variables
        self.pan, self.tilt, self.roll = None, None, None

        # Set Camera Parameter Matrices
        self.intrinsic_matrix, self.extrinsic_matrix, self.rotation_matrix = None, None, None

    # Update Parameter Variables
    def update_params(self, param_array):
        # Intrinsic-related
        self.fx, self.fy, self.cx, self.cy = \
            param_array[0], param_array[1], param_array[2], param_array[3]
        self.w = param_array[4]

        # Translation-related
        self.x, self.y, self.z = param_array[5], param_array[6], param_array[7]

        # Pan / Tilt / Roll
        self.pan, self.tilt, self.roll = param_array[8], param_array[9], param_array[10]

        # Intrinsic Matrix < 3 x 4 >
        self.intrinsic_matrix = np.array([[self.fx, self.w, self.cx, 0],
                                          [0, self.fy, self.cy, 0],
                                          [0, 0, 1, 0]], dtype=self.param_precision)

        # Rotation Matrix
        self.rotation_matrix = self.convert_ptr_to_rotation()

        # Extrinsic Matrix < 4 x 4 >
        translation_vector = np.matmul(
            self.rotation_matrix,
            np.array([self.x, self.y, self.z], dtype=self.param_precision).reshape((3, 1))
        )
        self.extrinsic_matrix = np.block(
            [np.vstack((self.rotation_matrix, np.zeros((1, 3)))), np.append(translation_vector, 1).reshape(-1, 1)]
        )

        # Get Projection Matrix and its Pseudo-inverse
        self.projection_matrix = np.matmul(self.intrinsic_matrix, self.extrinsic_matrix)
        self.pinv_projection_matrix = np.linalg.pinv(self.projection_matrix)

    # Convert PTR to Rotation Matrix
    def convert_ptr_to_rotation(self):
        r11 = np.sin(self.pan) * np.cos(self.roll) - np.cos(self.pan) * np.sin(self.tilt) * np.sin(self.roll)
        r12 = -np.cos(self.pan) * np.cos(self.roll) - np.sin(self.pan) * np.sin(self.tilt) * np.sin(self.roll)
        r13 = np.cos(self.tilt) * np.sin(self.roll)
        r21 = np.sin(self.pan) * np.sin(self.roll) + np.sin(self.tilt) * np.cos(self.pan) * np.cos(self.roll)
        r22 = -np.cos(self.pan) * np.sin(self.roll) + np.sin(self.tilt) * np.sin(self.pan) * np.cos(self.roll)
        r23 = -np.cos(self.tilt) * np.cos(self.roll)
        r31 = np.cos(self.tilt) * np.cos(self.pan)
        r32 = np.cos(self.tilt) * np.sin(self.pan)
        r33 = np.sin(self.tilt)

        rotation_matrix = np.array([[r11, r12, r13],
                                    [r21, r22, r23],
                                    [r31, r32, r33]], dtype=self.param_precision)
        return rotation_matrix


# # Multimodal Sensor Data Managing Class
# class ros_sensor_

# Synchronized Subscriber (from KIRO, SNU Adaptation)
class snu_SyncSubscriber(SyncSubscriber):
    def __init__(self, enable_color=True, enable_disparity=True, enable_ir=True, enable_aligned_disparity=True, enable_nv1=True, enable_thermal=True,
                 enable_color_camerainfo=True, enable_depth_camerainfo=True, enable_ir_camerainfo=True, enable_pointcloud=True, enable_odometry=True):
        SyncSubscriber.__init__(self, enable_color=enable_color, enable_depth=enable_disparity, enable_ir=enable_ir,
                                enable_aligned_depth=enable_aligned_disparity,  enable_nv1=enable_nv1, enable_thermal=enable_thermal,
                                enable_color_camerainfo=enable_color_camerainfo, enable_depth_camerainfo=enable_depth_camerainfo, enable_ir_camerainfo=enable_ir_camerainfo,
                                enable_pointcloud=enable_pointcloud, enable_odometry=enable_odometry)

    def get_sync_data(self):
        self.lock_flag.acquire()
        result_sync_dict = {
            "is_synced": self.sync_flag, "sync_stamp": self.sync_stamp, "odometry": self.sync_odometry,
            "color_frame": self.sync_color, "disparity_frame": self.sync_depth, "aligned_disparity_frame": self.sync_aligned_depth,
            "infrared_frame": self.sync_ir, "thermal_frame": self.sync_thermal, "nightvision_frame": self.sync_nv1, "pointcloud": self.sync_pointcloud,
            "color_camerainfo": self.sync_color_camerainfo, "disparity_camerainfo": self.sync_depth_camerainfo, "infrared_camerainfo": self.sync_ir_camerainfo
        }
        self.lock_flag.release()
        return result_sync_dict




















