#!/usr/bin/env python
"""
Object Recognition Module (SNU Integrated Module)
for Outdoor Surveillance Robots

    - Object Class Python Script for Individual Sensors

"""
import cv2
import numpy as np
import image_geometry
import pyquaternion
import matplotlib
import random
import time
import ros_numpy

from rospy.rostime import Time
from sensor_msgs.msg import CameraInfo


class BASE_CAMERA_PARAMS_OBJ(object):
    def __init__(self, param_precision):
        # Parameter Precision
        self.param_precision = param_precision

        # Initialize Projection Matrix and its Pseudo-inverse
        self.P = None
        self.Pinv = None

    def update_params(self, *args, **kwargs):
        raise NotImplementedError()


class ROS_CAMERA_PARAMS_OBJ(BASE_CAMERA_PARAMS_OBJ):
    def __init__(self, msg=None, param_precision=np.float32):
        super(ROS_CAMERA_PARAMS_OBJ, self).__init__(param_precision)

        """ Initialize Camera Parameter Matrices
        ----------------------------------------
        D: Distortion Matrix (5x1)
        K: Intrinsic Matrix (3x3)
        R: Rotation Matrix (3x3)
        P: Projection Matrix (3x4)
        ----------------------------------------
        """
        if msg is None:
            self.D, self.K, self.R, self.P = None, None, None, None
        else:
            self.D = np.asarray(msg.D).reshape((5, 1))  # Distortion Matrix
            self.K = np.asarray(msg.K).reshape((3, 3))  # Intrinsic Matrix
            self.R = np.asarray(msg.R).reshape((3, 3))  # Rotation Matrix
            self.P = np.asarray(msg.P).reshape((3, 4))  # Projection Matrix
            self.Pinv = np.linalg.pinv(self.P)

    def update_params(self, *args, **kwargs):
        raise NotImplementedError()

    # def update_params(self, msg):
    #     self.D = np.asarray(msg.D).reshape((5, 1))  # Distortion Matrix
    #     self.K = np.asarray(msg.K).reshape((3, 3))  # Intrinsic Matrix
    #     self.R = np.asarray(msg.R).reshape((3, 3))  # Rotation Matrix
    #     self.P = np.asarray(msg.P).reshape((3, 4))  # Projection Matrix
    #
    #     self.Pinv = np.linalg.pinv(self.P)


class FILE_CAMERA_PARAMS_OBJ(BASE_CAMERA_PARAMS_OBJ):
    """ Used for Static Camera (YAML-based Camera Parameter) """
    def __init__(self, param_precision=np.float32):
        super(FILE_CAMERA_PARAMS_OBJ, self).__init__(param_precision)

        #

        """ Private Attributes """
        #

    def update_params(self, *args, **kwargs):
        raise NotImplementedError()


class BASE_SENSOR_OBJ(object):
    def __init__(self, modal_type, timestamp, sensor_opts):
        if timestamp is not None:
            assert isinstance(timestamp, Time) or isinstance(timestamp, float)

        """ Sensor Options """
        if sensor_opts is not None:
            # Validity
            self.is_valid = sensor_opts["is_valid"]

            # ROS Topic Information
            self.imgmsg_to_cv2_encoding = sensor_opts["imgmsg_to_cv2_encoding"]
            self.rostopic_name = sensor_opts["rostopic_name"]
            self.camerainfo_rostopic_name = sensor_opts["camerainfo_rostopic_name"]
        else:
            self.is_valid, self.imgmsg_to_cv2_encoding, self.rostopic_name = \
                None, None, None
            self.camerainfo_rostopic_name = None

        # Camera Modal
        self.modal_type = modal_type

        # Data
        self.data = None

        # Timestamp
        self.timestamp = timestamp

        # Timestamp Difference
        self.d_timestamp = None

        # CameraInfo
        self.sensor_params = None

    def __repr__(self):
        return self.modal_type

    def get_time_difference(self, timestamp):
        if self.timestamp is None:
            return None
        else:
            assert isinstance(timestamp, Time) or isinstance(timestamp, float)
            if isinstance(timestamp, Time) is True:
                if isinstance(self.timestamp, Time) is True:
                    return (self.timestamp - timestamp).to_sec()
                else:
                    return self.timestamp - timestamp.to_time()
            else:
                if isinstance(self.timestamp, Time) is True:
                    return self.timestamp.to_time() - timestamp
                else:
                    return self.timestamp - timestamp

    """ Timestamp Comparison Operator """

    def __ge__(self, other):
        if isinstance(other, BASE_SENSOR_OBJ):
            t_diff = self.get_time_difference(timestamp=other.timestamp)
        elif isinstance(other, Time):
            t_diff = self.get_time_difference(timestamp=other)
        else:
            raise NotImplementedError()
        if t_diff is None:
            raise AssertionError()
        else:
            return True if t_diff >= 0 else False

    def __gt__(self, other):
        if isinstance(other, BASE_SENSOR_OBJ):
            t_diff = self.get_time_difference(timestamp=other.timestamp)
        elif isinstance(other, Time):
            t_diff = self.get_time_difference(timestamp=other)
        else:
            raise NotImplementedError()
        if t_diff is None:
            raise AssertionError()
        else:
            return True if t_diff > 0 else False

    def __eq__(self, other):
        if isinstance(other, BASE_SENSOR_OBJ):
            t_diff = self.get_time_difference(timestamp=other.timestamp)
        elif isinstance(other, Time):
            t_diff = self.get_time_difference(timestamp=other)
        else:
            raise NotImplementedError()
        if t_diff is None:
            raise AssertionError()
        else:
            return True if t_diff == 0 else False

    def __lt__(self, other):
        if isinstance(other, BASE_SENSOR_OBJ):
            t_diff = self.get_time_difference(timestamp=other.timestamp)
        elif isinstance(other, Time):
            t_diff = self.get_time_difference(timestamp=other)
        else:
            raise NotImplementedError()
        if t_diff is None:
            raise AssertionError()
        else:
            return True if t_diff < 0 else False

    def __le__(self, other):
        if isinstance(other, BASE_SENSOR_OBJ):
            t_diff = self.get_time_difference(timestamp=other.timestamp)
        elif isinstance(other, Time):
            t_diff = self.get_time_difference(timestamp=other)
        else:
            raise NotImplementedError()
        if t_diff is None:
            raise AssertionError()
        else:
            return True if t_diff <= 0 else False

    """ Timestamp Comparison Part Ended """

    def get_modal_type(self):
        return self.modal_type

    def get_data(self, *args, **kwargs):
        raise NotImplementedError()

    def get_timestamp(self):
        return self.timestamp

    def update_data(self, *args, **kwargs):
        raise NotImplementedError()

    def update_timestamp(self, timestamp):
        if self.d_timestamp is not None:
            self.d_timestamp = timestamp - self.timestamp
        else:
            self.d_timestamp = 0.0
        self.timestamp = timestamp

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def update_sensor_opts(self, sensor_opts):
        if sensor_opts is None:
            raise AssertionError()
        else:
            # Validity
            self.is_valid = sensor_opts["is_valid"]

            # ROS Topic Information
            self.imgmsg_to_cv2_encoding = sensor_opts["imgmsg_to_cv2_encoding"]
            self.rostopic_name = sensor_opts["rostopic_name"]
            self.camerainfo_rostopic_name = sensor_opts["camerainfo_rostopic_name"]

    def update_sensor_params(self, **kwargs):
        # Assertion
        if "mode" not in kwargs:
            raise AssertionError("Input Argument 'mode' is required...!")
        else:
            assert isinstance(kwargs["mode"], str), "Input Argument 'mode' must be a string type...!"
            if kwargs["mode"].upper() not in ["ROS", "FILE"]:
                raise AssertionError("Input Argument 'mode' must be either 'ROS' or 'FILE' (upper/lower case insensitive)...!")

        if kwargs["mode"].upper() == "ROS":
            if "msg" not in kwargs:
                raise AssertionError("Input Argument 'msg' is required when 'ROS' mode...!")
            else:
                assert isinstance(kwargs["msg"], CameraInfo), "Input Argument 'msg' must be a ROS 'CameraInfo' type...!"
            self.sensor_params = ROS_CAMERA_PARAMS_OBJ(msg=kwargs["msg"])
        else:
            raise NotImplementedError()
            # self.sensor_params = FILE_CAMERA_PARAMS_OBJ()


class IMAGE_SENSOR_BASE_OBJ(BASE_SENSOR_OBJ):
    def __init__(self, modal_type, timestamp, sensor_opts):
        super(IMAGE_SENSOR_BASE_OBJ, self).__init__(modal_type, timestamp, sensor_opts)

        # Initialize Frame Data
        self.frame = None
        delattr(self, "data")

        # Initialize Frame Shape
        self.WIDTH, self.HEIGHT = None, None

        """ Private Attributes """
        # Visualization Window Moving Flag
        self.__vis_window_moved = False

    def __add__(self, other, timestamp_concat_method="pre"):
        """
        Channel-wise Frame Concatenation Operation
        :param other: Same Class Object or an NumPy Image-like Array of Same Width(Column) and Height(Row)
        :return: Concatenated Frame
        """
        assert isinstance(other, IMAGE_SENSOR_BASE_OBJ) or isinstance(other, np.ndarray), \
            "Operation Error! The current operand type is {}...!".format(other.__class__.__name__)

        # Check for Width and Height and Assert if not matched
        if isinstance(other, IMAGE_SENSOR_BASE_OBJ):
            other_width, other_height = other.WIDTH, other.HEIGHT
        else:
            other_width, other_height = other.shape[1], other.shape[0]
        if self.WIDTH != other_width and self.HEIGHT != other_height:
            raise AssertionError("Height and Width does not Match!")
        elif self.WIDTH != other_width:
            raise AssertionError("Width does not Match!")
        elif self.HEIGHT != other_height:
            raise AssertionError("Height does not Match!")

        # Concatenate Frames Channel-wise
        if isinstance(other, IMAGE_SENSOR_BASE_OBJ):
            concat_frame = np.dstack((self.get_data(), other.get_data()))
        else:
            concat_frame = np.dstack((self.get_data(), other))

        # Get Concatenated Modal Type
        concat_modal_type = "{}+{}".format(self, other)

        # Timestamp Concatenation Method
        if timestamp_concat_method == "pre" or not isinstance(other, IMAGE_SENSOR_BASE_OBJ):
            concat_timestamp = self.get_timestamp()
        elif timestamp_concat_method == "post":
            concat_timestamp = other.get_timestamp()
        elif timestamp_concat_method == "avg":
            concat_timestamp = (self.get_timestamp() + other.get_timestamp()) / 2.0
        else:
            raise NotImplementedError()

        # Initialize Concatenated Image Sensor Object
        concat_obj = IMAGE_SENSOR_BASE_OBJ(
            sensor_opts=None, modal_type=concat_modal_type, timestamp=concat_timestamp)
        concat_obj.update_data(frame=concat_frame)

        # Return Concatenated Object
        return concat_obj

    def get_data(self):
        return self.__get_frame()

    def update_data(self, frame):
        self.__update_frame(frame=frame)

    def update(self, frame, timestamp):
        self.update_data(frame=frame)
        self.update_timestamp(timestamp=timestamp)

    def __get_frame(self):
        return self.frame

    def __update_frame(self, frame):
        self.frame = frame

    def get_normalized_frame(self, min_value=0.0, max_value=1.0):
        frame = self.get_data()
        if frame is not None:
            frame_max_value, frame_min_value = frame.max(), frame.min()
            minmax_normalized_frame = (frame - frame_min_value) / (frame_max_value - frame_min_value)
            normalized_frame = min_value + (max_value - min_value) * minmax_normalized_frame
        else:
            normalized_frame = None
        return normalized_frame

    def visualize(self):
        # Get Frame
        vis_frame = self.get_data()

        if vis_frame is not None:
            # Get Modal Type Name
            modal_type = "{}".format(self)

            # OpenCV Window Name
            winname = "[{}]".format(modal_type)

            # Make NamedWindow
            cv2.namedWindow(winname)

            # Move Window
            if self.__vis_window_moved is False:
                cv2.moveWindow(winname=winname, x=1000, y=500)
                self.__vis_window_moved = True

            # Show Image via Imshow
            if modal_type == "color":
                cv2.imshow(winname, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            else:
                cv2.imshow(winname, vis_frame)

            cv2.waitKey(1)


class COLOR_SENSOR_OBJ(IMAGE_SENSOR_BASE_OBJ):
    def __init__(self, color_opts):
        super(COLOR_SENSOR_OBJ, self).__init__(sensor_opts=color_opts, modal_type="color", timestamp=None)


class DEPTH_SENSOR_OBJ(IMAGE_SENSOR_BASE_OBJ):
    def __init__(self, depth_opts):
        super(DEPTH_SENSOR_OBJ, self).__init__(sensor_opts=depth_opts, modal_type="depth", timestamp=None)

        # Clipped Depth Flag
        self.is_clipped = False

        """ Private Attributes """
        # Clip Distance Options
        self.__clip_distance = {
            "min": depth_opts["clip_distance"]["min"],
            "max": depth_opts["clip_distance"]["max"],
        }
        self.__clip_value = depth_opts["clip_value"]

        # Raw Data(frame)
        self.__raw_depth_frame = self.frame

    def clip_depth_frame(self):
        # Clip Raw Depth Frame
        raw_depth_frame = self.get_data()
        clipped_depth_frame = np.where(
            (raw_depth_frame < self.__clip_distance["min"]) | (raw_depth_frame > self.__clip_distance["max"]),
            self.__clip_value, raw_depth_frame
        )

        # Replace Data(frame) with Clipped Frame
        self.frame = clipped_depth_frame
        self.is_clipped = True

    def update_data(self, frame, clip_mode=True):
        self.__update_frame(frame=frame)
        if clip_mode is True:
            self.clip_depth_frame()
        self.__raw_depth_frame = frame

    def update(self, frame, timestamp, clip_mode=True):
        self.update_data(frame=frame, clip_mode=clip_mode)
        self.update_timestamp(timestamp=timestamp)

    def __update_frame(self, frame):
        self.frame = frame
        self.is_clipped = False


class INFRARED_SENSOR_OBJ(IMAGE_SENSOR_BASE_OBJ):
    def __init__(self, infrared_opts):
        super(INFRARED_SENSOR_OBJ, self).__init__(sensor_opts=infrared_opts, modal_type="infrared", timestamp=None)


class THERMAL_SENSOR_OBJ(IMAGE_SENSOR_BASE_OBJ):
    def __init__(self, thermal_opts):
        super(THERMAL_SENSOR_OBJ, self).__init__(sensor_opts=thermal_opts, modal_type="thermal", timestamp=None)


class NIGHTVISION_SENSOR_OBJ(IMAGE_SENSOR_BASE_OBJ):
    def __init__(self, nightvision_opts):
        super(NIGHTVISION_SENSOR_OBJ, self).__init__(sensor_opts=nightvision_opts, modal_type="nightvision", timestamp=None)


class LIDAR_SENSOR_OBJ(BASE_SENSOR_OBJ):
    def __init__(self, lidar_opts):
        super(LIDAR_SENSOR_OBJ, self).__init__(sensor_opts=lidar_opts, modal_type="lidar", timestamp=None)

        # Initialize PointCloud Variables
        self.pc3 = {}
        delattr(self, "data")

        """ Private Attributes """
        self.__tf_updated = False
        self.__tf_transform_modal = None
        self.__tf_transform_info = {}

        # LiDAR PointCloud Message (ROS < PointCloud2 > Type)
        self.__raw_pc_msg = None

        # LiDAR Raw PointCloud Variable
        self.__raw_pc = None

        # Camera Model Variable
        self.__CAMERA_MODEL = image_geometry.PinholeCameraModel()

    def __repr__(self):
        if self.__tf_updated is True:
            return "LiDAR-[{}]".format(self.__tf_transform_info["modal"])
        else:
            return "LiDAR"

    def update_data(self, lidar_pc_msg, tf_transform=None, tf_modal="color"):
        # Update LiDAR Message
        self.__raw_pc_msg = lidar_pc_msg

        # Update Timestamp
        if lidar_pc_msg is not None:
            self.update_timestamp(timestamp=lidar_pc_msg.header.stamp)

        # Update TF Transform Info (for first time)
        if self.__tf_updated is False:
            self.__tf_updated = True
            self.__tf_transform_info["modal"] = tf_modal
            self.__tf_transform_modal = tf_modal

            # Rotation
            self.__tf_transform_info["R"] = pyquaternion.Quaternion(
                tf_transform.transform.rotation.w,
                tf_transform.transform.rotation.x,
                tf_transform.transform.rotation.y,
                tf_transform.transform.rotation.z,
            ).rotation_matrix

            # Translation
            self.__tf_transform_info["T"] = np.array([
                tf_transform.transform.translation.x,
                tf_transform.transform.translation.y,
                tf_transform.transform.translation.z
            ]).reshape(3, 1)

        # Transform Raw PointCloud Message to Numpy Array
        raw_pc = np.array(ros_numpy.numpify(lidar_pc_msg).tolist())
        self.__raw_pc = raw_pc
        if raw_pc is not None and self.__tf_updated is True:
            tf_R = self.__tf_transform_info["R"]
            tf_T = self.__tf_transform_info["T"]
            tf_pc = np.dot(raw_pc[:, 0:3], tf_R.T) + tf_T.T

            # Filer-out Points in Front of Camera
            inrange = np.where((tf_pc[:, 0] > -8) &
                               (tf_pc[:, 0] < 8) &
                               (tf_pc[:, 1] > -5) &
                               (tf_pc[:, 1] < 5) &
                               (tf_pc[:, 2] > -0) &
                               (tf_pc[:, 2] < 30))
            max_intensity = np.max(tf_pc[:, -1])
            tf_pc = tf_pc[inrange[0]]
            self.pc3["cloud"] = tf_pc

            # Straight Distance From Camera
            self.pc3["distance"] = np.sqrt(tf_pc[:, 0] * tf_pc[:, 0] + tf_pc[:, 1] * tf_pc[:, 1] + tf_pc[:, 2] * tf_pc[:, 2])

            # Color map for the points
            cmap = matplotlib.cm.get_cmap('jet')
            self.pc3["colors"] = cmap(tf_pc[:, -1] / max_intensity) * 255  # intensity color view

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def get_data(self, camerainfo_msg, random_sample_number=0, bbox_mode=False, bbox_coord=None):
        """
        - For efficiency, LiDAR do not project to UV 2D image when updating.
          Instead, project to image coordinates when retrieving data

        :param camerainfo_msg: CameraInfo ROS Message, where to project 3D PointCloud
        :param random_sample_number: Randomly Sample PointClouds (0: select all samples)
        :param bbox_mode: when False, project to entire image area
        :param bbox_coord: [Left Top Right Bottom] formatted bounding box
        :return:
            (1) uv_array (2-D Projected PointCloud)
            (2) pc_distances
            (3) pc_colors
        """
        # Assertion
        if self.pc3 == {}:
            raise AssertionError("XYZRGB PointCloud Data Missing...!")
        if bbox_mode is True and bbox_coord is None:
            raise AssertionError("BBOX Coordinate cannot be None...!")

        # Get PointCloud Data
        cloud, pc_distance, pc_colors = self.pc3["cloud"], self.pc3["distance"], self.pc3["colors"]

        # Get Projection Parameters from CameraInfo Message
        fx, fy, cx, cy, Tx, Ty = self.__get_project_params_from_camerainfo(camerainfo_msg=camerainfo_msg)

        # Project
        px = (fx * cloud[:, 0] + Tx) / cloud[:, 2] + cx
        py = (fy * cloud[:, 1] + Ty) / cloud[:, 2] + cy

        # Stack UV Image Coordinate Points
        uv = np.column_stack((px, py))
        if bbox_mode is False:
            inrange = np.where((uv[:, 0] >= bbox_coord[0]) & (uv[:, 1] >= bbox_coord[1]) &
                               (uv[:, 0] < bbox_coord[2]) & (uv[:, 1] < bbox_coord[3]))
        else:
            inrange = np.where((uv[:, 0] >= bbox_coord[0]) & (uv[:, 1] >= bbox_coord[1]) &
                               (uv[:, 0] < bbox_coord[2]) & (uv[:, 1] < bbox_coord[3]))
        uv_array = uv[inrange[0]].round().astype('int')
        pc_distances = pc_distance[inrange[0]]
        pc_colors = pc_colors[inrange[0]]

        # Random Sampling
        if random_sample_number > 0:
            rand_indices = sorted(random.sample(range(len(uv_array)), random_sample_number))
            uv_array = uv_array[rand_indices]
            pc_distances = pc_distances[rand_indices]
            pc_colors = pc_colors[rand_indices]

        return uv_array, pc_distances, pc_colors

    def update_data_by_sensor_obj(self, *args, **kwargs):
        raise NotImplementedError()

    def __get_project_params_from_camerainfo(self, camerainfo_msg):
        self.__CAMERA_MODEL.fromCameraInfo(msg=camerainfo_msg)

        # Get Camera Parameters
        fx, fy = self.__CAMERA_MODEL.fx(), self.__CAMERA_MODEL.fy()
        cx, cy = self.__CAMERA_MODEL.cx(), self.__CAMERA_MODEL.cy()
        Tx, Ty = self.__CAMERA_MODEL.Tx(), self.__CAMERA_MODEL.Ty()

        return fx, fy, cx, cy, Tx, Ty


if __name__ == "__main__":
    tt = IMAGE_SENSOR_BASE_OBJ(modal_type="rgb", timestamp=0.123)
    qq = COLOR_SENSOR_OBJ(color_opts=None)
    qq.update_timestamp(0.123)
    qq.update_timestamp(0.124)

    pass
