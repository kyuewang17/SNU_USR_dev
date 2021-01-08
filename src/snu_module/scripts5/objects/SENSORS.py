#!/usr/bin/env python
"""
Object Recognition Module (SNU Integrated Module)
for Outdoor Surveillance Robots

    - Object Class Python Script for Multimodal Sensor Environment (for Each Modal Camera)

"""
import cv2
import numpy as np
import time

from rospy.rostime import Time


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

    def get_data(self):
        return self.data

    def get_timestamp(self):
        return self.timestamp

    def update_data(self, data):
        self.data = data

    def update_timestamp(self, timestamp):
        if self.d_timestamp is not None:
            self.d_timestamp = timestamp - self.timestamp
        else:
            self.d_timestamp = 0.0
        self.timestamp = timestamp

    def update(self, data, timestamp):
        self.update_data(data=data)
        self.update_timestamp(timestamp=timestamp)

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


class IMAGE_SENSOR_BASE_OBJ(BASE_SENSOR_OBJ):
    def __init__(self, modal_type, timestamp, sensor_opts):
        super(IMAGE_SENSOR_BASE_OBJ, self).__init__(modal_type, timestamp, sensor_opts)

        # Replace "data" attribute with "frame"
        self.frame = self.data
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

    def update_data(self, frame):
        self.__update_frame(frame=frame)
        self.__raw_depth_frame = frame

    def update(self, frame, timestamp):
        self.update_data(frame=frame)
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
    def __init__(self, lidar_opts, lidar_pc_msg=None):
        super(LIDAR_SENSOR_OBJ, self).__init__(sensor_opts=lidar_opts, modal_type="lidar", timestamp=None)

        # LiDAR PointCloud Message (ROS < PointCloud2 > Type)
        self.raw_pc_msg = lidar_pc_msg









if __name__ == "__main__":
    tt = IMAGE_SENSOR_BASE_OBJ(modal_type="rgb", timestamp=0.123)
    qq = COLOR_SENSOR_OBJ(color_opts=None)
    qq.update_timestamp(0.123)
    qq.update_timestamp(0.124)

    pass
