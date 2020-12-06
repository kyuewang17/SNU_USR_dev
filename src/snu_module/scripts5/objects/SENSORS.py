#!/usr/bin/env python
"""
- Object Class Python Script for Multimodal Sensor Environment (for Each Modal Camera)

"""
import cv2
import numpy as np
import time

from rospy.rostime import Time


class BASE_SENSOR_OBJ(object):
    def __init__(self, modal_type, timestamp):
        if timestamp is not None:
            assert isinstance(timestamp, Time) or isinstance(timestamp, float)

        # Camera Modal
        self.modal_type = modal_type

        # Data
        self.data = None

        # Timestamp
        self.timestamp = timestamp

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
        self.timestamp = timestamp

    def update(self, data, timestamp):
        self.update_data(data=data)
        self.update_timestamp(timestamp=timestamp)


class COLOR_SENSOR_OBJ(BASE_SENSOR_OBJ):
    def __init__(self):
        super(COLOR_SENSOR_OBJ, self).__init__(modal_type="color", timestamp=None)


class DEPTH_SENSOR_OBJ(BASE_SENSOR_OBJ):
    def __init__(self):
        super(DEPTH_SENSOR_OBJ, self).__init__(modal_type="depth", timestamp=None)


class INFRARED_SENSOR_OBJ(BASE_SENSOR_OBJ):
    def __init__(self):
        super(INFRARED_SENSOR_OBJ, self).__init__(modal_type="infrared", timestamp=None)


class THERMAL_SENSOR_OBJ(BASE_SENSOR_OBJ):
    def __init__(self):
        super(THERMAL_SENSOR_OBJ, self).__init__(modal_type="thermal", timestamp=None)


class NIGHTVISION_SENSOR_OBJ(BASE_SENSOR_OBJ):
    def __init__(self):
        super(NIGHTVISION_SENSOR_OBJ, self).__init__(modal_type="nightvision", timestamp=None)


class LIDAR_SENSOR_OBJ(BASE_SENSOR_OBJ):
    def __init__(self):
        super(LIDAR_SENSOR_OBJ, self).__init__(modal_type="lidar", timestamp=None)



if __name__ == "__main__":
    pass
