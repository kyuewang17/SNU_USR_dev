"""
SNU Integrated Module v4.0
    - Sensor Object Class for ROS Multimodal Sensor Data
    - Sensor Frames
        - Camera Parameter
        - Calibration Information
    - Sensor PointCloud
        - Camera Parameter
        - Calibration Information
    - Sensor Parameter Class

"""
import numpy as np
from rospy.rostime import Time


class ros_sensor(object):
    def __init__(self, modal_type, stamp):
        # Modal Type
        self._modal_type = modal_type

        # Raw Data
        self._raw_data = None

        # Modal Timestamp
        self._timestamp = stamp

        # Modal Sensor Parameters
        self._sensor_params = None

    def __repr__(self):
        return self.get_modal_type()

    """ Comparison Operator w.r.t. Timestamp """
    def __ge__(self, other):
        if isinstance(other, ros_sensor):
            t_diff = (self._timestamp - other._timestamp).to_sec()
        elif isinstance(other, Time):
            t_diff = (self._timestamp - other).to_sec()
        else:
            raise NotImplementedError()
        return True if t_diff >= 0 else False

    def __gt__(self, other):
        if isinstance(other, ros_sensor):
            t_diff = (self._timestamp - other._timestamp).to_sec()
        elif isinstance(other, Time):
            t_diff = (self._timestamp - other).to_sec()
        else:
            raise NotImplementedError()
        return True if t_diff > 0 else False

    def __eq__(self, other):
        if isinstance(other, ros_sensor):
            t_diff = (self._timestamp - other._timestamp).to_sec()
        elif isinstance(other, Time):
            t_diff = (self._timestamp - other).to_sec()
        else:
            raise NotImplementedError()
        return True if t_diff == 0 else False

    def __lt__(self, other):
        if isinstance(other, ros_sensor):
            t_diff = (self._timestamp - other._timestamp).to_sec()
        elif isinstance(other, Time):
            t_diff = (self._timestamp - other).to_sec()
        else:
            raise NotImplementedError()
        return True if t_diff < 0 else False

    def __le__(self, other):
        if isinstance(other, ros_sensor):
            t_diff = (self._timestamp - other._timestamp).to_sec()
        elif isinstance(other, Time):
            t_diff = (self._timestamp - other).to_sec()
        else:
            raise NotImplementedError()
        return True if t_diff <= 0 else False
    """ Comparison Operator Part Ended """

    def get_time_difference(self, stamp):
        assert isinstance(stamp, Time), "Input Method Must be a type of {}".format(stamp.__class__.__name__)
        t_diff = (self._timestamp - stamp).to_sec()
        return t_diff

    def update_modal_type(self, modal_type):
        self._modal_type = modal_type

    def get_modal_type(self):
        return self._modal_type

    def update_raw_data(self, raw_data):
        self._raw_data = raw_data

    def get_raw_data(self):
        return self._raw_data

    def update_data(self, data, stamp):
        raise NotImplementedError()

    def get_data(self):
        raise NotImplementedError()

    def update_stamp(self, stamp):
        self._timestamp = stamp

    def get_stamp(self):
        return self._timestamp

    def update_sensor_params_rostopic(self, msg):
        if msg is not None:
            # Initialize Sensor Parameter Object (from rostopic)
            self.sensor_params = sensor_params_rostopic(param_precision=np.float32)

            # Update Parameters
            self.sensor_params.update_params(msg=msg)


class ros_sensor_image(ros_sensor):
    def __init__(self, modal_type, frame=None, stamp=None, predetermine_main_data="processed"):
        super(ros_sensor_image, self).__init__(modal_type=modal_type, stamp=stamp)

        assert predetermine_main_data in ["vanilla", "processed", "concat"], \
            "Pre-determined Main Data [{}] Undefined...!".format(predetermine_main_data)

        # Modal Frame
        self.__frame = frame

        # Processed Frame
        self.__processed_frame = None

        # Pre-determined Data Type
        self._predetermined_data = predetermine_main_data

        # Pre-determined Main Data
        if predetermine_main_data == "vanilla":
            self.frame = self.__frame
        elif predetermine_main_data == "processed":
            self.frame = self.__processed_frame
        else:
            self.frame = frame

    def __add__(self, other):
        """
        Channel-wise Frame Concatenation Operation
        :param other: Same Class Object or an NumPy Image-like Array of Same Width(Column) and Height(Row)
        :return: Concatenated Frame
        """
        assert isinstance(other, ros_sensor_image) or isinstance(other, np.ndarray), \
            "Operation Error! The current operand type is {}...!".format(type(other))

        if isinstance(other, ros_sensor_image):
            # Concatenate Channel-wise
            concat_frame = np.dstack((self.get_data(), other.get_data()))

            # Get Concatenated Modal Type
            concat_modal_type = "{}+{}".format(self, other)

            # Initialize Concatenated ROS Sensor Image Class
            concat_obj = ros_sensor_image(
                modal_type=concat_modal_type, frame=concat_frame, stamp=self.get_stamp()
            )

            # Return Concatenated ROS Sensor Image Class
            return concat_obj

    def get_predetermined_data_type(self):
        return self._predetermined_data

    def update_data(self, frame, stamp):
        pass

    def get_data(self):
        return self.frame

    def get_normalized_data(self, min_value=0.0, max_value=1.0):
        pass

    def get_z_normalized_data(self, mean, stdev):
        pass

























if __name__ == "__main__":
    pass
