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
from sync_subscriber import SyncSubscriber


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
            self._sensor_params = sensor_params_rostopic(param_precision=np.float32)

            # Update Parameters
            self._sensor_params.update_params(msg=msg)

    def get_sensor_params(self):
        return self._sensor_params


class ros_sensor_image(ros_sensor):
    def __init__(self, modal_type, frame=None, stamp=None):
        super(ros_sensor_image, self).__init__(modal_type=modal_type, stamp=stamp)

        # Modal Frame
        self.__frame = frame

    def __add__(self, other):
        """
        Channel-wise Frame Concatenation Operation
        :param other: Same Class Object or an NumPy Image-like Array of Same Width(Column) and Height(Row)
        :return: Concatenated Frame
        """
        assert isinstance(other, ros_sensor_image) or isinstance(other, np.ndarray), \
            "Operation Error! The current operand type is {}...!".format(other.__class__.__name__)

        # Concatenate Frames Channel-wise
        if isinstance(other, ros_sensor_image):
            concat_frame = np.dstack((self.get_data(), other.get_data()))
        else:
            concat_frame = np.dstack((self.get_data(), other))

        # Get Concatenated Modal Type
        concat_modal_type = "{}+{}".format(self, other)

        # Initialize Concatenated ROS Sensor Image Class
        concat_obj = ros_sensor_image(
            modal_type=concat_modal_type, frame=concat_frame, stamp=self.get_stamp()
        )

        # Return Concatenated ROS Sensor Image Class
        return concat_obj

    def update_data(self, frame, stamp):
        self.__frame = frame
        self.update_stamp(stamp=stamp)

    def get_data(self):
        return self.__frame

    def get_normalized_data(self, min_value=0.0, max_value=1.0):
        frame = self.get_data()
        frame_max_value, frame_min_value = frame.max(), frame.min()
        minmax_normalized_frame = (frame - frame_min_value) / (frame_max_value - frame_min_value)
        normalized_frame = min_value + (max_value - min_value) * minmax_normalized_frame
        return normalized_frame

    def get_z_normalized_data(self, stochastic_standard="channel"):
        assert stochastic_standard in ["channel", "all"], \
            "stochastic standard method {} is undefined...!".format(stochastic_standard)

        frame = self.get_data()

        # z-normalize frame
        if len(frame.shape) == 2:
            frame_mean, frame_stdev = frame.mean(), frame.std()
            z_frame = (frame - frame_mean) / frame_stdev
        elif len(frame.shape) == 3:
            if stochastic_standard == "channel":
                z_frame = np.zeros(shape=(frame.shape[0], frame.shape[1], frame.shape[2]))
                for channel_idx in range(len(frame.shape)):
                    channel_frame = frame[:, :, channel_idx]
                    frame_mean, frame_stdev = channel_frame.mean(), channel_frame.std()
                    z_frame[:, :, channel_idx] = (channel_frame - frame_mean) / frame_stdev
            else:
                frame_mean, frame_stdev = frame.mean(), frame.std()
                z_frame = (frame - frame_mean) / frame_stdev
        else:
            raise NotImplementedError()

        return z_frame


class ros_sensor_lidar(ros_sensor):
    def __init__(self, modal_type, lidar_data=None, stamp=None):
        super(ros_sensor_lidar, self).__init__(modal_type=modal_type, stamp=stamp)

        # TEMPORARY VARIABLE (will be changed)
        self.__lidar_data = lidar_data

    def update_data(self, data, stamp):
        raise NotImplementedError()

    def get_data(self):
        raise NotImplementedError()

    def project_to_sensor_data(self, sensor_data):
        assert isinstance(sensor_data, (ros_sensor_image, np.ndarray))


class sensor_params(object):
    def __init__(self, param_precision):
        # Parameter Precision
        self.param_precision = param_precision

        # Set Projection Matrix and its Pseudo-inverse
        self.projection_matrix = None
        self.pinv_projection_matrix = None

    def update_params(self, param_argument):
        raise NotImplementedError()


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


class snu_SyncSubscriber(SyncSubscriber):
    def __init__(self, ros_sync_switch_dict, options):
        super(snu_SyncSubscriber, self).__init__(ros_sync_switch_dict, options)

    def get_sync_data(self):
        self.lock_flag.acquire()
        if self.sync_flag is False:
            self.lock_flag.release()
            return None
        else:
            result_sync_frame_dict = {
                "color": self.sync_color, "disparity": self.sync_depth, "aligned_disparity": self.sync_aligned_depth,
                "thermal": self.sync_thermal, "infrared": self.sync_ir, "nightvision": self.sync_nv1
            }
            self.lock_flag.release()
            return self.sync_stamp, result_sync_frame_dict


if __name__ == "__main__":
    pass
