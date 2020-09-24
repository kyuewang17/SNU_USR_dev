"""
SNU Integrated Module v4.5
    - Python Script regarding LiDAR-related Utility Functions and Classes

"""
import numpy as np
from ros.sensors import ros_sensor_image, ros_sensor_disparity


class lidar_window(object):
    def __init__(self, sensor_data, pc_uv, pc_distance, window_size):
        assert isinstance(sensor_data, ros_sensor_image) or isinstance(sensor_data, ros_sensor_disparity)

        # Get Modality of Sensor Data
        self.kernel_modal = "{}".format(sensor_data)

        # Get Center uv-Coordinates (Projected-LiDAR)
        self.c_u, self.c_v = pc_uv[0], pc_uv[1]

        # Get LiDAR Point Distance
        self.pc_distance = pc_distance

        # Initialize Kernel Size (with Same Width and Height)
        self.window_size = window_size

        # Get LiDAR Window Data
        self.data = self._get_window_data(
            frame=sensor_data.get_data()
        )

    def get_window_average_depth(self):
        return np.average(self.data)

    def _get_window_data(self, frame):
        """
        :param frame: 2-D ndarray (only 2-D ndarray is supported for now...!)
        """
        assert len(frame.shape) == 2

        # u-coordinate compensation (column)
        u_min = np.round(self.c_u - 0.5*self.window_size).astype(int)
        u_min = 0 if u_min < 0 else u_min
        u_min = frame.shape[1] - 1 if u_min >= frame.shape[1] else u_min
        u_max = np.round(self.c_u + 0.5*self.window_size).astype(int)
        u_max = 0 if u_max < 0 else u_max
        u_max = frame.shape[1] - 1 if u_max >= frame.shape[1] else u_max

        # v-coordinate compensation (row)
        v_min = np.round(self.c_v - 0.5*self.window_size).astype(int)
        v_min = 0 if v_min < 0 else v_min
        v_min = frame.shape[0] - 1 if v_min >= frame.shape[1] else v_min
        v_max = np.round(self.c_v + 0.5*self.window_size).astype(int)
        v_max = 0 if v_max < 0 else v_max
        v_max = frame.shape[0] - 1 if v_max >= frame.shape[0] else v_max

        # Initialize Kernel Patch
        # TODO: < FIX THIS >
        _lidar_patch = np.empty(shape=(v_max - v_min + 1, u_max - u_min + 1)).astype(float)
        _lidar_patch.fill(self.pc_distance)

        alpha = 0.05
        _frame_patch = alpha*frame[v_min:v_max+1, u_min:u_max+1]
        window_patch = _frame_patch / 1000.0 + (1-alpha)*_lidar_patch

        return window_patch


