"""
(Script Name is Tentative, will be changed later)
SNU Integrated Module v4.0
    - Python Script about LiDAR-to-OtherSensor(Disparity) Fusion

"""
import numpy as np

from ros_utils_v4 import ros_sensor_image


class lidar_kernel(object):
    def __init__(self, sensor_data, pc_uv, pc_distance, kernel_size):
        assert isinstance(sensor_data, ros_sensor_image)

        # Get Modality of Sensor Data
        self.kernel_modal = "{}".format(sensor_data)

        # Get Center uv-Coordinates (Projected-LiDAR)
        self.c_u, self.c_v = pc_uv[0], pc_uv[1]

        # Get LiDAR Point Distance
        self.pc_distance = pc_distance

        # Initialize Kernel Size (with Same Width and Height)
        self.kernel_size = kernel_size

        # Get LiDAR Kernel Data
        self.data = self._get_kernel_data(
            frame=sensor_data.get_data(is_processed=True)
        )

        pass

    def get_kernel_average_depth(self):
        return np.average(self.data)

    def _get_kernel_data(self, frame):
        """
        :param frame: 2-D ndarray (only 2-D ndarray is supported for now...!)
        """
        assert len(frame.shape) == 2

        # u-coordinate compensation
        u_min = np.round(self.c_u - 0.5*self.kernel_size).astype(int)
        u_min = 0 if u_min < 0 else u_min
        u_max = np.round(self.c_u + 0.5*self.kernel_size).astype(int)
        u_max = frame.shape[1] - 1 if u_max >= frame.shape[1] else u_max

        # v-coordinate compensation
        v_min = np.round(self.c_v - 0.5*self.kernel_size).astype(int)
        v_min = 0 if v_min < 0 else v_min
        v_max = np.round(self.c_v + 0.5*self.kernel_size).astype(int)
        v_max = frame.shape[0] - 1 if v_max >= frame.shape[0] else v_max

        # Initialize Kernel Patch
        _lidar_patch = np.empty(shape=(v_max - v_min + 1, u_max - u_min + 1)).astype(float)
        _lidar_patch.fill(self.pc_distance)

        alpha = 0.05
        _frame_patch = alpha*frame[v_min:v_max+1, u_min:u_max+1]
        kernel_patch = _frame_patch + (1-alpha)*_lidar_patch

        # # Fill-in Kernel Patch
        # kernel_u_idx, kernel_v_idx = -1, -1
        # for u_idx in range(u_min, u_max+1):
        #     kernel_u_idx += 1
        #     for v_idx in range(v_min, v_max+1):
        #         # kernel_v_idx += 1
        #         # dist_arr = np.array([u_idx-self.c_u, v_idx-self.c_v])
        #         # _alpha = np.linalg.norm(dist_arr, 2)
        #         # alpha = _alpha / (np.sqrt(2)*self.kernel_size)
        #
        #         alpha = 0.05
        #
        #         # LiDAR Part
        #         _lidar_part = (1-alpha)*self.pc_distance
        #
        #         # Frame Part
        #         _frame_part = alpha*frame[v_idx, u_idx] / 1000.0
        #
        #         # Fill
        #         kernel_patch[kernel_u_idx, kernel_v_idx] = _lidar_part + _frame_part
        #     kernel_v_idx = -1

        return kernel_patch


class MULTIPLE_LIDAR_KERNELS(object):
    def __init__(self, lidar_kernels, sensor_patch):
        assert isinstance(lidar_kernels, list)

        # Initialize LiDAR Kernel List
        self.lidar_kernels = lidar_kernels

        # Initialize Sensor Patch
        self.patch = sensor_patch

    def replace_frame_with_kernels(self, frame):
        assert len(frame.shape) == 2

        u_list, v_list, depth_list = [], [], []
        for kernel_idx in range(len(self.lidar_kernels)):
            l_kernel = self.lidar_kernels[kernel_idx]
            assert isinstance(l_kernel, lidar_kernel)

            u_left = l_kernel.c_u - np.floor(l_kernel.data.shape[0] / 2)
            u_right = u_left + l_kernel.data.shape[0]
            v_up = l_kernel.c_v - np.floor(l_kernel.data.shape[1] / 2)
            v_bottom = v_up + l_kernel.data.shape[1]

            u_list.extend(list(range(u_left, u_right+1)))
            v_list.extend(list(range(v_up, v_bottom+1)))

            depth_list.append(l_kernel.pc_distance)




































if __name__ == "__main__":
    pass