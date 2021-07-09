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
import copy
import random
import cv2
import matplotlib
import pyquaternion
import ros_numpy
import utils.patch
import numpy as np
import image_geometry
from rospy.rostime import Time
from sync_subscriber import SyncSubscriber
from utils.profiling import Timer


class ros_sensor(object):
    def __init__(self, modal_type, stamp):
        # Modal Type
        self._modal_type = modal_type

        # Raw Data
        self._raw_data = None

        # Modal Timestamp
        self._timestamp = stamp

        # Initialize Camerainfo Message
        self._camerainfo_msg = None

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

    def get_data(self, *args, **kwargs):
        raise NotImplementedError()

    def update_stamp(self, stamp):
        self._timestamp = stamp

    def get_stamp(self):
        return self._timestamp

    def update_sensor_params_rostopic(self, msg):
        print(11)
        if msg is not None:
            # Save Camerainfo Message
            self._camerainfo_msg = msg

            # Initialize Sensor Parameter Object (from rostopic)
            self._sensor_params = sensor_params_rostopic(param_precision=np.float32)

            # Update Parameters
            self._sensor_params.update_params(msg=msg)

    def update_sensor_params_file_array(self, sensor_param_array):
        # Initialize Sensor Parameter Object (from file array)
        self._sensor_params = sensor_params_file_array(param_precision=np.float32)

        # Update Sensor Parameter Array
        self._sensor_params.update_params(param_array=sensor_param_array)

    def get_sensor_params(self):
        return self._sensor_params

    def get_camerainfo_msg(self):
        return self._camerainfo_msg


class ros_sensor_image(ros_sensor):
    def __init__(self, modal_type, frame=None, stamp=None):
        super(ros_sensor_image, self).__init__(modal_type=modal_type, stamp=stamp)

        # Modal Frame
        self._frame = frame

        # Initialize Frame Shape
        self.WIDTH, self.HEIGHT = None, None

        # Variable for Moving Visualization Window
        self.__vis_window_moved = False

    def __add__(self, other):
        """
        Channel-wise Frame Concatenation Operation
        :param other: Same Class Object or an NumPy Image-like Array of Same Width(Column) and Height(Row)
        :return: Concatenated Frame
        """
        assert isinstance(other, ros_sensor_image) or isinstance(other, ros_sensor_disparity) or isinstance(other, np.ndarray), \
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
        self._frame = frame
        self.update_stamp(stamp=stamp)

        if self.WIDTH is None and frame is not None:
            self.WIDTH = frame.shape[1]
        if self.HEIGHT is None and frame is not None:
            self.HEIGHT = frame.shape[0]

    def get_data(self, **kwargs):
        # Get Type Conversion
        type_conversion = kwargs.get("astype")

        if type_conversion is None:
            return self._frame
        else:
            if self.dtype() != type_conversion:
                return self._frame.astype(type_conversion)
            else:
                return self._frame.astype(self.dtype())

    def get_data_aux(self):
        return self._frame

    def astype(self, type):
        assert isinstance(self._frame, np.ndarray)
        self._frame.astype(type)

    def dtype(self):
        assert isinstance(self._frame, np.ndarray)
        return self._frame.dtype

    def type_minmax_normalization(self, **kwargs):
        # Get Type
        type_conversion = kwargs.get("astype")
        if type_conversion is None:
            # Get Keep Type
            keep_type = kwargs.get("keep_type", False)
            if keep_type is True:
                type_conversion = self.dtype()

        # Override Mode
        is_override = kwargs.get("override", False)

        # Convert
        frame_type_info = self.get_type_minmax()
        min_value, max_value = frame_type_info["min"], frame_type_info["max"]
        minmax_normalized_frame = (self._frame - min_value) / float(max_value - min_value)
        if type_conversion is not None:
            if is_override is False:
                return minmax_normalized_frame.astype(type_conversion)
            else:
                self._frame = minmax_normalized_frame.astype(type_conversion)
        else:
            if is_override is False:
                return minmax_normalized_frame
            else:
                self._frame = minmax_normalized_frame

    def get_type_minmax(self):
        type_iinfo = np.iinfo(self.dtype())
        return {"min": type_iinfo.min, "max": type_iinfo.max}

    def get_normalized_data(self, min_value=0.0, max_value=1.0):
        frame = self.get_data()
        if frame is not None:
            frame_max_value, frame_min_value = frame.max(), frame.min()
            minmax_normalized_frame = (frame - frame_min_value) / (frame_max_value - frame_min_value)
            normalized_frame = min_value + (max_value - min_value) * minmax_normalized_frame
        else:
            normalized_frame = None
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

    def get_patch(self, bbox, enlarge_factor=1.0):
        return utils.patch.get_patch(
            img=self.get_data(), bbox=bbox, enlarge_factor=enlarge_factor
        )

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

            # IMSHOW
            if modal_type.__contains__("color") is True:
                cv2.imshow(winname, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            else:
                cv2.imshow(winname, vis_frame)

            cv2.waitKey(1)


class ros_sensor_disparity(ros_sensor_image):
    def __init__(self, frame=None, stamp=None):
        super(ros_sensor_disparity, self).__init__(
            modal_type="disparity", frame=frame, stamp=stamp
        )

        self._processed_frame = None

    def process_data(self, disparity_sensor_opts):
        if self.get_data() is not None:
            frame = self.get_data().astype(np.float32)
            self._processed_frame = np.where(
                (frame < disparity_sensor_opts["clip_distance"]["min"]) |
                (frame > disparity_sensor_opts["clip_distance"]["max"]),
                disparity_sensor_opts["clip_value"], frame
            )

    def get_data(self, is_processed=False):
        if is_processed is False:
            return self.get_data_aux()
        else:
            if self._processed_frame is None:
                return self.get_data_aux()
            else:
                return self._processed_frame


class ros_sensor_lidar(ros_sensor):
    def __init__(self, lidar_pc_msg=None, stamp=None):
        super(ros_sensor_lidar, self).__init__(modal_type="lidar", stamp=stamp)

        # LiDAR PointCloud Message (ROS < PointCloud2 > Type)
        self.raw_pc_msg = lidar_pc_msg

        # Initialize Rotation and Translation Matrices between (Color/Thermal) and LiDAR Cameras
        self.R = {"color": None, "thermal": None}
        self.T = {"color": None, "thermal": None}

        # Initialize Projected Point Cloud Variable
        self.projected_cloud = {}

        # Define Camera Model Function Variable
        self.CAMERA_MODEL = image_geometry.PinholeCameraModel()

        # LiDAR Point Cloud XYZ, Distance, and Colors
        self.cloud = {}
        self.pc_distance, self.pc_colors = {}, {}

        # Tentative, uv-cloud
        self.uv_cloud = {}

    def __add__(self, other):
        assert isinstance(other, ros_sensor_image) or isinstance(other, ros_sensor_disparity)
        # Get Added Sensor Data Frame
        other_frame = other.get_data()

        # Initialize Projected LiDAR Sensor Data
        projected_sensor_data = ros_sensor_image(modal_type="{}+{}".format(other, "LiDAR"))

        # Project LiDAR Sensor Data to the Added Sensor Data
        tmp = self.project_xyz_to_uv_by_sensor_data(sensor_data=other)
        if tmp is not None:
            uv_arr, distances, colors = tmp[0], tmp[1], tmp[2]
            for idx in range(len(uv_arr)):
                uv_point = tuple(uv_arr[idx])
                cv2.circle(
                    img=other_frame, center=uv_point, radius=2, color=colors[idx], thickness=-1
                )

            # Update Projected LiDAR Sensor Data
            projected_sensor_data.update_data(frame=other_frame, stamp=self.get_stamp())

        else:
            projected_sensor_data = None

        return projected_sensor_data

    def get_data(self):
        raise NotImplementedError()

    def load_pc_xyz_data(self):
        assert self.projected_cloud != {}
        for modal, cloud in self.projected_cloud.items():
            # Filer-out Points in Front of Camera
            inrange = np.where((cloud[:, 0] > -8) &
                               (cloud[:, 0] < 8) &
                               (cloud[:, 1] > -5) &
                               (cloud[:, 1] < 5) &
                               (cloud[:, 2] > -0) &
                               (cloud[:, 2] < 30))
            max_intensity = np.max(cloud[:, -1])
            cloud = cloud[inrange[0]]
            self.cloud[modal] = cloud

            # Straight Distance From Camera
            self.pc_distance[modal] = \
                np.sqrt(cloud[:, 0] * cloud[:, 0] + cloud[:, 1] * cloud[:, 1] + cloud[:, 2] * cloud[:, 2])

            # Color map for the points (# intensity color view)
            cmap = matplotlib.cm.get_cmap('jet')
            self.pc_colors[modal] = cmap(cloud[:, -1] / max_intensity) * 255

    def update_data(self, lidar_pc_msg, tf_transform):
        # Update LiDAR Message
        self.raw_pc_msg = lidar_pc_msg

        # Update Stamp
        if lidar_pc_msg is not None:
            self.update_stamp(stamp=lidar_pc_msg.header.stamp)

        # # Update Rotation and Translation Matrices
        # if self.R__color is None and tf_transform is not None:
        #     self.R__color = pyquaternion.Quaternion(
        #         tf_transform.transform.rotation.w,
        #         tf_transform.transform.rotation.x,
        #         tf_transform.transform.rotation.y,
        #         tf_transform.transform.rotation.z,
        #     ).rotation_matrix
        # if self.T__color is None and tf_transform is not None:
        #     self.T__color = np.array([
        #         tf_transform.transform.translation.x,
        #         tf_transform.transform.translation.y,
        #         tf_transform.transform.translation.z
        #     ]).reshape(3, 1)

        # Update Rotation and Translation
        self.R["color"], self.T["color"] = \
            tf_transform.get_transform(src_sensor="lidar", dest_sensor="color")
        self.R["thermal"], self.T["thermal"] = \
            tf_transform.get_transform(src_sensor="lidar", dest_sensor="thermal")

        # Project Point Cloud
        pc = np.array(ros_numpy.numpify(self.raw_pc_msg).tolist())
        for modal in ["color", "thermal"]:
            R, T = self.R[modal], self.T[modal]
            self.projected_cloud[modal] = np.dot(pc[:, 0:3], R.T) + T.T

    # # Rescue for LiDAR
    # def update_RT_color(self, tf_transform=None, RT_color_params_base_path=None):
    #     if tf_transform is not None and RT_color_params_base_path is not None:
    #         raise AssertionError()
    #     elif tf_transform is None and RT_color_params_base_path is None:
    #         raise AssertionError()
    #     else:
    #         if tf_transform is not None:
    #             # Update Rotation and Translation Matrices
    #             if self.R__color is None:
    #                 self.R__color = pyquaternion.Quaternion(
    #                     tf_transform.transform.rotation.w,
    #                     tf_transform.transform.rotation.x,
    #                     tf_transform.transform.rotation.y,
    #                     tf_transform.transform.rotation.z,
    #                 ).rotation_matrix
    #             if self.T__color is None:
    #                 self.T__color = np.array([
    #                     tf_transform.transform.translation.x,
    #                     tf_transform.transform.translation.y,
    #                     tf_transform.transform.translation.z
    #                 ]).reshape(3, 1)
    #         else:
    #             import os
    #             R__color_file = os.path.join(RT_color_params_base_path, "R__color.npy")
    #             T__color_file = os.path.join(RT_color_params_base_path, "T__color.npy")
    #             if os.path.isfile(R__color_file) is False:
    #                 raise AssertionError()
    #             if os.path.isfile(T__color_file) is False:
    #                 raise AssertionError()
    #             self.R__color = np.load(R__color_file)
    #             self.T__color = np.load(T__color_file)

    # def force_update_data(self, pc_data_dict, stamp):
    #     self.uv_cloud = pc_data_dict["uv_cloud"]
    #     self.pc_colors = pc_data_dict["cloud_colors"]
    #     self.pc_distance = pc_data_dict["cloud_distance"]
    #     self.update_stamp(stamp=stamp)

    def project_xyz_to_uv_by_sensor_data(self, sensor_data, random_sample_number=0):
        """
        Project XYZ PointCloud Numpy Array Data using Input Sensor Data's CameraInfo
        """
        assert isinstance(sensor_data, ros_sensor_image)

        return self.project_xyz_to_uv(
            camerainfo_msg=sensor_data.get_camerainfo_msg(), modal=sensor_data._modal_type,
            frame_width=sensor_data.WIDTH, frame_height=sensor_data.HEIGHT,
            random_sample_number=random_sample_number
        )

    def project_xyz_to_uv(self, camerainfo_msg, modal, frame_width, frame_height, random_sample_number=0):
        if self.cloud is not None:
            # Update Camera Parameter to Pinhole Camera Model
            self.CAMERA_MODEL.fromCameraInfo(msg=camerainfo_msg)

            # Get Camera Parameters
            fx, fy = self.CAMERA_MODEL.fx(), self.CAMERA_MODEL.fy()
            cx, cy = self.CAMERA_MODEL.cx(), self.CAMERA_MODEL.cy()
            Tx, Ty = self.CAMERA_MODEL.Tx(), self.CAMERA_MODEL.Ty()

            px = (fx * self.cloud[modal][:, 0] + Tx) / self.cloud[modal][:, 2] + cx
            py = (fy * self.cloud[modal][:, 1] + Ty) / self.cloud[modal][:, 2] + cy

            # Stack UV Image Coordinate Points
            uv = np.column_stack((px, py))
            inrange = np.where((uv[:, 0] >= 0) & (uv[:, 1] >= 0) &
                               (uv[:, 0] < frame_width) & (uv[:, 1] < frame_height))
            uv_array = uv[inrange[0]].round().astype('int')
            pc_distances = self.pc_distance[modal][inrange[0]]
            pc_colors = self.pc_colors[modal][inrange[0]]

            if random_sample_number > 0:
                rand_indices = sorted(random.sample(range(len(uv_array)), random_sample_number))
                uv_array = uv_array[rand_indices]
                pc_distances = pc_distances[rand_indices]
                pc_colors = pc_colors[rand_indices]

            return uv_array, pc_distances, pc_colors

        else:
            return None

    def project_xyz_to_uv_inside_bbox(self, camerainfo_msg, modal, bbox, random_sample_number=0):
        if self.cloud is not None:
            # Update Camera Parameter to Pinhole Camera Model
            self.CAMERA_MODEL.fromCameraInfo(msg=camerainfo_msg)

            # Get Camera Parameters
            fx, fy = self.CAMERA_MODEL.fx(), self.CAMERA_MODEL.fy()
            cx, cy = self.CAMERA_MODEL.cx(), self.CAMERA_MODEL.cy()
            Tx, Ty = self.CAMERA_MODEL.Tx(), self.CAMERA_MODEL.Ty()

            px = (fx * self.cloud[modal][:, 0] + Tx) / self.cloud[modal][:, 2] + cx
            py = (fy * self.cloud[modal][:, 1] + Ty) / self.cloud[modal][:, 2] + cy

            # Stack UV Image Coordinate Points
            uv = np.column_stack((px, py))
            inrange = np.where((uv[:, 0] >= bbox[0]) & (uv[:, 1] >= bbox[1]) &
                               (uv[:, 0] < bbox[2]) & (uv[:, 1] < bbox[3]))
            uv_array = uv[inrange[0]].round().astype('int')
            pc_distances = self.pc_distance[modal][inrange[0]]
            pc_colors = self.pc_colors[modal][inrange[0]]

            if random_sample_number > 0:
                random_sample_number = min(random_sample_number, len(uv_array))
                rand_indices = sorted(random.sample(range(len(uv_array)), random_sample_number))
                uv_array = uv_array[rand_indices]
                pc_distances = pc_distances[rand_indices]
                pc_colors = pc_colors[rand_indices]

            return uv_array, pc_distances, pc_colors

        else:
            return None


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

    # def update_params(self, msg):
    #     self.D = msg.D.reshape((5, 1))  # Distortion Matrix
    #     self.K = msg.K.reshape((3, 3))  # Intrinsic Matrix
    #     self.R = msg.R.reshape((3, 3))  # Rotation Matrix
    #     self.P = msg.P.reshape((3, 4))  # Projection Matrix
    #
    #     self.projection_matrix = self.P
    #     self.pinv_projection_matrix = np.linalg.pinv(self.P)

    def update_params(self, msg):
        self.D = np.asarray(msg.D).reshape((5, 1))  # Distortion Matrix
        self.K = np.asarray(msg.K).reshape((3, 3))  # Intrinsic Matrix
        self.R = np.asarray(msg.R).reshape((3, 3))  # Rotation Matrix
        self.P = np.asarray(msg.P).reshape((3, 4))  # Projection Matrix

        self.projection_matrix = self.P
        self.pinv_projection_matrix = np.linalg.pinv(self.P)


class sensor_params_imseq(sensor_params_rostopic):
    def __init__(self, param_precision):
        super(sensor_params_imseq, self).__init__(param_precision)

    def update_params(self, npy_file_base_path):
        import os
        assert os.path.isdir(npy_file_base_path)
        file_list = os.listdir(npy_file_base_path)
        for file_name in file_list:
            # Check for any None *.npy files
            if file_name.split(".")[-1] != "npy":
                raise AssertionError()

            # Read npy file
            try:
                file_data = np.load(os.path.join(npy_file_base_path, file_name))
            except:
                file_data = None

            # Set Variables
            raw_file_name = file_name.split(".")[0]
            setattr(self, raw_file_name, file_data)

        self.projection_matrix = self.P
        self.pinv_projection_matrix = np.linalg.pinv(self.P) if self.P is not None else None


class sensor_params_file_array(sensor_params):
    def __init__(self, param_precision=np.float32):
        super(sensor_params_file_array, self).__init__(param_precision)

        # Initialize Intrinsic-related Variables
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        self.w = None

        # Initialize Translation-related Variables
        self.x, self.y, self.z = None, None, None

        # Initialize Pan(yaw) / Tilt(pitch) / Roll Variables
        self.pan, self.tilt, self.roll = None, None, None

        # Set Camera Parameter Matrices
        self.intrinsic_matrix, self.extrinsic_matrix, self.rotation_matrix = None, None, None
        self.translation_vector = None

        # Initalize World Center Coordinate
        self.world_center_coord = None

    # Update Parameter Variables
    def update_params(self, param_array):
        import math
        # Intrinsic-related
        self.fx, self.fy, self.cx, self.cy = \
            param_array[0], param_array[1], param_array[2], param_array[3]
        self.w = param_array[4]

        # Translation-related
        self.x, self.y, self.z = param_array[5], param_array[6], param_array[7]

        # Pan / Tilt / Roll
        self.pan = param_array[8] * (np.pi / 180.0)
        self.tilt = param_array[9] * (np.pi / 180.0)
        self.roll = param_array[10] * (np.pi / 180.0)
        # self.pan, self.tilt, self.roll = param_array[8], param_array[9], param_array[10]

        # # Intrinsic Matrix < 3 x 4 >
        # self.intrinsic_matrix = np.array([[self.fx, self.w, self.cx, 0],
        #                                   [0, self.fy, self.cy, 0],
        #                                   [0, 0, 1, 0]], dtype=self.param_precision)

        # Rotation Matrix
        self.rotation_matrix = self.convert_ptr_to_rotation()

        # Translation Vector
        translation_vector = -np.matmul(
            self.rotation_matrix,
            np.array([self.x, self.y, self.z], dtype=self.param_precision).reshape((3, 1))
        )
        self.translation_vector = translation_vector

        # Set World Center Coordinate
        self.world_center_coord = self.get_world_coord_center()

        # self.extrinsic_matrix = np.block(
        #     [np.vstack((self.rotation_matrix, np.zeros((1, 3)))), np.append(translation_vector, 1).reshape(-1, 1)]
        # )
        #
        # # Get Projection Matrix and its Pseudo-inverse
        # self.projection_matrix = np.matmul(self.intrinsic_matrix, self.extrinsic_matrix)
        # self.pinv_projection_matrix = np.linalg.pinv(self.projection_matrix)

    def get_camera_coords_from_image_coords(self, x, y, undistort=False):
        if undistort is False:
            u = (x - self.cx) / self.fx
            v = (y - self.cy) / self.fy
            return u, v
        else:
            x -= self.cx
            y -= self.cy
            x, y = self.__undistort(x, y)
            u, v = x / self.fx, y / self.fy
            return u, v

    def get_world_coords_from_camera_coords(self, u, v):
        return np.matmul(
            self.rotation_matrix.T,
            (np.array([u, v, 1.0]).reshape(3, 1)) - self.translation_vector
        )

    def get_world_coords_from_image_coords(self, x, y):
        u, v = self.get_camera_coords_from_image_coords(x=x, y=y)
        return self.get_world_coords_from_camera_coords(u=u, v=v)

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
        # rotation_matrix = np.array([[r11, r12, r13],
        #                             [r21, r22, r23],
        #                             [r31, r32, r33],
        #                             [0, 0, 0]], dtype=self.param_precision)
        return rotation_matrix

    def get_world_coord_center(self):
        return -np.matmul(self.rotation_matrix.T, self.translation_vector)

    def get_ground_plane_coord(self, x, y, norm_mode="pos"):
        assert norm_mode in ["pos", "vel"]
        if norm_mode == "pos":
            world_coord = self.get_world_coords_from_image_coords(x=x, y=y)
            x = self.world_center_coord[0, 0]
            y = self.world_center_coord[1, 0]
            z = self.world_center_coord[2, 0]
            k_G = z / (z - world_coord[2, 0])

            ground_coord = np.array([
                x + k_G*(world_coord[0, 0] - x),
                y + k_G*(world_coord[1, 0] - y),
                0.0
            ]).reshape(-1, 1)
            return ground_coord
        else:
            g1 = self.get_ground_plane_coord(x=x, y=y, norm_mode="pos")
            g2 = self.get_ground_plane_coord(x=0, y=0, norm_mode="pos")
            return g1-g2

    def __get_marked_k(self, u, v):
        pass

    def __get_ground_k(self, u, v):
        r13 = self.rotation_matrix[0, 2]
        r23 = self.rotation_matrix[1, 2]
        r33 = self.rotation_matrix[2, 2]
        t1 = self.translation_vector[0, 0]
        t2 = self.translation_vector[1, 0]
        t3 = self.translation_vector[2, 0]

        _numerator = t1*r13 + t2*r23 + t3*r33
        _denom = u*r13 + v*r23 + r33
        # _denom = u*r13 + v*r23

        return _numerator / _denom

    def __undistort(self, u, v, max_iter=100):
        err_thr = 2 * 0.01 / (self.fx + self.fy)

        # Newton Method
        rd = np.sqrt(u*u + v*v)
        ru = copy.copy(rd)

        # Iterate
        cnt = 0
        while cnt < max_iter:
            f = (1 + ru**2 + ru**4) * ru - ru
            fp = 1 + 3*(ru**2) + 5*(ru**4)
            ru = ru - f / fp
            if abs(f) < err_thr:
                break
            cnt += 1

        u += ru / rd
        v += ru / rd
        return u, v

    def get_ground_plane_coord_old(self, x, y, norm_mode="pos"):
        assert norm_mode in ["pos", "vel"]
        if norm_mode == "pos":
            # Normalize Coordinates
            u, v = (x - self.cx) / self.fx, (y - self.cy) / self.fy

            # Get Ground k value
            k_G = self.__get_ground_k(u=u, v=v)

            # Compute Ground Plane Coordinate
            ground_plane_coord = np.matmul(
                self.rotation_matrix.T,
                k_G * np.array([u, v, 1]).reshape(3, 1) - self.translation_vector
            )
            return ground_plane_coord
        else:
            g1 = self.get_ground_plane_coord(x=x, y=y, norm_mode="pos")
            g2 = self.get_ground_plane_coord(x=0, y=0, norm_mode="pos")
            return g1 - g2


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
