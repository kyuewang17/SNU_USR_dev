"""
SNU Integrated Module v3.0
    - Classes for ROS multimodal Sensors
    - Functions and Classes for managing seqstamp and timestamp
    - Message Wrapper for SNU Module Result to Publish Message to ETRI Module
"""
# Import Modules
import cv2
import os
import yaml
import numpy as np
import rospy
from ros_numpy import point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError

# Import ROS Messages
from osr_msgs.msg import Track, Tracks, BoundingBox
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg
from tf.transformations import quaternion_from_euler


# Sensor Modal Reference List
__sensor_modals__ = ["color", "disparity", "thermal", "infrared", "nightvision", "lidar"]


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
    def __init__(self, param_precision):
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
    def __init__(self, param_precision):
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


# Base Class for ROS Sensor Data
class ros_sensor(object):
    def __init__(self, modal_type):
        # Frame Index
        self.sensor_fidx = 0

        # New Data Flag
        self.is_new_data = None

        # ROS Header Seqstamp and Timestamp
        self.seq, self.stamp = None, None

        # Sensor Modal Type
        assert (modal_type in __sensor_modals__), "Argument Modal Type is Unknown!"
        self.modal_type = modal_type

    # Update Data
    def update(self, data, msg_header):
        raise NotImplementedError

    # Switch Back New Sensor Data Flag
    def flush_new_data_flag(self):
        self.is_new_data = None


# Class for Image-like ROS Sensor Data
class ros_sensor_image(ros_sensor):
    def __init__(self, modal_type):
        super(ros_sensor_image, self).__init__(modal_type)

        # Image Frame
        self.frame = None

        # Processed Frame
        self.processed_frame = None

        # Camera Parameters
        self.sensor_params = None

    # Addition Defined as Channel-wise Concatenation
    def __add__(self, other):
        assert isinstance(other, ros_sensor_image), "Argument 'other' must be same type!"

        modal_type = self.modal_type + "+" + other.modal_type
        null_header = Header()
        null_header.stamp = rospy.Time.now()

        # Concatenate Channel-wise
        concat_frame = np.dstack((self.get_data(), other.get_data()))

        # Initialize Concatenated Sensor Data Object
        concat_sensor_data = ros_sensor_image(modal_type=modal_type)
        concat_sensor_data.update(frame=concat_frame, msg_header=null_header)

    # Empty Frame Data
    def empty_frame(self):
        self.frame = None

    # Update Processed Frame
    def update_processed_frame(self, processed_frame):
        self.processed_frame = processed_frame

    # Get Data (processed data first priority)
    def get_data(self):
        if self.processed_frame is not None:
            return self.processed_frame
        else:
            return self.frame

    # Get Normalize Data
    def get_normalized_data(self, min_value, max_value):
        frame = self.get_data()
        frame_max_value, frame_min_value = frame.max(), frame.min()
        z_normalized_frame = (frame - frame_min_value) / (frame_max_value - frame_min_value)
        normalized_frame = min_value + (max_value - min_value) * z_normalized_frame
        return normalized_frame

    # Update Data
    def update(self, frame, msg_header):
        # Increase Frame Index
        self.sensor_fidx += 1

        # Update Seqstamp and Timestamp
        self.seq, self.stamp = msg_header.seq, msg_header.stamp

        # Update New Data Flag
        if frame is not None:
            self.is_new_data = True
        else:
            self.is_new_data = False

        # Update Image Frame
        self.frame = frame

    # Update Sensor Parameters by File
    def update_sensor_params_file(self, opts):
        # Get Sensor Parameter File Path
        sensor_param_file_path = \
            os.path.join(opts.sensor_param_base_path, self.modal_type + ".yml")

        if os.path.exists(sensor_param_file_path) is True:
            if self.sensor_params is None:
                # Initialize Sensor Parameter Object (from file)
                self.sensor_params = sensor_params_file(param_precision=np.float32)

                # Open YML File
                with open(sensor_param_file_path, "r") as stream:
                    tmp = yaml.safe_load(stream)
                sensor_param = np.asarray(tmp[opts.agent_name]["camera_param"])

                # Update Sensor Parameters
                self.sensor_params.update_params(param_array=sensor_param)

                # Message
                sensor_param_msg = "[%s] modal sensor parameters loaded as file...!" % self.modal_type

            else:
                assert 0, "Sensor Parameter Collision! (both rostopic and file exists!)"
        else:
            # Message
            sensor_param_msg = \
                "[NOTE]: (%s) modal does not have parameters as a 'yml' file!" % self.modal_type

        # Print Message
        print(sensor_param_msg)

    # Update Sensor Parameters by Rostopic
    def update_sensor_params_rostopic(self, msg):
        if msg is not None and self.modal_type in ["color", "disparity"]:
            # Initialize Sensor Parameter Object (from rostopic)
            self.sensor_params = sensor_params_rostopic(param_precision=np.float32)

            # Update Parameters
            self.sensor_params.update_params(msg=msg)
        else:
            raise NotImplementedError("[Error] rostopic for modal (%s) camerainfo message..!" % self.modal_type)


# Class for LiDAR ROS Sensor Data
class ros_sensor_lidar(ros_sensor):
    def __init__(self, modal_type="lidar"):
        super(ros_sensor_lidar, self).__init__(modal_type)

        # XYZ 3D-cloud Data
        self.xyz_arr = None

        # Projected LiDAR
        self.frame = None

    # Empty 3D-cloud Data
    def empty_xyzrgb(self):
        self.xyz_arr = None

    # Get Data
    def get_data(self):
        return self.frame[:, :, 0] if self.frame is not None else None

    # Get Data in millimeters
    def get_data_in_mm(self):
        pass

    # # Update LiDAR Data
    # def update(self, xyz_arr, msg_header):
    #     # Increase Frame Index
    #     self.sensor_fidx += 1
    #
    #     # Update Seqstamp and Timestamp
    #     self.seq, self.stamp = msg_header.seq, msg_header.stamp
    #
    #     # Update New Data Flag
    #     if xyz_arr is not None:
    #         self.is_new_data = True
    #     else:
    #         self.is_new_data = False
    #
    #     # Update LiDAR XYZ Array
    #     self.xyz_arr = xyz_arr

    # Update LiDAR Data
    def update(self, frame, msg_header):
        # Increase Frame Index
        self.sensor_fidx += 1

        # Update Seqstamp and Timestamp
        self.seq, self.stamp = msg_header.seq, msg_header.stamp

        # Update New Data Flag
        if frame is not None:
            self.is_new_data = True
        else:
            self.is_new_data = False

        # Update LiDAR XYZ Array
        self.frame = frame

    # Update LiDAR Rectification Parameters by File
    def update_rectification_parameters(self, opts):
        pass


# Define ROS Multimodal Subscribe Module Class
class ros_multimodal_subscriber(object):
    def __init__(self, opts):
        # Options
        self.opts = opts

        # (1) Color Modality
        self.color_msg = None
        self.color = ros_sensor_image(modal_type="color")

        # (2) Disparity Modality
        self.disparity_msg = None
        self.disparity = ros_sensor_image(modal_type="disparity")

        # (3) Thermal Modality
        self.thermal_msg = None
        self.thermal = ros_sensor_image(modal_type="thermal")

        # (4) Infrared Modality
        self.infrared_msg = None
        self.infrared = ros_sensor_image(modal_type="infrared")

        # (5) NightVision Modality
        self.nightvision_msg = None
        self.nightvision = ros_sensor_image(modal_type="nightvision")

        # (6) LiDAR Modality
        self.lidar_msg = None
        self.lidar = ros_sensor_lidar()

        # Odometry Message Variable (Pass-Through)
        self.odometry_msg = None

        # CvBridges
        self.sub_bridge, self.pub_bridge = CvBridge(), CvBridge()

        # Sensor Subscribers
        self.color_sub = rospy.Subscriber(opts.sensors.color["rostopic_name"], Image, self.color_callback)
        self.disparity_sub = rospy.Subscriber(opts.sensors.disparity["rostopic_name"], Image, self.disparity_callback)
        self.thermal_sub = rospy.Subscriber(opts.sensors.thermal["rostopic_name"], Image, self.thermal_callback)
        self.infrared_sub = rospy.Subscriber(opts.sensors.infrared["rostopic_name"], Image, self.infrared_callback)
        self.nightvision_sub = rospy.Subscriber(opts.sensors.nightvision["rostopic_name"], Image, self.nightvision_callback)
        # self.lidar_sub = rospy.Subscriber(opts.sensors.lidar["rostopic_name"], PointCloud2, self.lidar_callback)
        self.lidar_sub = rospy.Subscriber(opts.sensors.lidar["rostopic_name"], Image, self.lidar_callback)

        # Camerainfo Subscribers
        self.color_camerainfo_sub = rospy.Subscriber(
            opts.sensors.color["camerainfo_rostopic_name"], numpy_msg(CameraInfo), self.color_camerainfo_callback
        )
        self.disparity_camerainfo_sub = rospy.Subscriber(
            opts.sensors.disparity["camerainfo_rostopic_name"], numpy_msg(CameraInfo), self.disparity_camerainfo_callback
        )

        # Subscribe Odometry
        self.odometry_sub = rospy.Subscriber(opts.sensors.odometry["rostopic_name"], Odometry, self.odometry_callback)

    # Image Message to OpenCV Image
    def imgmsg_to_cv2(self, img_msg, msg_encode_type):
        return self.sub_bridge.imgmsg_to_cv2(img_msg, msg_encode_type)

    # Color Callback Function
    def color_callback(self, msg):
        self.color_msg = msg

    # Color Modal Update Function
    def update_color_sensor_data(self, color_sensor_opts, null_timestamp):
        if self.color_msg is not None:
            # for BGR format
            if self.color_msg.encoding.__contains__("bgr") is True:
                # Convert BGR to RGB
                color_frame = cv2.cvtColor(
                    self.imgmsg_to_cv2(self.color_msg, color_sensor_opts["imgmsg_to_cv2_encoding"]),
                    cv2.COLOR_BGR2RGB
                )
            # for RGB format
            elif self.color_msg.encoding.__contains__("rgb") is True:
                color_frame = self.imgmsg_to_cv2(self.color_msg, color_sensor_opts["imgmsg_to_cv2_encoding"])
            # for GrayScale image
            elif self.color_msg.encoding.__contains__("mono") is True:
                color_frame = self.imgmsg_to_cv2(self.color_msg, "8UC1")
            else:
                assert 0, "Current Encoding Type is not Defined!"

            # Update Color Frame
            self.color.update(color_frame, self.color_msg.header)

            # Message back to None
            self.color_msg = None
        else:
            null_header = Header()
            null_header.stamp = null_timestamp
            self.color.update(None, null_header)

    # Disparity Callback Function
    def disparity_callback(self, msg):
        self.disparity_msg = msg

    # Disparity Modal Update Function
    def update_disparity_sensor_data(self, disparity_sensor_opts, null_timestamp):
        if self.disparity_msg is not None:
            self.disparity.update(
                self.imgmsg_to_cv2(self.disparity_msg, disparity_sensor_opts["imgmsg_to_cv2_encoding"]),
                self.disparity_msg.header
            )

            # Message back to None
            self.disparity_msg = None
        else:
            null_header = Header()
            null_header.stamp = null_timestamp
            self.disparity.update(None, null_header)

    # Thermal Callback Function
    def thermal_callback(self, msg):
        self.thermal_msg = msg

    # Thermal Modal Update Function
    def update_thermal_sensor_data(self, thermal_sensor_opts, null_timestamp):
        if self.thermal_msg is not None:
            self.thermal.update(
                self.imgmsg_to_cv2(self.thermal_msg, thermal_sensor_opts["imgmsg_to_cv2_encoding"]),
                self.thermal_msg.header
            )

            # Message back to None
            self.thermal_msg = None
        else:
            null_header = Header()
            null_header.stamp = null_timestamp
            self.thermal.update(None, null_header)

    # Infrared Callback Function
    def infrared_callback(self, msg):
        self.infrared_msg = msg

    # Infrared Modal Update Function
    def update_infrared_sensor_data(self, infrared_sensor_opts, null_timestamp):
        if self.infrared_msg is not None:
            self.infrared.update(
                self.imgmsg_to_cv2(self.infrared_msg, infrared_sensor_opts["imgmsg_to_cv2_encoding"]),
                self.infrared_msg.header
            )

            # Message back to None
            self.infrared_msg = None
        else:
            null_header = Header()
            null_header.stamp = null_timestamp
            self.infrared.update(None, null_header)

    # Nightvision Callback Function
    def nightvision_callback(self, msg):
        self.nightvision_msg = msg

    # Nightvision Modal Update Function
    def update_nightvision_sensor_data(self, nightvision_sensor_opts, null_timestamp):
        if self.nightvision_msg is not None:
            self.nightvision.update(
                self.imgmsg_to_cv2(self.nightvision_msg, nightvision_sensor_opts["imgmsg_to_cv2_encoding"]),
                self.nightvision_msg.header
            )

            # Message back to None
            self.nightvision_msg = None
        else:
            null_header = Header()
            null_header.stamp = null_timestamp
            self.nightvision.update(None, null_header)

    # LiDAR Callback Function
    def lidar_callback(self, msg):
        self.lidar_msg = msg

    # LiDAR Modal Update Function
    def update_lidar_sensor_data(self, lidar_sensor_opts, null_timestamp):
        if self.lidar_msg is not None:
            # self.lidar.update(
            #     pc2.pointcloud2_to_xyz_array(self.lidar_msg),
            #     self.lidar_msg.header
            # )
            self.lidar.update(
                self.imgmsg_to_cv2(self.lidar_msg, lidar_sensor_opts["imgmsg_to_cv2_encoding"]),
                self.lidar_msg.header
            )

            # Message back to None
            self.lidar_msg = None
        else:
            null_header = Header()
            null_header.stamp = null_timestamp
            self.lidar.update(None, null_header)

    # Color Camerainfo Callback Function
    def color_camerainfo_callback(self, msg):
        if self.color.sensor_params is None:
            self.color.update_sensor_params_rostopic(msg)

    # Disparity Camerainfo Callback Function
    def disparity_camerainfo_callback(self, msg):
        if self.disparity.sensor_params is None:
            self.disparity.update_sensor_params_rostopic(msg)

    # Odometry Callback Function
    def odometry_callback(self, msg):
        self.odometry_msg = msg

    # Update All Modals
    def update_all_modals(self, null_timestamp):
        self.update_color_sensor_data(
            color_sensor_opts=self.opts.sensors.color, null_timestamp=null_timestamp
        )
        self.update_disparity_sensor_data(
            disparity_sensor_opts=self.opts.sensors.disparity, null_timestamp=null_timestamp
        )
        self.update_thermal_sensor_data(
            thermal_sensor_opts=self.opts.sensors.thermal, null_timestamp=null_timestamp
        )
        self.update_infrared_sensor_data(
            infrared_sensor_opts=self.opts.sensors.infrared, null_timestamp=null_timestamp
        )
        self.update_nightvision_sensor_data(
            nightvision_sensor_opts=self.opts.sensors.nightvision, null_timestamp=null_timestamp
        )
        self.update_lidar_sensor_data(
            lidar_sensor_opts=self.opts.sensors.lidar, null_timestamp=null_timestamp
        )

    # Collect All Timestamps
    def collect_all_headers(self):
        header_dict = {
            "color": self.color_msg.header,
            "disparity": self.disparity_msg.header,
            "thermal": self.thermal_msg.header,
            "infrared": self.infrared_msg.header,
            "nightvision": self.nightvision_msg.header,
            "lidar": self.lidar_msg.header
        }
        return header_dict

    # Collect All Messages
    def collect_all_messages(self):
        msg_dict = {
            "color": self.color_msg,
            "disparity": self.disparity_msg,
            "thermal": self.thermal_msg,
            "infrared": self.infrared_msg,
            "nightvision": self.nightvision_msg,
            "lidar": self.lidar_msg
        }
        return msg_dict

    # Collect All Sensor Data
    def collect_all_sensors(self):
        sensor_data = {
            "color": self.color,
            "disparity": self.disparity,
            "thermal": self.thermal,
            "infrared": self.infrared,
            "nightvision": self.nightvision,
            "lidar": self.lidar
        }
        return sensor_data

    # Collect all Sensor Existence Flag
    def collect_all_sensor_flags(self):
        sensor_flags = {
            "color": self.color.is_new_data,
            "disparity": self.disparity.is_new_data,
            "thermal": self.thermal.is_new_data,
            "infrared": self.infrared.is_new_data,
            "nightvision": self.nightvision.is_new_data,
            "lidar": self.lidar.is_new_data

        }
        return sensor_flags

    # Collect all Sensor Parameters by File (yml file)
    def gather_all_sensor_parameters(self):
        self.color.update_sensor_params_file(opts=self.opts)
        self.disparity.update_sensor_params_file(opts=self.opts)
        self.thermal.update_sensor_params_file(opts=self.opts)
        self.infrared.update_sensor_params_file(opts=self.opts)
        self.nightvision.update_sensor_params_file(opts=self.opts)
        self.lidar.update_rectification_parameters(opts=self.opts)


# Function for Publishing SNU Module Result to ETRI Module
def wrap_tracks(trackers, odometry):
    out_tracks = Tracks()

    # For the header stamp, record current time
    out_tracks.header.stamp = rospy.Time.now()

    # Odometry Information Passes Through SNU Module
    if odometry is not None:
        out_tracks.odom = odometry

    # For each Tracklets,
    for _, tracker in enumerate(trackers):
        # Get Tracklet State
        track_state = tracker.states[-1]
        if len(tracker.states) > 1:
            track_prev_state = tracker.states[-2]
        else:
            # [x,y,dx,dy,w,h]
            track_prev_state = np.zeros(6).reshape((6, 1))
        track_cam_coord_state = tracker.c3

        # Initialize Track
        track = Track()

        # Tracklet ID
        # important: set the Tracklet ID to modulus of 256 since the type is < uint8 >
        track.id = tracker.id % 256

        # Tracklet Object Type (1: Person // 2: Car)
        track.type = tracker.label

        # Tracklet Action Class (Posture)
        # 1: Stand, 2: Sit, 3: Lie
        # (publish if only tracklet object type is person)
        if tracker.label == 1:
            track.posture = tracker.pose
        else:
            track.posture = 0

        # Bounding Box Position [bbox_pose]
        track_bbox = BoundingBox()
        track_bbox.x = np.uint32(track_state[0][0])
        track_bbox.y = np.uint32(track_state[1][0])
        track_bbox.height = np.uint32(track_state[6][0])
        track_bbox.width = np.uint32(track_state[5][0])
        track.bbox_pose = track_bbox

        # Bounding Box Velocity [bbox_velocity]
        track_d_bbox = BoundingBox()
        track_d_bbox.x = np.uint32(track_state[3][0])
        track_d_bbox.y = np.uint32(track_state[4][0])
        track_d_bbox.height = np.uint32((track_state - track_prev_state)[6][0])
        track_d_bbox.width = np.uint32((track_state - track_prev_state)[5][0])
        track.bbox_velocity = track_d_bbox

        # [pose]
        cam_coord_pose = Pose()
        cam_coord_position = Point()
        cam_coord_orientation = Quaternion()

        cam_coord_position.x = np.float64(track_cam_coord_state[0][0])
        cam_coord_position.y = np.float64(track_cam_coord_state[1][0])
        cam_coord_position.z = np.float64(track_cam_coord_state[2][0])

        # Convert to Quaternion
        q = quaternion_from_euler(tracker.roll, tracker.pitch, tracker.yaw)
        cam_coord_orientation.x = np.float64(q[0])
        cam_coord_orientation.y = np.float64(q[1])
        cam_coord_orientation.z = np.float64(q[2])
        cam_coord_orientation.w = np.float64(q[3])

        cam_coord_pose.position = cam_coord_position
        cam_coord_pose.orientation = cam_coord_orientation
        track.pose = cam_coord_pose

        # [twist]
        cam_coord_twist = Twist()
        cam_coord_linear = Vector3()
        cam_coord_angular = Vector3()

        cam_coord_linear.x = np.float64(track_cam_coord_state[3][0])
        cam_coord_linear.y = np.float64(track_cam_coord_state[4][0])
        cam_coord_linear.z = np.float64(track_cam_coord_state[5][0])

        cam_coord_angular.x = np.float64(0)
        cam_coord_angular.y = np.float64(0)
        cam_coord_angular.z = np.float64(0)

        cam_coord_twist.linear = cam_coord_linear
        cam_coord_twist.angular = cam_coord_angular
        track.twist = cam_coord_twist

        # Append to Tracks
        out_tracks.tracks.append(track)

    return out_tracks
