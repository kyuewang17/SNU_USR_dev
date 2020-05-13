"""
SNU Integrated Module v3.0
    - Classes for ROS multimodal Sensors
    - Functions and Classes for managing seqstamp and timestamp
    - Message Wrapper for SNU Module Result to Publish Message to ETRI Module
"""
# Import Modules
import cv2
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
        self.cam_params = None

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

    # Update Camera Parameters
    def update_cam_params(self, msg):
        # For D435i Modalities (Color and Disparity), update camera parameters using rostopic messages
        if self.modal_type in ["color", "disparity"]:
            self.cam_params = {
                "D": msg.D.reshape((5, 1)),  # Distortion Matrix
                "K": msg.K.reshape((3, 3)),  # Intrinsic Matrix
                "R": msg.R.reshape((3, 3)),  # Rotation Matrix
                "P": msg.P.reshape((3, 4)),  # Projection Matrix
            }
        else:
            raise NotImplementedError


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
        self.color.update_cam_params(msg)

    # Disparity Camerainfo Callback Function
    def disparity_camerainfo_callback(self, msg):
        self.disparity.update_cam_params(msg)

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
        # Get Tracklet Information
        track_state = tracker.states[-1]
        if len(tracker.states) > 1:
            track_prev_state = tracker.states[-2]
        else:
            # [x,y,dx,dy,w,h]
            track_prev_state = np.zeros(6).reshape((6, 1))
        track_cam_coord_state = np.concatenate((tracker.cam_coord, tracker.cam_coord_vel))

        # Initialize Track
        track = Track()

        # Tracklet ID
        # important note: set the Tracklet ID to modulus of 256 since the type is < uint8 >
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
        track_bbox.height = np.uint32(track_state[5][0])
        track_bbox.width = np.uint32(track_state[4][0])
        track.bbox_pose = track_bbox

        # Bounding Box Velocity [bbox_velocity]
        track_d_bbox = BoundingBox()
        track_d_bbox.x = np.uint32(track_state[2][0])
        track_d_bbox.y = np.uint32(track_state[3][0])
        track_d_bbox.height = np.uint32((track_state - track_prev_state)[5][0])
        track_d_bbox.width = np.uint32((track_state - track_prev_state)[4][0])
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
