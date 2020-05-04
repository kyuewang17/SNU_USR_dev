"""
SNU Integrated Module v3.0
    - Classes for ROS multimodal Sensors
    - Functions and Classes for managing seqstamp and timestamp
    - Message Wrapper for SNU Module Result to Publish Message to ETRI Module
"""
# Import Modules
import numpy as np

# ImportROS Message Modules
# import sensor_msgs.point_cloud2 as pc2

# Import ROS Messages
import rospy
from osr_msgs.msg import Track, Tracks, BoundingBox
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
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
        self.is_new_data = False


# Class for Image-like ROS Sensor Data
class ros_sensor_image(ros_sensor):
    def __init__(self, modal_type):
        super(ros_sensor_image, self).__init__(modal_type)

        # Image Frame
        self.frame = None

        # Camera Parameters
        self.cam_params = None

    # Update Data
    def update(self, frame, msg_header):
        # Increase Frame Index
        self.sensor_fidx += 1

        # Update Seqstamp and Timestamp
        self.seq, self.stamp = msg_header.seq, msg_header.stamp

        # Update New Data Flag
        self.is_new_data = True

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
# for LiDAR, do not convert to XYZ (XYZRGB) format instantaneously
# instead, make a in-class function to convert it by declaring the function
# (ROS subscribe speed issue)
class ros_sensor_lidar(ros_sensor):
    def __init__(self, modal_type="lidar"):
        super(ros_sensor_lidar, self).__init__(modal_type)

        # Point-cloud Message
        self.pc2_msg = None

        # XYZRGB 3D-cloud Data
        self.xyzrgb = None

    # Update LiDAR Data
    def update(self, pc2_msg, msg_header):
        # Increase Frame Index
        self.sensor_fidx += 1

        # Update Seqstamp and Timestamp
        self.seq, self.stamp = msg_header.seq, msg_header.stamp

        # Update New Data Flag
        self.is_new_data = True

        # Update LiDAR Point-cloud Message
        self.pc2_msg = pc2_msg

    # Convert Point-cloud ROS Message to XYZRGB 3D-cloud format
    def convert_pc2_to_xyzrgb(self):
        # """
        # https://github.com/anshulpaigwar/Attentional-PointNet/blob/master/tools/pcl_helper.py
        # """
        # points_list = []
        #
        # if self.is_new_data is True:
        #     for data in pc2.read_points(self.pc2_msg, skip_nans=True):
        #         points_list.append([data[0], data[1], data[2], data[3]])
        #
        #     xyzrgb = pcl.PointCloud_PointXYZRGB()
        #     xyzrgb.from_list(points_list)
        # else:
        #     print("[WARNING/ERROR] Point Cloud Data is Deprecated...!")
        #     xyzrgb = None
        #
        # # Converted XYZRGB 3-D Cloud Data
        # self.xyzrgb = xyzrgb
        raise NotImplementedError


# Function to check if input "stamp" is a proper stamp variable
def check_if_stamp(stamp):
    assert (stamp.__class__.__name__ == "Time"), "Input Argument Type must be <time>!"
    assert (hasattr(stamp, "secs") and hasattr(stamp, "nsecs")), \
        "Input Argument Does not have appropriate attributes"
    assert (stamp.secs is not None and stamp.nsecs is not None), "Stamp has no values for time"


# Return total time in 'seconds' for "stamp"
def stamp_to_secs(stamp):
    check_if_stamp(stamp)
    return stamp.secs + (stamp.nsecs * 1e-9)


# List Version of 'stamp_to_secs'
def stamp_list_to_secs_list(stamp_list):
    assert (type(stamp_list) == list), "Input Argument Type must be a <list>!"
    secs_list = []
    for stamp in stamp_list:
        secs_list.append(stamp_to_secs(stamp))
    return secs_list


# Timestamp Class
class timestamp(object):
    def __init__(self):
        self.secs, self.nsecs = None, None

    # Update Timestamp
    def update(self, stamp):
        if stamp is None:
            self.secs, self.nsecs = None, None
        else:
            if hasattr(stamp, 'secs') is True and hasattr(stamp, 'secs') is True:
                self.secs, self.nsecs = stamp.secs, stamp.secs
            else:
                assert 0, "stamp needs to have both following attributes: <secs>, <nsecs>"

    # Update Timestamp with {secs, nsecs}
    def _update(self, secs, nsecs):
        self.secs, self.nsecs = secs, nsecs

    # Get Time Information of the Timestamp Class
    def get_time(self):
        if self.secs is None and self.nsecs is None:
            retVal = None
        elif self.nsecs is None:
            retVal = self.secs
        else:
            retVal = self.secs + self.nsecs * 1e-9
        return retVal


# Seqstamp Class
class seqstamp(object):
    def __init__(self, modal):
        self.seq = None
        self.timestamp = timestamp()
        self.modal = modal

    # Update Seqstamp
    def update(self, seq, stamp):
        self.seq = seq
        self.timestamp.update(stamp)

    # Get Time Information of the Seqstamp Class
    def get_time(self):
        return self.timestamp.get_time()


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
