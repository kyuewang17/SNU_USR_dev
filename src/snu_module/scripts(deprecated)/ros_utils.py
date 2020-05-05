"""
SNU Integrated Module v2.05
  - ROS-related functions

"""
# Import Modules
import numpy as np

import snu_utils.general_functions as gfuncs

# Import ROS Messages
import rospy
from osr_msgs.msg import Track, Tracks, BoundingBox
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from tf.transformations import quaternion_from_euler


# Check if input "stamp" is a proper stamp variable
def check_if_stamp(stamp):
    assert (stamp.__class__.__name__ == "Time"), "Input Argument Type must be a <Time>!"
    assert (hasattr(stamp, "secs") and hasattr(stamp, "nsecs")), \
        "Input Argument Does not have appropriate attributes"
    assert (stamp.secs is not None and stamp.nsecs is not None), "Stamp has no values for time"


# Return total time in 'seconds' for "stamp"
def stamp_to_secs(stamp):
    check_if_stamp(stamp)

    return stamp.secs + (stamp.nsecs * 1e-9)


# Return Time-stamp Difference in secs
def stamp_difference_in_secs(stamp1, stamp2):
    s1, s2 = stamp_to_secs(stamp1), stamp_to_secs(stamp2)
    return np.abs(s1-s2)


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
        if self.modal in ['lidar1', 'lidar2']:
            # If modal is LIDAR Image, consider inevitable lagging and keep previous existing value
            if seq is not None:
                self.seq = seq
            if stamp is not None:
                self.timestamp = stamp
        else:
            self.seq = seq
            self.timestamp.update(stamp)

    # Get Time Information of the Seqstamp Class
    def get_time(self):
        return self.timestamp.get_time()


# Check Synchronous/Asynchronous Sensors
def check_stamps_sync(stamps_dict, init_stamps_dict, frame_rate):
    secs_list = stamp_list_to_secs_list(stamps_dict.values())
    init_secs_list = stamp_list_to_secs_list(init_stamps_dict.values())

    # Net Seconds
    net_secs_list = [i - j for i, j in zip(secs_list, init_secs_list)]

    # Threshold Time in Seconds
    dtime_thresh = (1.0 / frame_rate) * 0.5

    # Net Seconds Absolute Difference Matrix, zero-out matrix element(thresholding)
    net_secs_diff_matrix = np.abs(gfuncs.array_difference_matrix(np.asarray(net_secs_list)))
    ret_net_secs_diff_matrix = np.copy(net_secs_diff_matrix)
    net_secs_diff_matrix[net_secs_diff_matrix > dtime_thresh] = 0

    # Synchronous/Asynchronous modality list init
    synchronous_exclude_indices = []
    for row_idx in range(net_secs_diff_matrix.shape[0]):
        sensor_affinity = np.sum(net_secs_diff_matrix[row_idx, :])
        if sensor_affinity == 0:
            synchronous_exclude_indices.append(row_idx)
    synchronous_modality_list = gfuncs.exclude_from_list(stamps_dict.keys(), synchronous_exclude_indices)

    if len(synchronous_exclude_indices) == 0:
        is_all_synchronized = True
    else:
        is_all_synchronized = False

    return synchronous_modality_list, is_all_synchronized, ret_net_secs_diff_matrix


# Synchronize Data by Stamp (Deprecated)
def is_stamps_synchronized(stamps_dict, init_stamps_dict, frame_rate):
    secs_list = stamp_list_to_secs_list(stamps_dict.values())
    init_secs_list = stamp_list_to_secs_list(init_stamps_dict.values())

    net_secs_list = [i-j for i, j in zip(secs_list, init_secs_list)]

    max_time_difference = max(net_secs_list) - min(net_secs_list)
    if max_time_difference <= (1.0/frame_rate)*0.5:
        return True, net_secs_list, max_time_difference
    else:
        return False, net_secs_list, max_time_difference


# Function for Publishing SNU Module Result to ETRI Module
def wrap_tracks(trackers, odometry, rgb_timestamp):
    out_tracks = Tracks()

    # Need to Initialize Timestamp Independently
    # if rgb_timestamp.secs is not None and rgb_timestamp.nsecs is not None:
    #     out_tracks.header.stamp = rospy.Time(rgb_timestamp.secs, rgb_timestamp.nsecs)
    # else:
    #     out_tracks.header.stamp = rospy.Time()
    out_tracks.header.stamp = rospy.Time.now()

    if odometry is not None:
        out_tracks.odom = odometry

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

        # Tracklet ID (uint8)
        track.id = tracker.id % 256

        # Tracklet Object Type (1: Person // 2: Car)
        track.type = tracker.label

        # Tracklet Action Class (Posture)
        # 1: Stand, 2: Sit, 3: Lie
        # (publish if only person)
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
