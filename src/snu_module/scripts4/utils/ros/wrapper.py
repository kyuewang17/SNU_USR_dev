"""
SNU Integrated Module v4.0
    - Message Wrapper for Publishing Tracks Message to ETRI Agent

"""
import rospy
import numpy as np
from osr_msgs.msg import Tracks, Track, BoundingBox
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from tf.transformations import quaternion_from_euler


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
            track_prev_state = np.zeros(7).reshape((7, 1))
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
            track.posture = (tracker.pose if tracker.pose is not None else 0)
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


if __name__ == "__main__":
    pass
