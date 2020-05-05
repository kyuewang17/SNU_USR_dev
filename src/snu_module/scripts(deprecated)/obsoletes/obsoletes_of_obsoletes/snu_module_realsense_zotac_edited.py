#!/usr/bin/env python
"""
SNU Detection and MOT Module v2.0 (Realsense [d-435] adapted version)

    Detection Module
        - Code written/modified by : [XYZ] (xyz@qwerty.ac.kr)

    Tracking Module
        - Code written/modified by : [Kyuewang Lee] (kyuewang5056@gmail.com)

    Code Environment
        - python 2.7
        - tensorflow == 1.5.0
        - CUDA 9.0
            -> with cuDNN 7.0.5
        - ROS-kinetics

        < Dependencies >
            - [scikit-learn], [scikit-image], [FilterPy]
            - [numpy], [numba], [scipy], [matplotlib], [opencv-python]

    Source Code all rights reserved

"""

# Local Coordinate [X, Y, Z]

# Import Modules
import os
import cv2
import datetime
import copy
import numpy as np
import detector__lighthead_rcnn as detector
import mot__multimodal as mmodaltracker
import action_recognition as ar
import mot_module as mot


# Parameter Struct
class STRUCT:
    def __init__(self):
        pass

# Import ROS Modules
import rospy
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge, CvBridgeError

# Import ROS Messages
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from osr_msgs.msg import Track, Tracks
from osr_msgs.msg import BoundingBox
from geometry_msgs.msg import Pose, Twist
from geometry_msgs.msg import Point, Quaternion
from geometry_msgs.msg import Vector3
from tf.transformations import quaternion_from_euler



# CvBridge
bridge = CvBridge()

color_img, depth_img, odom, extrinsic = None, None, None, None
color_img_stamp = None

# Camera Intrinsic Matrix (Color, Depth)
color_cam_K, depth_cam_K = None, None

# Camera Projection Matrix (Color, Depth)
color_cam_P, depth_cam_P = None, None


def color_cb(msg):
    global color_img, color_img_stamp
    color_img_stamp = msg.header.stamp
    color_img = bridge.imgmsg_to_cv2(msg, '8UC3')


def depth_cb(msg):
    global depth_img
    depth_img = bridge.imgmsg_to_cv2(msg, '16UC1')


def odom_cb(msg):
    global odom
    odom = msg


def color_cam_param_cb(msg):
    global color_cam_K, color_cam_P
    color_cam_K = msg.K.reshape((3, 3))
    color_cam_P = msg.P.reshape((3, 4))

    sub_color_cam_params.unregister()


def depth_cam_param_cb(msg):
    global depth_cam_K, depth_cam_P
    depth_cam_K = msg.K.reshape((3, 3))
    depth_cam_P = msg.P.reshape((3, 4))

    sub_depth_cam_params.unregister()


# Code Execution Timestamp
script_info = STRUCT
script_info.CODE_TIMESTAMP = datetime.datetime.now()
script_info.EXECUTE_FILENAME = os.path.basename(__file__)

################# Depth Camera Clip #################
# Clip to this Value
DEPTH_CLIPPED_VALUE = -1

# (in millimeters)
DEPTH_CLIP_MIN_DISTANCE = 1000
# DEPTH_CLIP_MIN_DISTANCE = 500
# DEPTH_CLIP_MAX_DISTANCE = 15000
DEPTH_CLIP_MAX_DISTANCE = 20000
#####################################################

################## SNU Module Options ##################
###### Detection Parameters ######
model_base_path = os.path.dirname(os.path.abspath(__file__))
print(model_base_path)
detector_name = "lighthead_rcnn"
model_path = model_base_path + "/" + detector_name + "/model/detector.ckpt"
gpu_id = '0'

# Push in to the parameter struct
detparams = STRUCT
detparams.model_path = model_path
detparams.gpu_id = gpu_id
##################################

######### MOT Parameters #########
# [Tracklet Candidate --> Tracklet] Association Age
trkc_to_trk_asso_age = 3

# Destroy Objects
unasso_trk_destroy = 3          # Destroy unassociated tracklets with this amount of continuous unassociation
unasso_trkc_destroy = trkc_to_trk_asso_age + 1      # Destroy unassociated tracklet candidates ""

# Association Threshold
######################
cost_thresh = 0.5

# Depth Histogram Bin Number
dhist_bin = 500

# Push in to the parameter struct
motparams = STRUCT
motparams.unasso_trk_destroy = unasso_trk_destroy
motparams.unasso_trkc_destroy = unasso_trkc_destroy
motparams.cost_thresh = cost_thresh
motparams.trkc_to_trk_asso_age = trkc_to_trk_asso_age
motparams.dhist_bin = dhist_bin
motparams.depth_min_distance = DEPTH_CLIP_MIN_DISTANCE
motparams.depth_max_distance = DEPTH_CLIP_MAX_DISTANCE

motparams.calib = None
##################################

###### Visualization Options ######
is_vis_detection = True
is_vis_tracking = True
is_vis_action_result = False

# openCV Font Options
CV_FONT = cv2.FONT_HERSHEY_PLAIN
###################################

########################################################


# Detector Function
def snu_detector(image, infer_func, inputs):
    # Start Timestamp for DETECTION
    DET_TS_START = datetime.datetime.now()
    # Activate Detection Module
    result_dict = detector.detector(image, infer_func, inputs)
    # Convert to Detection BBOXES
    curr_dets = detector.get_detection_bboxes(result_dict)
    # Stop Timestamp for DETECTION
    DET_TS_STOP = datetime.datetime.now()
    # Elapsed Time for the DETECTION MODULE (ms)
    DET_ELAPSED_TIME = (DET_TS_STOP - DET_TS_START).total_seconds() * 1000

    return curr_dets, DET_ELAPSED_TIME


# MMMOT Function
def snu_mmmot(color_image, depth_image, fidx, dets, motparams, trackers, tracker_cands, max_trk_id):
    # Start Timestamp for MultiModal Multi-Object Tracker
    MMMOT_TS_START = datetime.datetime.now()
    # MMMOT Module
    trackers, tracker_cands, max_id = mmodaltracker.tracker(color_image, depth_image, fidx, dets, motparams, max_trk_id, trackers, tracker_cands)  # mot__multimodal as mmodaltracker
    # STOP Timestamp for MultiModal Multi-Object Tracker
    MMMOT_TS_STOP = datetime.datetime.now()
    # Elapsed Time for the MMMOT Module (ms)
    MMMOT_ELAPSED_TIME = (MMMOT_TS_STOP - MMMOT_TS_START).total_seconds() * 1000

    return trackers, tracker_cands, MMMOT_ELAPSED_TIME, max_id


def snu_ar(color_img, trackers):
    # Start Timestamp for Action Recognition
    AR_TS_START = datetime.datetime.now()
    # AR module
    trackers = ar.svm_clf(color_img, trackers)
    # STOP Timestamp for Action Recognition
    AR_TS_STOP = datetime.datetime.now()
    # Elapsed Time for the AR Module (ms)
    AR_ELAPSED_TIME = (AR_TS_STOP - AR_TS_START).total_seconds() * 1000
    return trackers, AR_ELAPSED_TIME


# Visualize Detections (openCV version)
def visualize_detections(img, dets, line_width=2):
    for det in dets:
        det = det.astype(np.int32)

        # Draw Rectangle BBOX
        cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]),
                      (0, 0, 255), line_width)


# Visualize Tracklets (openCV version)
def visualize_tracklets(img, trks, colors, line_width=2):
    for trk in trks:
        zs, _ = mot.zx_to_bbox(trk.states[-1])
        zs = zs.astype(np.int32)

        # Tracklet BBOX Color
        trk_color = colors[(trk.id % 3), :] * 255

        # Draw Rectangle BBOX
        cv2.rectangle(img, (zs[0], zs[1]), (zs[2], zs[3]),
                      (trk_color[0], trk_color[1], trk_color[2]), line_width)

        # Tracking INFO Font Size
        font_size = 1.25
        pad_pxls = 2
        info_interval = 4

        # Visualize Tracklet ID
        trk_id_str = "id:" + str(trk.id) + ""
        (tw, th) = cv2.getTextSize(trk_id_str, CV_FONT, fontScale=font_size, thickness=2)[0]
        text_x = int((zs[0] + zs[2]) / 2.0 - tw / 2.0)
        text_y = int(zs[1] + th)
        box_coords = ((int(text_x - pad_pxls / 2.0), int(text_y - th - pad_pxls / 2.0)),
                      (int(text_x + tw + pad_pxls / 2.0), int(text_y + pad_pxls / 2.0)))
        cv2.rectangle(img, box_coords[0], box_coords[1], (trk_color[0], trk_color[1], trk_color[2]), cv2.FILLED)
        # cv2.putText receives "bottom-left" point of the text
        cv2.putText(img, trk_id_str, (text_x, text_y), CV_FONT, font_size,
                    (255 - trk_color[0], 255 - trk_color[1], 255 - trk_color[2]), thickness=2)

        # Visualize Smoothed Camera Coordinate Velocity
        if trk.cam_coord_estimated_state is not None:
            cam_coord_vel = trk.cam_coord_estimated_state[2:-1]
            cam_coord_vel_str = "(dX,dY,dZ)=(" + str(round(cam_coord_vel[0][0], 2)) + "," + \
                                str(round(cam_coord_vel[1][0], 2)) + "," + \
                                str(round(cam_coord_vel[2][0], 2)) + ")"
            (tw, th) = cv2.getTextSize(cam_coord_vel_str, CV_FONT, fontScale=font_size*0.8, thickness=2)[0]
            text_x = int((zs[0] + zs[2]) / 2.0 - tw / 2.0)
            text_y = box_coords[0][1] - info_interval
            box_coords = ((int(text_x - pad_pxls / 2.0), int(text_y - th - pad_pxls / 2.0)),
                          (int(text_x + tw + pad_pxls / 2.0), int(text_y + pad_pxls / 2.0)))
            cv2.rectangle(img, box_coords[0], box_coords[1], (trk_color[0], trk_color[1], trk_color[2]), cv2.FILLED)
            # cv2.putText receives "bottom-left" point of the text
            cv2.putText(img, cam_coord_vel_str, (text_x, text_y), CV_FONT, font_size*0.8,
                        (255 - trk_color[0], 255 - trk_color[1], 255 - trk_color[2]), thickness=2)

        # Visualize Smoothed Camera Coordinate Position
        if trk.cam_coord_estimated_state is not None:
            cam_coord_pos = trk.cam_coord_estimated_state[0:3]
            cam_coord_pos_str = "(X,Y,Z)=(" + str(round(cam_coord_pos[0][0], 2)) + "," + \
                                str(round(cam_coord_pos[1][0], 2)) + "," + \
                                str(round(cam_coord_pos[2][0], 2)) + ")"
            (tw, th) = cv2.getTextSize(cam_coord_pos_str, CV_FONT, fontScale=font_size * 0.8, thickness=2)[0]
            text_x = int((zs[0] + zs[2]) / 2.0 - tw / 2.0)
            text_y = box_coords[0][1] - info_interval
            box_coords = ((int(text_x - pad_pxls / 2.0), int(text_y - th - pad_pxls / 2.0)),
                          (int(text_x + tw + pad_pxls / 2.0), int(text_y + pad_pxls / 2.0)))
            cv2.rectangle(img, box_coords[0], box_coords[1], (trk_color[0], trk_color[1], trk_color[2]), cv2.FILLED)
            # cv2.putText receives "bottom-left" point of the text
            cv2.putText(img, cam_coord_pos_str, (text_x, text_y), CV_FONT, font_size * 0.8,
                        (255 - trk_color[0], 255 - trk_color[1], 255 - trk_color[2]), thickness=2)

        # Visualize Tracklet Depth (tentative, will be changed)
        if trk.depth is not None:
            trk_depth_str = "d=" + str(round(trk.depth, 3)) + "(m)"
            (tw, th) = cv2.getTextSize(trk_depth_str, CV_FONT, fontScale=1.2, thickness=2)[0]
            text_x = int((zs[0] + zs[2]) / 2.0 - tw / 2.0)
            text_y = int((zs[1] + zs[3]) / 2.0 - th / 2.0)

            # Put Depth Text (Tentative)
            cv2.putText(img, trk_depth_str, (text_x, text_y), CV_FONT, 1.2,
                        (255 - trk_color[0], 255 - trk_color[1], 255 - trk_color[2]), thickness=2)

        # Visualize Action Recognition Result
        if trk.pose is not None and is_vis_action_result is True:
            H = img.shape[0]
            W = img.shape[1]
            cv2.putText(img, str(int(trk.pose) + 1), (min(int(trk.x[0] + (trk.x[4] / 2)), W - 1), min(int(trk.x[1] + (trk.x[5] / 2)), H - 1)),
                        CV_FONT, 3, (255 - trk_color[0], 255 - trk_color[1], 255 - trk_color[2]), thickness=3)


# Main Function
def main():
    global color_img, depth_img, odom, is_working, color_img_stamp
    global color_cam_K, color_cam_P, depth_cam_K, depth_cam_P
    global sub_color_cam_params, sub_depth_cam_params

    rospy.init_node("snu_module", anonymous=True)
    # Subscribe Images(Sensor Data) and Odometry from KIRO
    sub_color_img = rospy.Subscriber("osr/image_color", Image, color_cb)
    sub_depth_img = rospy.Subscriber("osr/image_depth", Image, depth_cb)
    sub_odometry = rospy.Subscriber("robot_odom", Odometry, odom_cb)

    # Subscribe Only Once
    # sub_color_cam_params = rospy.Subscriber("osr/d435_color_camera_info", numpy_msg(CameraInfo), color_cam_param_cb)
    # sub_depth_cam_params = rospy.Subscriber("osr/d435_depth_camera_info", numpy_msg(CameraInfo), depth_cam_param_cb)
    sub_color_cam_params = rospy.Subscriber("camera/color/camera_info", numpy_msg(CameraInfo), color_cam_param_cb)
    sub_depth_cam_params = rospy.Subscriber("camera/depth/camera_info", numpy_msg(CameraInfo), depth_cam_param_cb)

    # Publisher
    pub_tracks = rospy.Publisher("/osr/tracks", Tracks, queue_size=1)
    pub_img = rospy.Publisher("/osr/snu_result_image", Image, queue_size=1)

    # Load Detection Model
    infer_func, inputs = detector.load_model(model_path, gpu_id)

    # Tracklet Color (Visualization)
    # (Later, generate colormap)
    trk_colors = np.random.rand(32, 3)

    # Initialize Tracklet and Tracklet Candidate Object
    trackers = []
    tracker_cands = []

    # Initialize Frame Index
    fidx = 0

    # Tracklet Maximum ID Storage
    max_trk_id = 0

    # Camera Calibration Matrix
    motparams.color_cam_K = color_cam_K
    motparams.color_cam_P = color_cam_P
    motparams.depth_cam_K = depth_cam_K
    motparams.depth_cam_P = depth_cam_P

    # CvBridge
    bridge2 = CvBridge()

    while not rospy.is_shutdown():
        #############################################
        if color_img is None or depth_img is None:
            continue
        in_color_img_stamp = copy.deepcopy(color_img_stamp)
        in_color_img = copy.deepcopy(color_img)
        # in_color_img = cv2.cvtColor(in_color_img, cv2.COLOR_BGR2RGB)
        # BRG to RGB

        # out_img = bridge2.cv2_to_imgmsg(in_color_img, "8UC3")
        # pub_img.publish(out_img)

        # Convert Depth Image Type (uint16 --> float32)
        depth_img = depth_img.astype(np.float32)

        # Clip Depth Image
        depth_img = np.where((depth_img < DEPTH_CLIP_MIN_DISTANCE) | (depth_img > DEPTH_CLIP_MAX_DISTANCE),
                             DEPTH_CLIPPED_VALUE, depth_img)

        # print(depth_img[240, 320:600])
        # print(depth_img)

        #############################################
        # Increase Frame Index
        fidx += 1

        # print("COLOR")
        # print(color_cam_P)
        # print("====")
        # print("DEPTH")
        # print(depth_cam_P)

        # DETECTION MODULE
        curr_dets, DET_TIME = snu_detector(in_color_img, infer_func, inputs)

        # MultiModal Tracking MODULE
        trackers, tracker_cands, MMMOT_TIME, max_trk_id = \
            snu_mmmot(in_color_img, depth_img, fidx, curr_dets, motparams, trackers, tracker_cands, max_trk_id)

        # Action Recognition MODULE
        # trackers, AR_TIME = snu_ar(in_color_img, trackers)

        # # Detection Visualization
        if is_vis_detection is True:
            visualize_detections(in_color_img, curr_dets, line_width=2)
        #
        # MMMOT Visualization
        if is_vis_tracking is True:
            visualize_tracklets(in_color_img, trackers, trk_colors, line_width=3)

        ###############################################
        # ROS Tracklet Feed-in (Publish to ETRI module)
        out_tracks = Tracks()
        out_tracks.header.stamp = in_color_img_stamp
        for _, tracker in enumerate(trackers):
            # Get Tracklet Information
            track_state = tracker.states[-1]
            if len(tracker.states) > 1:
                track_prev_state = tracker.states[-2]
            else:
                track_prev_state = np.zeros(6).reshape(6, 1)
            track_cam_coord_state = tracker.cam_coord_estimated_state

            # Initialize track
            track = Track()

            # Tracklet ID
            track.id = tracker.id

            # Object Type
            track.type = 1

            # [bbox_pose]
            track_bbox = BoundingBox()
            track_bbox.x = np.uint32(track_state[0][0])
            track_bbox.y = np.uint32(track_state[1][0])
            track_bbox.height = np.uint32(track_state[5][0])
            track_bbox.width = np.uint32(track_state[4][0])
            track.bbox_pose = track_bbox

            # [bbox_velocity]
            track_d_bbox = BoundingBox()
            track_d_bbox.x = np.uint32(track_state[2][0])  # dx
            track_d_bbox.y = np.uint32(track_state[3][0])  # dy
            track_d_bbox.height = np.uint32((track_state - track_prev_state)[5][0])    # d_height
            track_d_bbox.width = np.uint32((track_state - track_prev_state)[4][0])     # d_width
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

            # Odometry
            out_tracks.odom = odom

        # Publish Tracks, Odometry, Result Image
        pub_tracks.publish(out_tracks)
        pub_img.publish(bridge2.cv2_to_imgmsg(in_color_img, "8UC3"))

        # Speed Visualization
        # det_fps = "Detector Speed: " + str(round(1000 / DET_TIME, 2)) + " (fps)"
        # mmmot_fps = "Tracker Speed: " + str(round(1000 / MMMOT_TIME, 2)) + " (fps)"
        # # ar_fps = "AR Speed: " + str(round(1000 / AR_TIME, 2)) + " (fps)"
        # cv2.putText(in_color_img, det_fps, (10, 20), CV_FONT, 1.3, (255, 0, 255), thickness=2)
        # cv2.putText(in_color_img, mmmot_fps, (10, 50), CV_FONT, 1.3, (255, 0, 255), thickness=2)
        # # cv2.putText(color_image, ar_fps, (10, 80), CV_FONT, 1.3, (255, 0, 255), thickness=2)

        # # Visualization Window (using OpenCV-Python)
        if is_vis_detection is True or is_vis_tracking is True:
            winname = "SNU Result"
            cv2.namedWindow(winname)
            if fidx == 1:
                cv2.moveWindow(winname, 200, 200)
            cv2.imshow(winname, in_color_img)

        cv2.waitKey(1)


# Main Function
if __name__ == '__main__':
    main()
