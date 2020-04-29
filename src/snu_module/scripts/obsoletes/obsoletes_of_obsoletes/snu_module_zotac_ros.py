#!/usr/bin/env python
"""
SNU Detection and MOT Module v1.9

    - Realsense [D435i] Camera + ROS adaptation version

    - Detection Module
        - Code written/modified by
            - MIPAL, SNU (www.mipal.snu.ac.kr)

        - Contacts
            - (1) yoojy31@snu.ac.kr
            - (2) hojun815@snu.ac.kr
            - (3) jis3613@snu.ac.kr

    - Tracking Module
        - Code written/modified by
            - PIL, SNU (www.pil.snu.ac.kr)

        - Contacts
            - (1) kyuewang5056@gmail.com
            - (2) umdaeho1@gmail.com

    - Code Environment
        - python==2.7
        - tensorflow-gpu==1.5.0
        - CUDA 9.0 + cuDNN 7.0.5
        - ROS-kinetics

    - virtualenv dependencies (not all are listed necessarily)
        - [scikit-learn], [scikit-image], [scipy]
        - [FilterPy]
        - [numpy], [numba], [matplotlib]
        - [opencv-python]
        - [IPython], [easydict]

        - OPTIONAL DEPENDENCIES
            - [pykitti] (to test code for KITTI dataset)
            - [pyrealsense2] (to directly use D435 realsense camera input)

    * Source Code All Rights Reserved *
"""

# Import Modules
import os
import cv2
import datetime
import numpy as np
import copy

# Import function modules
import detector__lighthead_rcnn as detector
import mot__multimodal as mmodaltracker
import mot_module as mot
import action_recognition as ar

# Import ROS Modules
import rospy
from cv_bridge import CvBridge, CvBridgeError
import message_filters

# Import ROS Message Modules
from std_msgs.msg import String
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from osr_msgs.msg import BoundingBox, Track, Tracks


# Parameter Struct
class STRUCT:
    def __init__(self):
        pass


########## Realsense (D435) Options ##########
DEPTH_CLIPPED_VALUE = -1
DEPTH_CLIP_DISTANCE = 15    # (in meters)
##############################################

########## SNU Module Options ##########
#### Detection Parameters ####
model_base_path = os.path.dirname(os.path.abspath(__file__))
print(model_base_path)
detector_name = "lighthead_rcnn"
model_path = model_base_path + "/" + detector_name + "/model/detector.ckpt"
gpu_id = '0'

# Push in to the parameter struct
detparams = STRUCT
detparams.model_path = model_path
detparams.gpu_id = gpu_id
########################################

#### MOT Parameters ####
# < Tracklet Candidate --> Tracklet > Association Age
trkc_to_trk_asso_age = 3

# Destroy unassociated tracklets with this amount of continuous unassociation
unasso_trk_destroy = 3

# Destroy unassociated tracklet candidates with this amount of continuous unassociation
unasso_trkc_destroy = trkc_to_trk_asso_age + 1

# Association Threshold
cost_thresh = 0.03

# Depth Histogram Bin Number
dhist_bin = 500

# Push in to the parameter struct
motparams = STRUCT
motparams.unasso_trk_destroy = unasso_trk_destroy
motparams.unasso_trkc_destroy = unasso_trkc_destroy
motparams.cost_thresh = cost_thresh
motparams.trkc_to_trk_asso_age = trkc_to_trk_asso_age
motparams.dhist_bin = dhist_bin
motparams.DEPTH_CLIP_DIST = DEPTH_CLIP_DISTANCE
motparams.calib = None

# # Camera Calibration Matrices
# extrinsics = STRUCT
# extrinsics.
########################

###### Visualization Options ######
is_vis_detection = True
is_vis_tracking = True
is_vis_action_result = False

# openCV Font Options
CV_FONT = cv2.FONT_HERSHEY_PLAIN
###################################


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

        # Visualize Tracklet ID
        pad_pxls = -10
        trk_id_str = "id:" + str(trk.id) + ""
        (tw, th) = cv2.getTextSize(trk_id_str, CV_FONT, fontScale=1.2, thickness=2)[0]
        text_x = int((zs[0] + zs[2]) / 2.0 - tw / 2.0)
        text_y = int((10 * zs[1] - zs[3]) / 9.0 - th / 2.0)
        box_coords = ((text_x, text_y), (text_x + tw - pad_pxls, text_y - th + pad_pxls))
        cv2.rectangle(img, box_coords[0], box_coords[1], (trk_color[0], trk_color[1], trk_color[2]), cv2.FILLED)
        cv2.putText(img, trk_id_str, (text_x, text_y), CV_FONT, 1.6,
                    (255 - trk_color[0], 255 - trk_color[1], 255 - trk_color[2]), thickness=2)

        # Visualize Tracklet Depth (tentative, will be changed)
        if trk.depth is not None:
            trk_depth_str = "d=" + str(round(trk.depth, 3)) + "(m)"
            (tw, th) = cv2.getTextSize(trk_depth_str, CV_FONT, fontScale=1.6, thickness=2)[0]
            text_x = int((zs[0] + zs[2]) / 2.0 - tw / 2.0)
            text_y = int((zs[1] + zs[3]) / 2.0 - th / 2.0)

            # Put Depth Text (Tentative)
            cv2.putText(img, trk_depth_str, (text_x, text_y), CV_FONT, 1.6,
                        (255 - trk_color[0], 255 - trk_color[1], 255 - trk_color[2]), thickness=2)

        # Visualize Action Recognition Result
        if trk.pose is not None and is_vis_action_result is True:
            H = img.shape[0]
            W = img.shape[1]
            cv2.putText(img, str(int(trk.pose) + 1), (min(int(trk.x[0] + (trk.x[4] / 2)), W - 1), min(int(trk.x[1] + (trk.x[5] / 2)), H - 1)),
                        CV_FONT, 3, (255 - trk_color[0], 255 - trk_color[1], 255 - trk_color[2]), thickness=3)


# Class for ROS embedding
class snu_module(object):
    # SNU ROS Initialization
    def __init__(self):
        # Initialize Frame Index Variable
        self.fidx = 0

        # Initialize Tracklet Maximum ID Storage Variable
        self.max_trk_id = 0

        # Initialize Detection Model (lighthead-RCNN)
        self.infer_func, self.inputs = None, None
        self.load_detection_model(model_path, gpu_id)

        # Initialize MOT Variables
        self.trackers = []
        self.tracker_cands = []
        self.trk_colors = np.random.rand(32, 3)

        # Color Image and Timestamp
        self.color_image = None
        self.color_image_timestamp = None

        # Depth Image
        self.depth_image = None
        self.depth_image_timestamp = None

        # Odometry
        self.odometry = None

        # Extrinsics
        self.extrinsics = None

        # CvBridge
        self.bridge = CvBridge()

        # Subscribers


        # SNU Callback
        self.snu_ros_callback()

        # Publisher
        self.pub_tracks = rospy.Publisher("/osr/tracks", Tracks, queue_size=1)

    # Load Detection Model
    def load_detection_model(self, detection_model_path, detection_gpu_id):
        self.infer_func, self.inputs = detector.load_model(detection_model_path, detection_gpu_id)

    # Callback
    def snu_ros_callback(self):
    # def snu_ros_callback(self, sub_color_img, sub_depth_img, sub_odometry, sub_extrinsics):
        # Frame Index Update
        self.fidx += 1

        # Detector Module (light-head RCNN)
        curr_dets, DET_TIME = snu_detector(self.color_image, self.infer_func, self.inputs)

        # Multimodal Tracking Module (Two-step association)
        trackers, tracker_cands, MMMOT_TIME, self.max_trk_id = \
            snu_mmmot(self.color_image, self.depth_image, self.fidx, curr_dets, motparams, self.trackers, self.tracker_cands, self.max_trk_id)

        # Action Recognition Module (HOG+SVM)
        # trackers, AR_TIME = snu_ar(self.color_image, trackers)

        # Update tracklets
        self.trackers, self.tracker_cands = trackers, tracker_cands

        # ROS Tracklet Feed-in
        out_tracks = Tracks()
        out_tracks.header.stamp = self.color_image_timestamp
        for _, tracker in enumerate(self.trackers):
            track = Track()

            # Tracklet ID
            track.id = tracker.id

            # Object Type
            track.type = 1

            # Bounding Box
            track_bbox = BoundingBox()
            track_state = tracker.x

            track_bbox.x = track_state[0]
            track_bbox.y = track_state[1]
            track_bbox.height = track_state[4]
            track_bbox.width = track_state[5]

            track.bbox_pose = track_bbox

            # Append to Tracks
            out_tracks.tracks.append(track)

        # Publish Tracks
        self.pub_tracks.publish(out_tracks)

        # Detection Visualization
        if is_vis_detection is True:
            visualize_detections(self.color_image, curr_dets, line_width=2)

        # MMMOT Visualization
        if is_vis_tracking is True:
            visualize_tracklets(self.color_image, self.trackers, self.trk_colors, line_width=3)

        # Speed Visualization
        det_fps = "Detector Speed: " + str(round(1000 / DET_TIME, 2)) + " (fps)"
        mmmot_fps = "Tracker Speed: " + str(round(1000 / MMMOT_TIME, 2)) + " (fps)"
        # ar_fps = "AR Speed: " + str(round(1000 / AR_TIME, 2)) + " (fps)"
        cv2.putText(self.color_image, det_fps, (10, 20), CV_FONT, 1.3, (255, 0, 255), thickness=2)
        cv2.putText(self.color_image, mmmot_fps, (10, 50), CV_FONT, 1.3, (255, 0, 255), thickness=2)
        # cv2.putText(color_image, ar_fps, (10, 80), CV_FONT, 1.3, (255, 0, 255), thickness=2)

        # Visualization Window (using OpenCV-Python)
        if is_vis_detection is True or is_vis_tracking is True:
            cv2.imshow('Tracking', self.color_image)

        # OpenCV waitkey
        cv2.waitKey(1)


# Color Image Callback
def color_image_callback(msg):
    color_image = self.bridge.imgmsg_to_cv2(msg, '8UC3')
    color_image_timestamp = msg.header.stamp

    self.color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    self.color_image_timestamp = color_image_timestamp


# Depth Image Callback
def depth_image_callback(msg):
    depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
    depth_image_timestamp = msg.header.stamp
    self.depth_image = depth_image
    self.depth_image_timestamp = depth_image_timestamp


# Main Function
def main():
    # ROS Node Initialization
    rospy.init_node("snu_module", anonymous=True)

    # ROS Subscribers
    rospy.Subscriber("osr/d435_color_image", Image, self.color_image_callback)
    rospy.Subscriber("osr/d435_depth_image", Image, self.depth_image_callback)

    # Initialize ROS Class Module
    snu_ros_module = snu_module()



    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting Down SNU Module"


# Code Starts Here
if __name__ == '__main__':
    main()















