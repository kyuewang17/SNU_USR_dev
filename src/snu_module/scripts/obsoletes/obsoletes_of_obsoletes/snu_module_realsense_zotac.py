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
# import pyrealsense2 as rs
# Import SNU Modules
import detector__lighthead_rcnn as detector
import mot__multimodal as mmodaltracker
import action_recognition as ar
import mot_module as mot


# Parameter Struct
class STRUCT:
    def __init__(self):
        pass


#######################################

# Import ROS Modules
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from osr_msgs.msg import Track, Tracks
from osr_msgs.msg import BoundingBox



# CvBridge
bridge = CvBridge()

color_img, depth_img, odom = None, None, None
color_img_stamp = None
def color_cb(msg):
    global color_img, color_img_stamp
    # if color_img_stamp == msg.header.stamp:
    #     color_img = None
    #     return None
    # else:
    # color_img = None
    color_img_stamp = msg.header.stamp
    color_img = bridge.imgmsg_to_cv2(msg, '8UC3')
        # color_img_stamp = msg.header.stamp
        # return None


def depth_cb(msg):
    global depth_img
    depth_img = bridge.imgmsg_to_cv2(msg, '32FC1')


def odom_cb(msg):
    global odom
    pass






#######################################







# Code Execution Timestamp
script_info = STRUCT
script_info.CODE_TIMESTAMP = datetime.datetime.now()
script_info.EXECUTE_FILENAME = os.path.basename(__file__)

################# Realsense Options #################
is_realsense = "Live"
# is_realsense = "Recorded"
# is_realsense = "KITTI"

DEPTH_CLIPPED_VALUE = -1
DEPTH_CLIP_DISTANCE = 15    # (in meters)
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
motparams.DEPTH_CLIP_DIST = DEPTH_CLIP_DISTANCE
motparams.calib = None
##################################

###### Visualization Options ######
is_vis_detection = True
is_vis_tracking = True
is_vis_action_result = False

# Save Figure Options
is_save_fig = False

# openCV Font Options
CV_FONT = cv2.FONT_HERSHEY_PLAIN
###################################

# Save Visualization
is_save_vis = False
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


# Main Function
def main():

    ###################################################
    global color_img, depth_img, odom, is_working, color_img_stamp
    rospy.init_node("snu_module", anonymous=True)
    # Subscribe Images(Sensor Data) and Odometry from KIRO
    sub_color_img = rospy.Subscriber("osr/d435_color_image", Image, color_cb)
    sub_depth_img = rospy.Subscriber("osr/d435_depth_image", Image, depth_cb)
    sub_odometry = rospy.Subscriber("realsense/odometry", Odometry, odom_cb)

    # Publisher
    pub_tracks = rospy.Publisher("/osr/tracks", Tracks, queue_size=1)
    pub_img = rospy.Publisher("/osr/image_test", Image, queue_size=1)

    ##################################################

    # Load Detection Model
    infer_func, inputs = detector.load_model(model_path, gpu_id)

    # Tracklet Color (Visualization)
    # (Later, generate colormap)
    trk_colors = np.random.rand(32, 3)

    if is_realsense is "Live":
        # Configure Depth and Color Streams
        # pipeline = rs.pipeline()
        # config = rs.config()
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        #
        # # Start Streaming
        # profile = pipeline.start(config)

        # # Getting the depth sensor's depth scale (see rs-align example for explanation)
        # depth_sensor = profile.get_device().first_depth_sensor()
        # depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: ", depth_scale)

        # clipping_distance_in_meters = DEPTH_CLIP_DISTANCE
        # clipping_distance = clipping_distance_in_meters / depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        # align_to = rs.stream.color
        # align = rs.align(align_to)

        # Generalize Save Path
        # res_save_basepath = "../../../realsense_result/"
        # if os.path.isdir(res_save_basepath) is False:
        #     os.mkdir(res_save_basepath)

        # Initialize Tracklet and Tracklet Candidate Object
        trackers = []
        tracker_cands = []

        # Initialize Frame Index
        fidx = 0

        # Tracklet Maximum ID Storage
        max_trk_id = 0

        # CvBridge
        bridge2 = CvBridge()

        while not rospy.is_shutdown():
            #############################################
            if color_img is None or depth_img is None:
                continue
            in_color_img_stamp = copy.deepcopy(color_img_stamp)
            in_color_img = copy.deepcopy(color_img)
            in_color_img = cv2.cvtColor(in_color_img, cv2.COLOR_BGR2RGB)
            # BRG to RGB

            out_img = bridge2.cv2_to_imgmsg(in_color_img, "8UC3")
            pub_img.publish(out_img)



            #############################################
            # Increase Frame Index
            fidx += 1
            #
            # # Wait for a coherent pair of frames: RBGD
            # frames = pipeline.wait_for_frames()
            #
            # # Align the depth frame to color frame
            # aligned_frames = align.process(frames)
            #
            # # DEPTH (Aligned depth frame)
            # depth_frame = aligned_frames.get_depth_frame()
            #
            # # RGB
            # color_frame = aligned_frames.get_color_frame()
            #
            # if not depth_frame or not color_frame:
            #     continue
            #
            # # Convert Images to numpy arrays
            # depth_image = np.asanyarray(depth_frame.get_data())
            # color_image = np.asanyarray(color_frame.get_data())
            #
            # # Clip Depth Image
            # clipped_depth_image = np.where((depth_image > clipping_distance) | (depth_image <= 0), DEPTH_CLIPPED_VALUE, depth_image)

            # Apply ColorMap on DEPTH Image
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # DETECTION MODULE
            curr_dets, DET_TIME = snu_detector(in_color_img, infer_func, inputs)

            # MultiModal Tracking MODULE
            trackers, tracker_cands, MMMOT_TIME, max_trk_id = \
                snu_mmmot(in_color_img, depth_img, fidx, curr_dets, motparams, trackers, tracker_cands, max_trk_id)

            # Action Recognition MODULE
            # trackers, AR_TIME = snu_ar(in_color_img, trackers)

            #
            # # Detection Visualization
            if is_vis_detection is True:
                visualize_detections(in_color_img, curr_dets, line_width=2)
            #
            # MMMOT Visualization
            if is_vis_tracking is True:
                visualize_tracklets(in_color_img, trackers, trk_colors, line_width=3)

            # ROS Tracklet Feed-in
            out_tracks = Tracks()
            out_tracks.header.stamp = in_color_img_stamp
            for _, tracker in enumerate(trackers):
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
            pub_tracks.publish(out_tracks)

            # Speed Visualization
            det_fps = "Detector Speed: " + str(round(1000 / DET_TIME, 2)) + " (fps)"
            mmmot_fps = "Tracker Speed: " + str(round(1000 / MMMOT_TIME, 2)) + " (fps)"
            # ar_fps = "AR Speed: " + str(round(1000 / AR_TIME, 2)) + " (fps)"
            cv2.putText(in_color_img, det_fps, (10, 20), CV_FONT, 1.3, (255, 0, 255), thickness=2)
            cv2.putText(in_color_img, mmmot_fps, (10, 50), CV_FONT, 1.3, (255, 0, 255), thickness=2)
            # cv2.putText(color_image, ar_fps, (10, 80), CV_FONT, 1.3, (255, 0, 255), thickness=2)
            #

            # # Visualization Window (using OpenCV-Python)
            if is_vis_detection is True or is_vis_tracking is True:
                cv2.imshow('Tracking', in_color_img)


            #
            # # Later, save code must be written
            #
            # # OpenCV imshow window
            in_color_img = None
            cv2.waitKey(1)

            # # Destroy All opencv windows
            # cv2.destroyAllWindows()

    elif is_realsense is "Recorded":
        seq_basepath = "/home/daeho/snu1/realsense_data/"
        res_save_basepath = "/home/daeho/snu1/realsense_result/"

        # Dynamic Camera Scene
        # sequence_name = "[190517_142812]_aligned"
        # sequence_name = "[190517_152236]_aligned"
        # sequence_name = "[190517_152513]_aligned"
        sequence_name = "[190517_152808]_aligned"

        # Static Camera Scene
        # sequence_name = "[190517_160215]_aligned"     # Very Far
        # sequence_name = "[190517_160813]_aligned"     # Far
        # sequence_name = "[190517_161149]_aligned"     # Near

        sequence_path = seq_basepath + sequence_name + "/"

        # Save Visualization Path
        res_savepath = res_save_basepath + sequence_name + "/"
        if os.path.isdir(res_savepath) is False:
            os.mkdir(res_savepath)

        # Color Image Path
        color_image_path = sequence_path + "color/"
        color_imageNameList = os.listdir(color_image_path)
        color_imageList = [color_image_path + filename for filename in color_imageNameList]
        color_imageList.sort()

        # Depth Image Path
        depth_image_path = sequence_path + "depth_image/"
        depth_imageNameList = os.listdir(depth_image_path)
        depth_imageList = [depth_image_path + filename for filename in depth_imageNameList]
        depth_imageList.sort()

        # Depth Value Path
        depth_value_path = sequence_path + "depth/"
        depth_valueNameList = os.listdir(depth_value_path)
        depth_valueList = [depth_value_path + filename for filename in depth_valueNameList]
        depth_valueList.sort()

        # Read depth scale
        depth_scale_filename = sequence_path + "depth_scale.txt"
        if os.path.isfile(depth_scale_filename) is True:
            depth_scale_file = open(depth_scale_filename, 'r')
            depth_scale = float(depth_scale_file.read())
        else:
            print("[Error] Need depth scale..!")
            assert 0
            # depth_scale = 1e-30     # Very Small Number

        # Set Clipping Distance
        clipping_distance_in_meters = DEPTH_CLIP_DISTANCE  # 1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        # Check for File Length in both Paths, truncate maximum frame length to the minimum value
        max_frame_length = min(len(color_imageList), len(depth_valueList))
        if len(color_imageList) != len(depth_valueList):
            print("[WARNING] Color Image and Depth Value Length Does Not Match!")

        # Initialize Tracklet and Tracklet Candidate Object
        trackers = []
        tracker_cands = []

        # For every frames in the image sequences
        for fidx in range(max_frame_length):
            # Frame Counter
            fidx += 1

            # Color Frame Image
            color_image = cv2.imread(color_imageList[fidx-1])

            # Depth Image from Depth Value (Load text file frame-by-frame)
            depth_value = np.loadtxt(depth_valueList[fidx-1])

            # Clip Depth Image
            clipped_depth_image = np.where((depth_value > clipping_distance) | (depth_value <= 0), DEPTH_CLIPPED_VALUE, depth_value)

            # Apply ColorMap on DEPTH Image
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # DETECTION MODULE
            curr_dets, DET_TIME = snu_detector(color_image, infer_func, inputs)

            # MultiModal Tracking MODULE
            trackers, tracker_cands, MMMOT_TIME = snu_mmmot(color_image, clipped_depth_image, fidx, curr_dets, motparams, trackers, tracker_cands)
            # trackers, tracker_cands, MMMOT_TIME = snu_mmmot(color_image, depth_image, fidx, curr_dets, motparams, trackers, tracker_cands)

            # Detection Visualization
            if is_vis_detection is True:
                visualize_detections(color_image, curr_dets, line_width=2)

            # MMMOT Visualization
            if is_vis_tracking is True:
                visualize_tracklets(color_image, trackers, trk_colors, line_width=3)

            # Detection Speed Visualization
            det_fps = "Detector Speed: " + str(round(1000 / DET_TIME, 2)) + " (fps)"
            mmmot_fps = "Tracker Speed: " + str(round(1000 / MMMOT_TIME, 2)) + " (fps)"
            cv2.putText(color_image, det_fps, (10, 20), CV_FONT, 1.3, (255, 0, 255), thickness=2)
            cv2.putText(color_image, mmmot_fps, (10, 50), CV_FONT, 1.3, (255, 0, 255), thickness=2)

            # Visualization Window (using OpenCV-Python)
            cv2.imshow('Tracking', color_image)

            # Save Visualization Window
            if is_save_vis is True:
                res_filename = ("%05d" % fidx) + "_result.png"
                cv2.imwrite(res_savepath + res_filename, color_image)

            # OpenCV imshow window
            cv2.waitKey(1)

    else:
        # Sequence Path
        seq_basepath = '../DATA/'
        #sequences = ['US']
        sequences = ['ETH-Crossing', 'KITTI-17']

        res_save_basepath = "/home/daeho/snu1/kitti_result/"

        # For every image sequences
        for seq_num, seq in enumerate(sequences):
            # Sequence Number
            seq_num += 1

            # Current Sequence Path
            seqpath = seq_basepath + seq

            # Save Path
            res_savepath = res_save_basepath + seq + "/"
            if os.path.isdir(res_savepath) is False:
                os.mkdir(res_savepath)

            # Current Image Sequence Paths
            seq_imgspath = seqpath + '/imgs/'

            # Load Image Sequence Lists
            imgNameList = os.listdir(seq_imgspath)
            imgList = [seq_imgspath + filename for filename in imgNameList]
            imgList.sort()

            # Generalize Save Path

            # Initialize Tracklet and Tracklet Candidate Object
            trackers = []
            tracker_cands = []

            # For every frames in the image sequence
            for fidx, frameName in enumerate(imgList):
                # Frame Index Starts from 1
                fidx += 1

                # Load Current Frame Image
                # frame = io.imread(frameName)
                frame = cv2.imread(frameName)

                # DETECTION MODULE
                curr_dets, DET_TIME = snu_detector(frame, infer_func, inputs)

                # MultiModal Tracking MODULE
                trackers, tracker_cands, MMMOT_TIME = snu_mmmot(frame, [], fidx, curr_dets, motparams, trackers, tracker_cands)

##########################################################################################
##########################################################################################
                # Action Recognition MODULE
                trackers, AR_TIME = snu_ar(frame, trackers)
##########################################################################################
##########################################################################################

                # Detection Visualization
                if is_vis_detection is True:
                    visualize_detections(frame, curr_dets, line_width=2)

                # MMMOT Visualization
                if is_vis_tracking is True:
                    visualize_tracklets(frame, trackers, trk_colors, line_width=3)

                # Detection Speed Visualization
                det_fps = "Detector Speed: " + str(round(1000 / DET_TIME, 2)) + " (fps)"
                mmmot_fps = "Tracker Speed: " + str(round(1000 / MMMOT_TIME, 2)) + " (fps)"
                ar_fps = "AR Speed: "+ str(round(1000 / AR_TIME, 2)) + " (fps)"
                cv2.putText(frame, det_fps, (10, 20), CV_FONT, 1.3, (255, 0, 255), thickness=2)
                cv2.putText(frame, mmmot_fps, (10, 50), CV_FONT, 1.3, (255, 0, 255), thickness=2)
                cv2.putText(frame, ar_fps, (10, 80), CV_FONT, 1.3, (255, 0, 255), thickness=2)
                # Visualization Window (using OpenCV-Python)
                cv2.imshow('Tracking', frame)

                # Save Visualization Window
                if is_save_vis is True:
                    res_filename = ("%05d" % fidx) + "_result.png"
                    cv2.imwrite(res_savepath + res_filename, frame)

                # OpenCV imshow window
                cv2.waitKey(1)

        # Destroy openCV window
        cv2.destroyAllWindows()


# Main Function
if __name__ == '__main__':
    main()
