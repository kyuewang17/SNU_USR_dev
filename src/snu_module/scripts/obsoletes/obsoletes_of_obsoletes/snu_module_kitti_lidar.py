"""
SNU Detection and MOT Module v2.5 (KITTI RGB+LiDAR Version)

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
            - [pykitti], [IPython], [easydict]

    Source Code all rights reserved


"""

# Import Modules
import os
import cv2
import datetime
import pykitti
import numpy as np
import pyrealsense2 as rs

# Import SNU Modules
import detector__lighthead_rcnn as detector
import mot__multimodal as mmodaltracker
import mot_module as mot
import action_recognition as ar


# Parameter Struct
class STRUCT:
    def __init__(self):
        pass


# Code Execution Timestamp
script_info = STRUCT
script_info.CODE_TIMESTAMP = datetime.datetime.now()
script_info.EXECUTE_FILENAME = os.path.basename(__file__)

################## SNU Module Options ##################
###### Detection Parameters ######
model_base_path = os.path.dirname(os.path.abspath(__file__))
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
motparams.DEPTH_CLIP_DIST = np.float64("inf")
motparams.DEPTH_CLIP_VALUE = -1.0

# Tracking Result Save Option
is_save_tracking_result = False
##################################

###### Visualization Options ######
is_vis_detection = False
is_vis_tracking = True
is_vis_action_result = True

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
def snu_mmmot(color_image, depth_image, fidx, dets, motparams, trackers, tracker_cands):
    # Start Timestamp for MultiModal Multi-Object Tracker
    MMMOT_TS_START = datetime.datetime.now()
    # MMMOT Module
    trackers, tracker_cands = mmodaltracker.tracker(color_image, depth_image, fidx, dets, motparams, trackers, tracker_cands)
    # STOP Timestamp for MultiModal Multi-Object Tracker
    MMMOT_TS_STOP = datetime.datetime.now()
    # Elapsed Time for the MMMOT Module (ms)
    MMMOT_ELAPSED_TIME = (MMMOT_TS_STOP - MMMOT_TS_START).total_seconds() * 1000

    return trackers, tracker_cands, MMMOT_ELAPSED_TIME


# Action Recognition Function
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


# Visualize Tracklets (openCV version) - visualize action recognition result also
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
        text_y = int((10*zs[1] - zs[3]) / 9.0 - th / 2.0)
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
                        (255-trk_color[0], 255-trk_color[1], 255-trk_color[2]), thickness=2)

        # Visualize Action Recognition Result
        if trk.pose is not None and is_vis_action_result is True:
            H = img.shape[0]
            W = img.shape[1]
            cv2.putText(img, str(int(trk.pose)+1), (min(int(trk.x[0]+(trk.x[4]/2)), W-1), min(int(trk.x[1]+(trk.x[5]/2)), H-1)),
                        CV_FONT, 3, (255-trk_color[0], 255-trk_color[1], 255-trk_color[2]), thickness=3)


# Parse KITTI Sequence Name
def parse_kitti_sequence_name(sequence_name):
    name_segs = sequence_name.split('_')

    sequence_date = name_segs[0] + '_' + name_segs[1] + '_' + name_segs[2]
    sequence_drive = name_segs[4]

    return sequence_date, sequence_drive


# Save Tracking Result
def save_tracking_result(save_path, img_name, trks, fidx):
    # array: [ fidx || id || center_x || center_y || width || height || depth || pose ]
    tracking_result = -np.ones((len(trks), 8))
    for trk_idx, trk in enumerate(trks):
        # (1) Tracklet ID
        tracking_result[trk_idx, 0] = fidx

        # (2) Tracklet State
        tracking_result[trk_idx, 1] = trk.id

        # (3) Tracklet BBOX center x,y,w,h
        tracking_result[trk_idx, 2] = trk.x[0]
        tracking_result[trk_idx, 3] = trk.x[1]
        tracking_result[trk_idx, 4] = trk.x[2]
        tracking_result[trk_idx, 5] = trk.x[3]

        # (4) Tracklet BBOX Depth
        tracking_result[trk_idx, 6] = trk.depth

        # (5) Tracklet BBOX Action
        tracking_result[trk_idx, 7] = trk.pose

    result_txt_filepath = save_path + img_name.split('.')[0].split('/')[-1] + '.txt'

    # Save Result TXT File
    np.savetxt(result_txt_filepath, tracking_result)


# Main Function
def main():
    # Load Detection Model
    infer_func, inputs = detector.load_model(model_path, gpu_id)

    # Tracklet Color (Visualization)
    # (Later, generate colormap)
    trk_colors = np.random.rand(32, 3)

    # Sequence Path
    seq_basepath = '/home/kyle/KITTI/'
    res_save_basepath = seq_basepath + 'tracking_results/'
    # sequence = '2011_09_26_drive_0091_sync'
    # sequence = '2011_09_28_drive_0016_sync'   # static
    sequence = '2011_09_28_drive_0021_sync'   # static
    # sequence = '2011_09_28_drive_0037_sync'
    # sequence = '2011_09_28_drive_0038_sync'
    # sequence = '2011_09_28_drive_0053_sync'

    sequence_date, sequence_drive = parse_kitti_sequence_name(sequence)

    # KITTI Raw Data Structure
    kitti_data = pykitti.raw(seq_basepath, sequence_date, sequence_drive)
    motparams.calib = kitti_data.calib

    # Load Image Sequence Lists (KITTI Left Image)
    imgList = kitti_data.cam2_files

    # Load LiDAR Data Lists
    lidar_dataList = kitti_data.velo_files

    # Generalize Save Path
    if os.path.isdir(res_save_basepath) is False:
        os.mkdir(res_save_basepath)
    res_savepath = res_save_basepath + sequence + '/'
    if os.path.isdir(res_savepath) is False:
        os.mkdir(res_savepath)

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

        # Load Current Frame LiDAR Point Cloud Data
        lidar_pc = np.fromfile(lidar_dataList[fidx-1], dtype=np.float32)
        lidar_pc = lidar_pc.reshape((-1, 4))

        # DETECTION MODULE
        curr_dets, DET_TIME = snu_detector(frame, infer_func, inputs)

        # MultiModal Tracking MODULE
        trackers, tracker_cands, MMMOT_TIME = snu_mmmot(frame, lidar_pc, fidx, curr_dets, motparams, trackers, tracker_cands)

        # Action Recognition MODULE
        trackers, AR_TIME = snu_ar(frame, trackers)

        # Detection Visualization
        if is_vis_detection is True:
            visualize_detections(frame, curr_dets, line_width=2)

        # MMMOT Visualization
        if is_vis_tracking is True:
            visualize_tracklets(frame, trackers, trk_colors, line_width=3)

        # Speed Visualization
        det_fps = "Detector Speed: " + str(round(1000 / DET_TIME, 2)) + " (fps)"
        mmmot_fps = "Tracker Speed: " + str(round(1000 / MMMOT_TIME, 2)) + " (fps)"
        ar_fps = "AR Speed: " + str(round(1000 / AR_TIME, 2)) + " (fps)"
        cv2.putText(frame, det_fps, (10, 20), CV_FONT, 1.3, (255, 0, 255), thickness=2)
        cv2.putText(frame, mmmot_fps, (10, 50), CV_FONT, 1.3, (255, 0, 255), thickness=2)
        cv2.putText(frame, ar_fps, (10, 80), CV_FONT, 1.3, (255, 0, 255), thickness=2)

        # Visualization Window (using OpenCV-Python)
        if is_vis_detection is True or is_vis_tracking is True:
            cv2.imshow('Tracking', frame)

        # Save Tracking Result
        if is_save_tracking_result is True:
            save_tracking_result(res_savepath, frameName, trackers, fidx)

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
