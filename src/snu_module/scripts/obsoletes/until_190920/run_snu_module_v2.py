#!/usr/bin/env python
"""
SNU Integrated Module v2.0
  - Main Execution Code (RUN CODE)

    << Project Information >>
    - "Development of multimodal sensor-based intelligent systems for outdoor surveillance robots"
    - [2017-04-01]~[2021-12-31]

    << Institutions & Researchers >>
    - Perception and Intelligence Laboratory (PIL)
        [1] Kyuewang Lee (kyuewang@snu.ac.kr)
        [2] Daeho Um (umdaeho1@gmail.com)
    - Machine Intelligence and Pattern Recognition Laboratory (MIPAL)
        [1] Jae-Young Yoo (yoojy31@snu.ac.kr)
        [2] Hojun Lee (hojun815@snu.ac.kr)
        [3] Inseop Chung (jis3613@snu.ac.kr)

    << Code Environment >>
        - python == 2.7
        - pyTorch ==
        - CUDA ==
        - ROS-kinetics
          (for ROS embedding)

        < Dependencies >

"""
# Import Modules
import os
import cv2
import datetime, time
import copy
import numpy as np
# import detector__lighthead_rcnn as detector

# import action_recognition as ar

import module_detection_old as snu_det
import detection_option as det_opt
import module_tracking_old as snu_mmt
import module_action as snu_ar


#### ROS Embedding Selection ####
is_ros_embedded = True
#################################

if is_ros_embedded is True:
    # Import ROS Modules
    import rospy
    from rospy.numpy_msg import numpy_msg
    from cv_bridge import CvBridge, CvBridgeError

    # Import ROS Messages
    from std_msgs.msg import String
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2
    from nav_msgs.msg import Odometry
    from osr_msgs.msg import Track, Tracks, BoundingBox
    from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
    from tf.transformations import quaternion_from_euler


# Parameter Struct
class STRUCT:
    def __init__(self):
        pass


# Script Execution Timestamp
script_info = STRUCT()
script_info.CODE_TIMESTAMP = datetime.datetime.now()
script_info.EXECUTE_FILENAME = os.path.basename(__file__)


# CvBridge
bridge = CvBridge()     # Multimodal Sensor Callback
bridge2 = CvBridge()    # Publish SNU Results

###############################
# Initialize Global Variables #
###############################
color_img, depth_img, odom, extrinsic = None, None, None, None
lidar_img, lidar_img2, lidar_pc = None, None, None
infrared_img, thermal_img, nv_img = None, None, None

color_img_stamp = None
lidar_img_stamp = None
lidar_img2_stamp = None

# Camera Intrinsic Matrix (Color, Depth)
color_cam_K, depth_cam_K = None, None

# Camera Projection Matrix (Color, Depth)
color_cam_P, depth_cam_P = None, None


############################
# Callback Functions (ROS) #
############################
if is_ros_embedded is True:
    # [1] RGB
    def color_callback(msg):
        global color_img, color_img_stamp
        color_img_stamp = msg.header.stamp
        color_img = bridge.imgmsg_to_cv2(msg, "8UC3")

    # [2] Depth
    def depth_callback(msg):
        global depth_img
        depth_img = bridge.imgmsg_to_cv2(msg, "16UC1")

    # [3] LiDAR (Image x 2  +  sensor_msgs.msg.PointCloud2)
    def lidar_callback(msg):
        global lidar_img, lidar_img_stamp
        lidar_img = bridge.imgmsg_to_cv2(msg, "8UC3")
        lidar_img_stamp = msg.header.seq

    def lidar2_callback(msg):
        global lidar_img2, lidar_img2_stamp
        lidar_img2 = bridge.imgmsg_to_cv2(msg, "8UC3")
        lidar_img2_stamp = msg.header.seq

    def lidar_pc_callback(msg):
        global lidar_pc
        lidar_pc = -1

    # [4] Infrared
    def infrared_callback(msg):
        global infrared_img
        infrared_img = bridge.imgmsg_to_cv2(msg, "8UC1")

    # [5] Thermal
    def thermal_callback(msg):
        global thermal_img
        thermal_img = bridge.imgmsg_to_cv2(msg, "8UC1")

    # [6] NightVision
    def nightvision_callback(msg):
        global nv_img
        nv_img = bridge.imgmsg_to_cv2(msg, "8UC3")

    def odom_callback(msg):
        global odom
        odom = msg

    def color_cam_param_callback(msg):
        global color_cam_K, color_cam_P
        color_cam_K = msg.K.reshape((3, 3))
        color_cam_P = msg.P.reshape((3, 4))

        # Subscribe Only Once
        sub_color_cam_params.unregister()

    def depth_cam_param_callback(msg):
        global depth_cam_K, depth_cam_P
        depth_cam_K = msg.K.reshape((3, 3))
        depth_cam_P = msg.P.reshape((3, 4))

        # Subscribe Only Once
        sub_depth_cam_params.unregister()


############ Define Parameter Struct ############
# Initialization
params = STRUCT()

# Image Struct
params.image = STRUCT()
params.image.rgb = STRUCT()
params.image.depth = STRUCT()
params.image.lidar = STRUCT()
params.image.infrared = STRUCT()
params.image.thermal = STRUCT()
params.image.nightvision = STRUCT()

# Boolean to check ROS Embedding Activation
params.is_ros = is_ros_embedded
#################################################

########### Depth Camera Clip Distance ###########
# Clip out-of-range distance to this value
params.image.depth.CLIP_VALUE = -1

# Clip distance range (min and max distance)
# (the scale is in "millimeters")
params.image.depth.CLIP_MIN_DISTANCE = 1000
params.image.depth.CLIP_MAX_DISTANCE = 20000
##################################################

######## LIDAR DATA SCALING Parameters ########
# Scaling Parameter
params.image.lidar.scaling = float(50) / float(255)

# Clip out-of-range lidar values to this value
params.image.lidar.CLIP_VALUE = -2


###############################################

################## SNU Module Options ##################
########### Detection Parameters ###########
# model_base_path = os.path.dirname(os.path.abspath(__file__))
# detector_name = "lighthead_rcnn"
# model_path = model_base_path + "/" + detector_name + "/model/detector.ckpt"
# gpu_id = "0"

# Push in to the parameter struct
# params.detector = STRUCT()
# params.detector.model_path = model_path
# params.detector.gpu_id = gpu_id
############################################

######### Multi-target Tracking Parameters #########
# < Tracklet Candidate -> Tracklet > Association Age (Tracklet Initialization)
trk_init_age = 10

# Destroy unassociated tracklets with this amount of continuous unassociation
trk_destroy_age = 4

# Destroy unassociated tracklet candidates with this amount of continuous unassociation
trkc_destroy_age = 6

# Association Thresholds
cost_thresh_loose = 0.3
cost_thresh_tight = 0.6

# Multi-modal Histogram Bin Number
hist_bin = 500

# Push in to the parameter struct
params.tracker = STRUCT()
params.tracker.association = STRUCT()
params.tracker.association.age = STRUCT()

params.tracker.hist_bin = hist_bin
params.tracker.association.age.trk_init = trk_init_age
params.tracker.association.age.trk_destroy = trk_destroy_age
params.tracker.association.age.trkc_destroy = trkc_destroy_age
params.tracker.association.threshold_loose = cost_thresh_loose
params.tracker.association.threshold_tight = cost_thresh_tight
####################################################

######### Visualization Options #########
params.visualization = STRUCT()
params.visualization.detection = True
params.visualization.tracking = True
params.visualization.action = True

# OpenCV Font Options
CV_FONT = cv2.FONT_HERSHEY_PLAIN

# Tracklet Visualization Color
params.visualization.tracklet_color = np.random.rand(32, 3)

# More Options

#########################################
########################################################


########### Runner Functions ###########
# Detector Runner
def snu_detector(framework, imgStruct, device):
    # Start Timestamp for Detection
    DET_TS_START = datetime.datetime.now()

    # Activate Detector Module (RefineDet)
    curr_dets = snu_det.detect(framework, imgStruct.rgb.raw, device)

    # Stop Timestamp for Detection
    DET_TS_STOP = datetime.datetime.now()
    # Elapsed Time for the Detection Module (seconds)
    DET_ELAPSED_TIME = (DET_TS_STOP - DET_TS_START).total_seconds()

    return curr_dets, DET_ELAPSED_TIME


# Multimodal Multi-target Tracker Runner
def snu_tracker(imgStruct, fidx, dets, trackers, tracker_cands, tparams, max_id):
    # Start Timestamp for Multimodal Multi-target Tracker (MMT)
    MMT_TS_START = datetime.datetime.now()
    # Activate MMT Module
    trackers, tracker_cands, max_id = snu_mmt.tracker(imgStruct, fidx, dets, tparams, max_id, trackers, tracker_cands)
    # Stop Timestamp for MMT
    MMT_TS_STOP = datetime.datetime.now()
    # Elapsed Time for the MMT Module (seconds)
    MMT_ELAPSED_TIME = (MMT_TS_STOP - MMT_TS_START).total_seconds()

    return trackers, tracker_cands, MMT_ELAPSED_TIME, max_id


# Action Recognition Runner
def snu_action_recognition(imgStruct, trackers):
    # Start Timestamp for Action Recognition
    AR_TS_START = datetime.datetime.now()
    # Activate AR Module
    trackers = snu_ar.res_clf(imgStruct.rgb.raw, trackers)
    # Stop Timestamp for Action Recognition
    AR_TS_STOP = datetime.datetime.now()
    # Elapsed Time for the AR Module (seconds)
    AR_ELAPSED_TIME = (AR_TS_STOP - AR_TS_START).total_seconds()

    return trackers, AR_ELAPSED_TIME


####### Visualizer Runner #######
def visualize_detections(imgStruct, dets, vparams, linewidth=2):
    for det in dets:
        det = det.astype(np.int32)

        # Draw Rectangle BBOX (Color is RED)
        cv2.rectangle(imgStruct.rgb.draw,
                      (det[0], det[1]), (det[2], det[3]),
                      (0, 0, 255), linewidth)


# << Visualize Tracklet Bounding Box and Action Recognition Results >>
def visualize_tracklets(imgStruct, trks, vparams, linewidth=2):
    img = imgStruct.rgb.draw
    for trk in trks:
        zs, _ = snu_mmt.fbbox.zx_to_bbox(trk.states[-1])
        zs = zs.astype(np.int32)

        trk_color = trk.color

        # Draw Rectangle BBOX
        cv2.rectangle(img, (zs[0], zs[1]), (zs[2], zs[3]),
                      (trk_color[0], trk_color[1], trk_color[2]), linewidth)

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
            (tw, th) = cv2.getTextSize(cam_coord_vel_str, CV_FONT, fontScale=font_size * 0.8, thickness=2)[0]
            text_x = int((zs[0] + zs[2]) / 2.0 - tw / 2.0)
            text_y = box_coords[0][1] - info_interval
            box_coords = ((int(text_x - pad_pxls / 2.0), int(text_y - th - pad_pxls / 2.0)),
                          (int(text_x + tw + pad_pxls / 2.0), int(text_y + pad_pxls / 2.0)))
            cv2.rectangle(img, box_coords[0], box_coords[1], (trk_color[0], trk_color[1], trk_color[2]), cv2.FILLED)
            # cv2.putText receives "bottom-left" point of the text
            cv2.putText(img, cam_coord_vel_str, (text_x, text_y), CV_FONT, font_size * 0.8,
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
        if trk.pose is not None and vparams.visualization.action is True:
            H = img.shape[0]
            W = img.shape[1]
            cv2.putText(img, str(int(trk.pose) + 1), (min(int(trk.x[0] + (trk.x[4] / 2)), W - 1), min(int(trk.x[1] + (trk.x[5] / 2)), H - 1)),
                        CV_FONT, 3, (255 - trk_color[0], 255 - trk_color[1], 255 - trk_color[2]), thickness=3)

        # Draw
        imgStruct.rgb.draw = img
#################################
########################################


# Main Function
def main():
    if is_ros_embedded is True:
        global is_working
        global color_img_stamp, odom
        global lidar_img_stamp, lidar_img2_stamp
        global color_img, depth_img, lidar_img, lidar_img2, lidar_pc, infrared_img, thermal_img, nv_img
        global color_cam_K, color_cam_P, depth_cam_K, depth_cam_P
        global sub_color_cam_params, sub_depth_cam_params

        rospy.init_node("snu_module", anonymous=True)

        ######### Subscribe Multimodal Sensor Data #########
        # [1] RGB (Image)
        sub_rgb = rospy.Subscriber("osr/image_color", Image, color_callback)
        # [2] Depth (Image)
        sub_depth = rospy.Subscriber("osr/image_depth", Image, depth_callback)
        # [3] LiDAR (Image and PointCloud2)
        sub_lidar = rospy.Subscriber("camera_lidar", Image, lidar_callback)
        sub_lidar2 = rospy.Subscriber("camera_lidar2", Image, lidar2_callback)
        sub_lidar_pc = rospy.Subscriber("osr/lidar_pointcloud", PointCloud2, lidar_pc_callback)
        # [4] Infrared (Image)
        sub_infrared = rospy.Subscriber("osr/image_ir", Image, infrared_callback)
        # [5] Thermal
        sub_thermal = rospy.Subscriber("osr/image_thermal", Image, thermal_callback)
        # [6] NightVision
        sub_nightvision = rospy.Subscriber("osr/image_nv1", Image, nightvision_callback)
        ####################################################

        # Subscribe Odometry
        sub_odom = rospy.Subscriber("robot_odom", Odometry, odom_callback)

        # Subscribe Only Once (Camera Info)
        sub_color_cam_params = rospy.Subscriber("camera/color/camera_info", numpy_msg(CameraInfo), color_cam_param_callback)
        sub_depth_cam_params = rospy.Subscriber("camera/depth/camera_info", numpy_msg(CameraInfo), depth_cam_param_callback)

        ################### Publisher ###################
        pub_tracks = rospy.Publisher("/osr/tracks", Tracks, queue_size=1)
        pub_img = rospy.Publisher("/osr/snu_result_image", Image, queue_size=1)
        #################################################

    #######################
    # Load [Lighthead-RCNN] Detector Model
    # infer_func, inputs = detector.load_model(model_path, gpu_id)
    detector = snu_det.load_model(det_opt.detection_model_dir, det_opt.device)

    # Initialize Tracklet and Tracklet Candidate
    trackers, tracker_cands = [], []

    # Initialize Frame Index
    fidx = 0

    # Tracklet Maximum ID Storage
    max_id = 0

    # Get Camera Calibration Matrix
    params.calibration = STRUCT()
    if is_ros_embedded is True:
        params.calibration.rgb_K = color_cam_K
        params.calibration.rgb_P = color_cam_P
        params.calibration.depth_K = depth_cam_K
        params.calibration.depth_P = depth_cam_P
    else:
        params.calibration.rgb_K, params.calibration.rgb_P = None, None
        params.calibration.depth_K, params.calibration.depth_P = None, None

    # Initialize imgStruct
    imgStruct = STRUCT()
    if is_ros_embedded is True:
        imgStruct.rgb, imgStruct.depth, imgStruct.lidar, imgStruct.infrared, imgStruct.thermal, imgStruct.nightvision = \
            STRUCT(), STRUCT(), STRUCT(), STRUCT(), STRUCT(), STRUCT()
        imgStruct.rgb.raw, imgStruct.rgb.draw = None, None
        imgStruct.depth.raw, imgStruct.depth.processed = None, None
        imgStruct.lidar.point_cloud = None
        imgStruct.lidar.raw, imgStruct.lidar.raw2 = None, None
        imgStruct.lidar.timestamp, imgStruct.lidar.timestamp2 = None, None
        imgStruct.lidar.prev_timestamp, imgStruct.lidar.prev_timestamp2 = None, None
        imgStruct.lidar.processed, imgStruct.lidar.processed2 = None, None
        imgStruct.infrared.raw, imgStruct.infrared.processed = None, None
        imgStruct.thermal.raw, imgStruct.thermal.processed = None, None
        imgStruct.nightvision.raw, imgStruct.nightvision.processed = None, None
    else:
        imgStruct.rgb, imgStruct.depth = STRUCT(), STRUCT()
        imgStruct.rgb.raw, imgStruct.rgb.draw = None, None
        imgStruct.depth.raw, imgStruct.depth.processed = None, None

    ############ Practical Algorithm Starts from HERE ############
    # [ROS Embedding] Version
    if is_ros_embedded is True:
        while not rospy.is_shutdown():
            if color_img is None or depth_img is None:
                continue

            # Update LIDAR Timestamp
            imgStruct.lidar.prev_timestamp = copy.deepcopy(imgStruct.lidar.timestamp)
            imgStruct.lidar.prev_timestamp2 = copy.deepcopy(imgStruct.lidar.timestamp2)

            # Push Multimodal Sensor Data into [imgStruct]
            imgStruct.rgb.raw, imgStruct.rgb.draw = color_img, copy.deepcopy(color_img)
            imgStruct.depth.raw = copy.deepcopy(depth_img)

            if lidar_img is not None:
                # If LIDAR Timestamp Increases
                if lidar_img_stamp > imgStruct.lidar.prev_timestamp:
                    imgStruct.lidar.raw = copy.deepcopy(lidar_img[:, :, 1])
                    imgStruct.lidar.timestamp = copy.deepcopy(lidar_img_stamp)
                else:
                    imgStruct.lidar.raw = None

            if lidar_img2 is not None:
                # If LIDAR Timestamp Increases
                if lidar_img2_stamp > imgStruct.lidar.prev_timestamp2:
                    imgStruct.lidar.raw2 = copy.deepcopy(lidar_img2[:, :, 1])
                    imgStruct.lidar.timestamp2 = copy.deepcopy(lidar_img2_stamp)
                else:
                    imgStruct.lidar.raw2 = None

            imgStruct.lidar.point_cloud = lidar_pc
            imgStruct.infrared.raw = copy.deepcopy(infrared_img)
            imgStruct.thermal.raw = copy.deepcopy(thermal_img)
            imgStruct.nightvision.raw = copy.deepcopy(nv_img)

            # in_color_img = copy.deepcopy(color_img)
            in_color_img_stamp = copy.deepcopy(color_img_stamp)

            # Covert Depth Image Type (uint16 --> float32)
            depth_img = depth_img.astype(np.float32)

            # Clip Depth Image
            depth_img = np.where((depth_img < params.image.depth.CLIP_MIN_DISTANCE) |
                                 (depth_img > params.image.depth.CLIP_MAX_DISTANCE),
                                 params.image.depth.CLIP_VALUE, depth_img)
            imgStruct.depth.processed = depth_img

            # Process Lidar Images (map "zero" values to -2), convert to [millimeters]
            if imgStruct.lidar.raw is not None:
                lidar_processed = imgStruct.lidar.raw * params.image.lidar.scaling * 1000
                imgStruct.lidar.processed = np.where(lidar_processed == 0,
                                                     params.image.lidar.CLIP_VALUE,
                                                     lidar_processed)
            else:
                imgStruct.lidar.processed = None
            if imgStruct.lidar.raw2 is not None:
                lidar_processed2 = imgStruct.lidar.raw2 * params.image.lidar.scaling * 1000
                imgStruct.lidar.processed2 = np.where(lidar_processed2 == 0,
                                                      params.image.lidar.CLIP_VALUE,
                                                      lidar_processed2)
            else:
                imgStruct.lidar.processed2 = None

            # Increase Frame Count Index
            fidx += 1

            ############################################################
            ###### Detection Module ######
            curr_dets, DET_TIME = snu_detector(detector, imgStruct, det_opt.device)
            ##############################

            ###### Multimodal Multi-target Tracking Module ######
            trackers, tracker_cands, MMT_TIME, max_id = \
                snu_tracker(imgStruct, fidx, curr_dets, trackers, tracker_cands, params, max_id)
            #####################################################

            ###### Action Recognition Module ######
            trackers, AR_TIME = snu_action_recognition(imgStruct, trackers)
            #######################################
            ############################################################

            # print("MMT Time is: %2.4f" % (1/MMT_TIME))
            # print("AR Time is: %2.4f" % (1/AR_TIME))

            ############## Visualization ##############
            # [1] Detection Result Visualizer
            if params.visualization.detection is True:
                visualize_detections(imgStruct, curr_dets, params, linewidth=2)
            # [2] Tracking and Action Recognition Result Visualizer
            if params.visualization.tracking is True:
                visualize_tracklets(imgStruct, trackers, params, linewidth=2)
            ###########################################

            #### ROS Tracklet Feed-in (Publish to ETRI Module) ####
            out_tracks = Tracks()
            out_tracks.header.stamp = in_color_img_stamp

            # Odometry
            out_tracks.odom = odom

            for _, tracker in enumerate(trackers):
                # Get Tracklet Information
                track_state = tracker.states[-1]
                if len(tracker.states) > 1:
                    track_prev_state = tracker.states[-2]
                else:
                    track_prev_state = np.zeros(6).reshape((6, 1))
                track_cam_coord_state = tracker.cam_coord_estimated_state

                # Initialize Track
                track = Track()

                # Tracklet ID
                track.id = tracker.id

                # Tracklet Object Type
                track.type = 1

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

            # Publish Tracks, Result Image
            pub_tracks.publish(out_tracks)
            if params.visualization.detection is True or params.visualization.tracking is True:
                pub_img.publish(bridge2.cv2_to_imgmsg(color_img, "8UC3"))

            # Visualization Window (using OpenCV-Python)
            if params.visualization.detection is True or params.visualization.tracking is True:
                winname = "SNU Result"
                cv2.namedWindow(winname)
                if fidx == 1:
                    cv2.moveWindow(winname, 200, 200)
                cv2.imshow(winname, imgStruct.rgb.draw)
            cv2.waitKey(1)
    else:
        # Align(synchronize) Multimodal Input Data w.r.t. COLOR(rgb) Image
        rgb_frames = []

        # Convert synchronized multimodal inputs into imgStructs <List>
        imgStruct = STRUCT()
        imgStructs = []

        for fidx, curr_imgStruct in enumerate(imgStructs):
            # Load Multimodal Sensor Data and Process
            # load_multimodal_data()

            ############################################################
            ###### Detection Module ######
            curr_dets, DET_TIME = snu_detector(imgStruct)
            ##############################

            ###### Multimodal Multi-target Tracking Module ######
            trackers, tracker_cands, MMT_TIME, max_id = \
                snu_tracker(imgStruct, fidx, curr_dets, trackers, tracker_cands, params, max_id)
            #####################################################

            ###### Action Recognition Module ######
            trackers, AR_TIME = snu_action_recognition(imgStruct, trackers)
            #######################################
            ############################################################

            ############## Visualization ##############
            # [1] Detection Result Visualizer
            if params.visualization.detection is True:
                visualize_detections(imgStruct, curr_dets, linewidth=2)
            # [2] Tracking and Action Recognition Result Visualizer
            if params.visualization.tracking is True:
                visualize_tracklets(imgStruct, trackers, linewidth=2)
            ###########################################

            # Visualization Window (using OpenCV-Python)
            if params.visualization.detection is True or params.visualization.tracking is True:
                winname = "SNU Result"
                cv2.namedWindow(winname)
                if fidx == 1:
                    cv2.moveWindow(winname, 200, 200)
                cv2.imshow(winname, imgStruct.rgb.draw)
            cv2.waitKey(1)


# Main Function
if __name__ == '__main__':
    main()
