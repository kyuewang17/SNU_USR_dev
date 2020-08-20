#!/usr/bin/env python
"""
Extract Multimodal Frames and LiDAR PointCloud Data
from rosbag file

(Unmanned Surveillance Robot Project)

Code Written by Kyuewang Lee
    - Perception and Intelligence Laboratory

* About Camera Parameter ROS Message
For Camera Parameter Specifics, refer to the following link
[LINK] : http://docs.ros.org/kinetic/api/sensor_msgs/html/msg/CameraInfo.html

** Frame Formats
[1] Color (D435-RGB)
    - 8UC3
    - Save as *.png format
        - Frame File name based on Time-stamp
    - CameraInfo
        - Save Camera Parameters as a single *.npz file
[2] Disparity (D435-Depth)
    - 16UC1
    - Save as *.png format
        - Frame File name based on Time-stamp
    - CameraInfo
        - Save Camera Parameters as a single *.npz file
[3] Thermal
    - 16UC1
    - Save as *.png format
        - Frame File name based on Time-stamp
    - CameraInfo
        - Save Camera Parameters as a single *.npz file
[4] Infrared
    - 8UC1
    - Save as *.png format
        - Frame File name based on Time-stamp
    - CameraInfo
        - Save Camera Parameters as a single *.npz file
[5] NightVision
    - 8UC3
    - Save as *.png format
        - Frame File name based on Time-stamp
[6] LiDAR
    - Numpy Array
    - Save as *.npy format, frame-by-frame
        - Frame File name based on Time-stamp

*** Frame Index Decision Rule
    - Use approximate synchronization rule based on time-stamp

**** [IMPORTANT] Description
    - Place this python script at the same directory to the
      target rosbag file, to be converted

    - Executing this will create a directory that contains
      converted data of each modalities

    - < In-terminal execution >
        - Control script input parameters with argparse functionality

        - One may utilize shell script to iteratively process
          for multiple rosbag files


"""
import os
import logging
import argparse
import time
import cv2
import matplotlib
import matplotlib.cm
import numpy as np

import rosbag
import ros_numpy
import pyquaternion
from cv_bridge import CvBridge


# CameraInfo Object
class camerainfo_obj(object):
    def __init__(self):
        # Distortion Matrix
        self.D = None

        # Intrinsic Camera Matrix
        self.K = None

        # Rectification Matrix (for Stereo Cameras Only)
        self.R = None

        # Projection Matrix
        self.P = None

    def update_matrix_from_msg(self, msg):
        self.D = np.array(msg.D).reshape((1, 5))
        self.K = np.array(msg.K).reshape((3, 3))
        self.R = np.array(msg.R).reshape((3, 3))
        self.P = np.array(msg.P).reshape((3, 4))


# Modal Image Object
class modal_obj(object):
    def __init__(self, modal_type, is_convert, topic_name, msg_encoding=None, camerainfo_topic_name=None):
        # Modal Type
        self.modal_type = modal_type

        # Convert FLAG
        self.is_convert = is_convert

        # Modal Topic Name
        self.topic_name = topic_name

        # Encoding Message
        self.msg_encoding = msg_encoding

        # Modal CameraInfo Topic Name
        self.camerainfo_topic_name = camerainfo_topic_name

        # List for Saving Data and Timestamp
        self.data_list, self.stamp_list = [], []

        # CameraInfo Object
        self.camerainfo = None if camerainfo_topic_name is None else camerainfo_obj()

    def __repr__(self):
        return self.modal_type

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return {"data": self.get_data(idx), "stamp": self.get_stamp(idx)}

    def append_data(self, data, stamp):
        self.data_list.append(data)
        self.stamp_list.append(stamp)

    def get_data(self, idx):
        return self.data_list[idx]

    def get_stamp(self, idx):
        return self.stamp_list[idx]

    def erase_data(self, indices_list):
        for index in sorted(indices_list, reverse=True):
            del self.data_list[index]

    def erase_stamp(self, indices_list):
        for index in sorted(indices_list, reverse=True):
            del self.stamp_list[index]


# Image-type Object
class image_modal_obj(modal_obj):
    def __init__(self, modal_type, is_convert, topic_name, msg_encoding=None, camerainfo_topic_name=None):
        super(image_modal_obj, self).__init__(modal_type, is_convert, topic_name, msg_encoding, camerainfo_topic_name)


# LiDAR Object
class lidar_modal_obj(modal_obj):
    def __init__(self, modal_type, is_convert, topic_name):
        super(lidar_modal_obj, self).__init__(modal_type, is_convert, topic_name)

        # 3D Raw PointCloud Data
        self.raw_pc_data_list = []

        # Rotation Matrix, Translation Matrix from TF Static
        self.R__color, self.T__color = None, None

    # Update [ R | T ] Matrices
    def update_tf_static_matrices(self, R, T):
        # Rotation Matrix
        self.R__color = pyquaternion.Quaternion(
            R.w, R.x, R.y, R.z
        ).rotation_matrix

        # Translation Matrix
        self.T__color = np.array([
            T.x, T.y, T.z
        ]).reshape(3, 1)


# Data Dictionary
data_dict = {
    # D435-Color
    "color": image_modal_obj(
        modal_type="color", is_convert=True, topic_name="/osr/image_color", msg_encoding="8UC3",
        camerainfo_topic_name="/osr/image_color_camerainfo"
    ),

    # D435-Depth
    "depth": image_modal_obj(
        modal_type="depth", is_convert=True, topic_name="/osr/image_aligned_depth", msg_encoding="16UC1",
        camerainfo_topic_name="/osr/image_depth_camerainfo"
    ),

    # Thermal
    "thermal": image_modal_obj(
        modal_type="thermal", is_convert=True, topic_name="/osr/image_thermal", msg_encoding="16UC1",
        camerainfo_topic_name="/osr/image_thermal_camerainfo"
    ),

    # Infrared
    "infrared": image_modal_obj(
        modal_type="infrared", is_convert=True, topic_name="/osr/image_ir", msg_encoding="8UC1",
        camerainfo_topic_name="/osr/image_infrared_camerainfo"
    ),

    # NightVision
    "nightvision": image_modal_obj(
        modal_type="nightvision", is_convert=True, topic_name="/osr/image_nv1", msg_encoding="8UC3"
    ),

    # LiDAR
    "lidar": lidar_modal_obj(
        modal_type="lidar", is_convert=True, topic_name="/osr/lidar_pointcloud"
    )
}

# Get Conversion Target Modal Objects
tmp = {}
for __modal, __data_obj in data_dict.items():
    if __data_obj.is_convert is True:
        tmp[__modal] = __data_obj
data_dict = tmp
del tmp, __modal, __data_obj


# Argument Parser Function
def argument_parser():
    # Define Argument Parser
    parser = argparse.ArgumentParser(
        prog="bag2seq.py",
        description="Python Script to Convert ROS Bag file to Approximately Synchronized Sequence Data"
    )
    subparser = parser.add_subparsers(help="Sub-parser Command")

    # Create Sub-Parsing Command for Testing this Code
    code_test_parser = subparser.add_parser("test", help="for Testing this Code")
    # code_test_parser.add_argument(
    #     "--verbose", "-V", action="store_true",
    #     help="Option for Making Script Verbose"
    # )

    # Create Sub-Parsing Command to Actually Use this Code
    code_use_parser = subparser.add_parser("convert", help="for Actually Using this Code")
    code_use_parser.add_argument(
        "--target-bag-file-name", "-T",
        type=str, help="Target Bag File Directory in Relative Path"
    )

    # Parse Arguments
    args = parser.parse_args()

    return args


# Set Logger
def set_logger(logging_level=logging.INFO):
    # Define Logger
    logger = logging.getLogger()

    # Set Logger Display Level
    logger.setLevel(level=logging_level)

    # Set Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("[%(levelname)s] | %(asctime)s : %(message)s")
    )
    logger.addHandler(stream_handler)

    return logger


def read_bag_data(bag_file_path, logger):
    # Load Bag File
    bag = rosbag.Bag(bag_file_path, "r")

    # Declare CvBridge
    bridge = CvBridge()

    # Read CameraInfo Message from Bag File
    for modal, data_obj in data_dict.items():
        # CameraInfo Break Flag
        camerainfo_break_flag = False

        # Continue to Next CameraInfo if CameraInfo is None or Conversion Flag is False
        if data_obj.camerainfo is None or data_obj.is_convert is False:
            logger.warn("Skipping CameraInfo [{}] Conversion...".format(modal))
            continue

        # Read CameraInfo
        for _, msg, _ in bag.read_messages(topics=data_obj.camerainfo_topic_name):
            if camerainfo_break_flag is True:
                break

            # Update CameraInfo Break Flag
            camerainfo_break_flag = True

            # Update Camera Parameter Matrix
            data_obj.camerainfo.update_matrix_from_msg(msg=msg)

            # CameraInfo Update Log
            logger.info("CameraInfo [{}] Updated....!".format(data_obj.camerainfo_topic_name))

    # Read TF Static ( RGB-to-Velodyne [ R | T ] Matrix )
    tf_static_break_flag = False
    for topic, msg, stamp in bag.read_messages(topics="/tf_static"):
        # Get Transforms (Color-to-LiDAR)
        tf_msg_list = msg.transforms
        for tf_msg in tf_msg_list:
            if tf_msg.child_frame_id == "velodyne_frame_from_rgb":
                data_dict["lidar"].update_tf_static_matrices(
                    R=tf_msg.transform.rotation, T=tf_msg.transform.translation
                )
                tf_static_break_flag = True
                logger.info("TF-Static Data Read for LiDAR Projection...!")

        # Break
        if tf_static_break_flag is True:
            break
    if tf_static_break_flag is False:
        logger.warn("TF-Static Data Loading Failed..!!!")

    # Log to Notify Data Reading Sequence
    logger.info("About to Read Bag Data....!")

    # Time Sleep
    time.sleep(1)

    # Read Data Message from Bag File
    for modal, data_obj in data_dict.items():
        # Continue to Next Modal Topic if Conversion Flag is False
        if data_obj.is_convert is False:
            logger.warn("Skipping Modal [{}] Conversion...".format(modal))
            continue

        # Frame Index Counter Initialization
        fidx_count = 0

        # Get Current Modal ROS Topic Name and Message Encoding
        topic_name = data_obj.topic_name
        msg_encoding = data_obj.msg_encoding

        # Time Sleep
        time.sleep(0.5)

        # Read Data
        for topic, msg, stamp in bag.read_messages(topics=topic_name):
            # Increase Frame Index Counter
            fidx_count += 1

            # Log Data Reading Status
            logger.info(
                "[READ] (Modal: {}) || Timestamp: {:.2f} || Frame Index: {}".format(
                    modal, stamp.to_sec(), fidx_count
                )
            )

            # Convert Data Message to Python-readable Data
            if data_obj.modal_type != "lidar":
                # Get Modal Data
                data = bridge.imgmsg_to_cv2(img_msg=msg, desired_encoding=msg_encoding)

                # Change BGR to RGB
                if msg.encoding.__contains__("bgr") is True:
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

                # Append Data and Stamp
                data_obj.append_data(data=data, stamp=stamp)

            # For LiDAR Data
            else:
                # Get Raw 3D PointCloud Data
                raw_pc_data = np.array(ros_numpy.point_cloud2.pointcloud2_to_array(msg).tolist())
                data_obj.raw_pc_data_list.append(raw_pc_data)

                # Project Cloud If [ R | T ] Exists
                if data_obj.R__color is not None and data_obj.T__color is not None:
                    projected_cloud =\
                        np.dot(raw_pc_data[:, 0:3], data_obj.R__color.T) + data_obj.T__color.T

                    # Filter-out Points not in-front-of Camera
                    in_range = np.where((projected_cloud[:, 0] > -8) &
                                        (projected_cloud[:, 0] < 8) &
                                        (projected_cloud[:, 1] > -5) &
                                        (projected_cloud[:, 1] < 5) &
                                        (projected_cloud[:, 2] > -0) &
                                        (projected_cloud[:, 2] < 30))
                    max_intensity = np.max(projected_cloud[:, -1])
                    valid_cloud = projected_cloud[in_range[0]]

                    # Straight Distance From Camera
                    cloud_distance = \
                        np.sqrt(valid_cloud[:, 0] * valid_cloud[:, 0] + valid_cloud[:, 1] * valid_cloud[:, 1] + valid_cloud[:, 2] * valid_cloud[:, 2])

                    # Color Map for the Points (intensity color view)
                    cmap = matplotlib.cm.get_cmap('jet')

                    cloud_colors = cmap(valid_cloud[:, -1] / max_intensity) * 255

                    # Pack Data as Dictionary
                    data = {
                        "uv_cloud": valid_cloud,
                        "cloud_distance": cloud_distance,
                        "cloud_colors": cloud_colors
                    }

                    # Append Data and Stamp
                    data_obj.append_data(data=data, stamp=stamp)

                # If Not, Skip to Next Modal Data
                else:
                    logger.warn("Since TF-STATIC is UNREADABLE, 2D Projected Cloud Data is Not Available....!")
                    time.sleep(1)
                    break


# Synchronize Multimodal Data
def synchronize_multimodal_data(dtime_thresh, logger):
    # Get Timestamp Dictionary
    modal_stamp_dict = {}
    for modal, data_obj in data_dict.items():
        modal_stamp_dict[modal] = data_obj.stamp_list

    modal_list, modal_stamp_list = [], []
    for modal, stamp_list in modal_stamp_dict.items():
        modal_list.append(modal)
        modal_stamp_list.append(np.asarray(stamp_list))

    # Asynchronous Multimodal Data Synchronization based on Time Threshold
    cp_idx = 0
    sync_modal_stamp_idx_list = []
    while cp_idx < len(modal_list) - 1:
        i_modal, i_modal_stamp_arr = modal_list[cp_idx], modal_stamp_list[cp_idx]
        j_modal, j_modal_stamp_arr = modal_list[cp_idx+1], modal_stamp_list[cp_idx+1]

        # Log
        logger.info(
            "Comparing BTW Modals <{}> and <{}>.....!".format(i_modal, j_modal)
        )

        # Init Selected Indices Lists
        selected_i_indices, selected_j_indices = [], []

        for i_idx, i_modal_stamp in enumerate(i_modal_stamp_arr):
            # Continue when Index is Previously Chosen
            if i_idx in selected_i_indices:
                continue

            for j_idx, j_modal_stamp in enumerate(j_modal_stamp_arr):
                # Continue when Index is Previously Chosen
                if j_idx in selected_j_indices:
                    continue

                # Stamp Synchronized btw 2 Modals
                if abs((i_modal_stamp - j_modal_stamp).to_sec()) <= dtime_thresh:
                    # Append to Selected Indices List
                    selected_i_indices.append(i_idx)
                    selected_j_indices.append(j_idx)

                    if cp_idx == 0:
                        sync_modal_stamp_idx_list.append([i_idx, j_idx])
                    else:
                        for m_idx, m in enumerate(sync_modal_stamp_idx_list):
                            if m[-1] == i_idx:
                                sync_modal_stamp_idx_list[m_idx].append(j_idx)
                                break

        # Increase Index
        cp_idx += 1

    # Filter-out
    filtered_sync_modal_stamp_idx_list = []
    for m in sync_modal_stamp_idx_list:
        if len(m) == len(data_dict):
            filtered_sync_modal_stamp_idx_list.append(m)
    fsms_arr = np.asarray(filtered_sync_modal_stamp_idx_list)

    # Get Deletion Index for Each Modal Data and Synchronize Data
    for modal_idx, (modal, data_obj) in enumerate(data_dict.items()):
        curr_modal_fsms_set = set(fsms_arr[:, modal_idx])
        del_indices_list = sorted(list(set(range(len(data_obj))).difference(curr_modal_fsms_set)))

        # Log
        logger.info("Synchronizing Modal [{}]".format(modal))

        # Synchronize Data
        data_obj.erase_data(indices_list=del_indices_list)
        data_obj.erase_stamp(indices_list=del_indices_list)


def save_multimodal_data(save_base_path, logger):
    # Save Synchronized Data
    for modal, data_obj in data_dict.items():
        curr_modal_data_save_path = os.path.join(save_base_path, "{}".format(modal))
        curr_modal_camera_params_save_path = os.path.join(
            save_base_path, "camera_params", "{}".format(modal)
        )
        if os.path.isdir(curr_modal_data_save_path) is False:
            raise AssertionError()
        if os.path.isdir(curr_modal_camera_params_save_path) is False:
            os.mkdir(curr_modal_camera_params_save_path)

        # Save Frame Data
        for obj_idx, data in enumerate(data_obj):
            frame_data, stamp_data = data["data"], data["stamp"]

            # If Object is an image type,
            if data_obj.modal_type != "lidar":
                # Frame File Name
                frame_filename = "{:08d}__{}.png".format(obj_idx, stamp_data.to_nsec())
                frame_filepath = os.path.join(curr_modal_data_save_path, frame_filename)

                # Image Convert if RGB
                if data_obj.modal_type == "color":
                    frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)

                # Save Image
                if os.path.isfile(frame_filepath) is False:
                    cv2.imwrite(frame_filepath, frame_data)
                    logger.info("Saving Modal [{}] ({}/{})".format(data_obj, obj_idx + 1, len(data_obj)))
                else:
                    logger.warn("Modal [{}] | File {} Already Exists...!".format(data_obj, frame_filename))

                # Save Camera Params
                if data_obj.camerainfo is not None:
                    np.save(os.path.join(curr_modal_camera_params_save_path, "D.npy"), data_obj.camerainfo.D)
                    np.save(os.path.join(curr_modal_camera_params_save_path, "K.npy"), data_obj.camerainfo.K)
                    np.save(os.path.join(curr_modal_camera_params_save_path, "P.npy"), data_obj.camerainfo.P)
                    np.save(os.path.join(curr_modal_camera_params_save_path, "R.npy"), data_obj.camerainfo.R)

            # If Object is a lidar type,
            else:
                # PointCloud File Name
                lidar_pc_filename = "{:08d}__{}.npy".format(obj_idx, stamp_data.to_nsec())
                lidar_pc_filepath = os.path.join(curr_modal_data_save_path, lidar_pc_filename)

                # Save PointCloud Data as Numpy Array (npy format)
                if os.path.isfile(lidar_pc_filepath) is False:
                    np.save(lidar_pc_filepath, frame_data)
                    logger.info("Saving Modal [{}] ({}/{})".format(data_obj, obj_idx + 1, len(data_obj)))
                else:
                    logger.warn("Modal [{}] | File {} Already Exists...!".format(data_obj, lidar_pc_filename))

                # Save Camera Params
                np.save(os.path.join(curr_modal_camera_params_save_path, "R__color.npy"), data_obj.R__color)
                np.save(os.path.join(curr_modal_camera_params_save_path, "T__color.npy"), data_obj.T__color)

        # Pause
        time.sleep(1)


# Main Function
def main():
    # Argument Parser
    args = argument_parser()

    # Set Logger
    logger = set_logger(logging_level=logging.INFO)

    # Time Sleep
    time.sleep(1)

    # Get Conversion Target ROS Bag File Path
    if hasattr(args, "target_bag_file_name") is True:
        # Check if Target Bag File Exists in Current Patch
        bag_file_name = args.target_bag_file_name
        bag_file_path = os.path.join(os.path.dirname(__file__), bag_file_name)
        logger.info("Located Bag File at {}".format(bag_file_path))
        if os.path.isfile(bag_file_path):
            # Make Conversion Folder
            cvt_folder_path = os.path.join(
                os.path.dirname(bag_file_path),
                "_cvt_data__[{}]".format(bag_file_name.split(".")[0])
            )
            if os.path.isdir(cvt_folder_path) is False:
                os.mkdir(cvt_folder_path)
                logger.info("Conversion Folder Generated...! (PATH: {})".format(cvt_folder_path))
            else:
                logger.warn("Conversion Folder Already Exists..! (PATH: {})".format(cvt_folder_path))
        else:
            raise AssertionError("Bag File Does Not Exist at Path: {}".format(bag_file_path))
    else:
        # Test Bag File Path
        bag_file_path = os.path.join(os.path.dirname(__file__), "test", "test.bag")
        logger.info("Located Bag File at {}".format(bag_file_path))
        if os.path.isfile(bag_file_path):
            # Make Conversion Folder
            cvt_folder_path = os.path.join(os.path.dirname(bag_file_path), "_cvt_data__[test]")
            if os.path.isdir(cvt_folder_path) is False:
                os.mkdir(cvt_folder_path)
                logger.info("Conversion Folder Generated...! (PATH: {})".format(cvt_folder_path))
            else:
                logger.warn("Conversion Folder Already Exists..! (PATH: {})".format(cvt_folder_path))
        else:
            raise AssertionError("Bag File Does Not Exist at Path: {}".format(bag_file_path))

    # Time Sleep
    time.sleep(1)

    # Make Current Bag File Camera Parameter Save Path
    camera_param_save_path = os.path.join(cvt_folder_path, "camera_params")
    if os.path.isdir(camera_param_save_path) is False:
        os.mkdir(camera_param_save_path)

    # Log Target Modal Type to Convert
    for modal, data_obj in data_dict.items():
        if data_obj.is_convert is True:
            # Log
            logger.info(
                "Modal '{}' (topic name: {}) pending for Conversion... [CameraInfo Topic Name: {}]".format(
                    data_obj, data_obj.topic_name, data_obj.camerainfo_topic_name
                )
            )

            # Make Current Modal Save Path
            curr_modal_save_path = os.path.join(cvt_folder_path, modal)
            if os.path.isdir(curr_modal_save_path) is False:
                os.mkdir(curr_modal_save_path)

    # Time Sleep
    time.sleep(1)

    # Read Bag File Data
    read_bag_data(bag_file_path=bag_file_path, logger=logger)

    # Synchronize Multimodal Data
    synchronize_multimodal_data(dtime_thresh=1.0/50, logger=logger)

    # Save Data
    save_multimodal_data(save_base_path=cvt_folder_path, logger=logger)

    pass


# Main Namespace
if __name__ == '__main__':
    main()
