#!/usr/bin/env python
"""
Extract Images from rosbag file

* About Camera Parameter ROS Message
For Camera Parameter Specifics, refer to the following link
[LINK] : http://docs.ros.org/kinetic/api/sensor_msgs/html/msg/CameraInfo.html

** About Bag Files (GOOGLE DRIVE)
    --> Needs permission (please ask)
[LINK] https://drive.google.com/drive/folders/1OOsroxrmmQB5cKO2sTt8eM8SfJswmJ9d


"""
import os
import time
import cv2
import numpy as np

import rosbag
import rospy
from cv_bridge import CvBridge


# Options Dictionary
opts_dict = {
    # Target Paths, bag file name, etc.
    "paths": {
        # Current File's Directory
        "curr_script_dir": os.path.dirname(__file__),

        # Rosbag File Base Directory
        "rosbag_file_base_dir": "/mnt/usb-USB_3.0_Device_0_000000004858-0:0-part1/",
        
        # Rosbag File Name
        "rosbag_filename": "190823_kiro_lidar_camera_calib.bag",
    },

    # Image Topics
    "image_topics": {
        "color": {
            "topic_name": "/osr/image_color",
            "msg_encoding": "8UC3",
        },

        "disparity": {
            "topic_name": "/osr/image_depth",
            "msg_encoding": "16UC1",
        },

        "thermal": {
            "topic_name": "/osr/image_thermal",
            "msg_encoding": "16UC1",
        },

        "infrared": {
            "topic_name": "/osr/image_ir",
            "msg_encoding": "8UC1",
        },

        "nightvision": {
            "topic_name": "/osr/image_nv1",
            "msg_encoding": "8UC3",
        },

        "lidar": {
            "topic_name": "/camera_lidar",
            "msg_encoding": "8UC3",
        },
    },

    # Camera Parameter Topics
    "cam_param_topics": {
        "color": "/camera/color/camera_info",
        "disparity": "/camera/depth/camera_info",
    },
}

# Save Flag (for code test)
is_save = True


# bag2imseq
def bag2imseq(bag_file_path):
    # Bag File Message
    print "Bag File Loaded!....(file from: %s)" % bag_file_path
    time.sleep(3)

    # Read Bag File
    bag = rosbag.Bag(bag_file_path, "r")

    # Declare CvBridge
    bridge = CvBridge()

    # Set Current bag imseq save directory (make directory if not exist)
    folder_name = "__image_sequence__[BAG_FILE]_[" + opts_dict["paths"]["rosbag_filename"].split(".")[0] + "]"
    output_save_dir = os.path.join(opts_dict["paths"]["rosbag_file_base_dir"], folder_name)
    if os.path.isdir(output_save_dir) is False:
        os.mkdir(output_save_dir)

    # Read Message from bag file by Camera Parameter Topics
    for modal_cam_param_type, topic_name in opts_dict["cam_param_topics"].items():
        # Camera Parameter Save Message
        print "Saving camera parameters: [%s]" % modal_cam_param_type

        # For Loop Count (inside bag file)
        loop_count = 0

        # Set Save File base name (need to save each respective parameters)
        file_base_name = "%s__" % modal_cam_param_type

        # Set Sub-directory for camera parameters
        if modal_cam_param_type.__contains__("color") is True:
            output_cam_param_save_dir = os.path.join(output_save_dir, "color_cam_params")
            if os.path.isdir(output_cam_param_save_dir) is False:
                os.mkdir(output_cam_param_save_dir)
        elif modal_cam_param_type.__contains__("disparity") is True:
            output_cam_param_save_dir = os.path.join(output_save_dir, "disparity_cam_params")
            if os.path.isdir(output_cam_param_save_dir) is False:
                os.mkdir(output_cam_param_save_dir)
        else:
            assert 0, "Unknown Camera modal Parameter!"

        # Topic-wise bag file read
        for topic, msg, t in bag.read_messages(topics=topic_name):
            # Break after looped once
            if loop_count > 0:
                break

            # Distortion Parameter
            D = np.array(msg.D).reshape((1, 5))
            distortion_filename = file_base_name + "[D].npy"

            # Intrinsic Camera Matrix
            K = np.array(msg.K).reshape((3, 3))
            intrinsic_filename = file_base_name + "[K].npy"

            # Rectification Matrix (for stereo cameras only)
            R = np.array(msg.R).reshape((3, 3))
            rectification_filename = file_base_name + "[R].npy"

            # Projection Matrix
            P = np.array(msg.P).reshape((3, 4))
            projection_filename = file_base_name + "[P].npy"

            if is_save is True:
                np.save(os.path.join(output_cam_param_save_dir, distortion_filename), D)
                np.save(os.path.join(output_cam_param_save_dir, intrinsic_filename), K)
                np.save(os.path.join(output_cam_param_save_dir, rectification_filename), R)
                np.save(os.path.join(output_cam_param_save_dir, projection_filename), P)

            # Increase Loop Count
            loop_count += 1

    # Read Message from bag file by Image Topics
    set_file_path = os.path.join(output_save_dir, "video.txt")
    set_file = open(set_file_path, "w")

    for modal, modal_dict in opts_dict["image_topics"].items():
        # Frame Index Count Init
        fidx_count = 0

        # Get Current Modal ROS Topic Name and Message Encoding
        topic_name = modal_dict["topic_name"]
        msg_encoding = modal_dict["msg_encoding"]

        # Set Current Modal Save Sub-directory
        modal_save_dir = os.path.join(output_save_dir, modal)
        if os.path.isdir(modal_save_dir) is False:
            os.mkdir(modal_save_dir)
        else:
            print("Image Sequence of Modal [%s] already exists..! (Proceeding to next Modal)" % modal)
            continue

        # List for *.npz data (initialization)
        npz_data = []

        # Topic-wise Bag File Read
        for topic, msg, t in bag.read_messages(topics=topic_name):
            # Increase Frame Index Count
            fidx_count += 1

            # Convert 'Time' to Seconds
            stamp_secs = t.to_sec()

            # Convert Image Message to Numpy Array (to use OpenCV functions)
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding=msg_encoding)

            # Save Frame Images
            if msg_encoding not in ["8UC1", "8UC3"]:
                npz_data.append(frame)

            curr_frame_file_path = os.path.join(modal_save_dir, "%09d__time[%s].png" % (fidx_count, str(stamp_secs)))
            set_file.write("%09d\n" % fidx_count)
            if os.path.exists(curr_frame_file_path) is False:
                if fidx_count % 10 == 0:
                    frame_save_msg = \
                        "Saving Modal [%s] frame (fidx: %09d)" % (modal, fidx_count)
                    print(frame_save_msg)
            else:
                assert 0, "This case should not happen!"

            if is_save is True:
                cv2.imwrite(curr_frame_file_path, frame)

        # Save Modal Frames
        # For Non-uint8 Modals, directly store numpy array as a file
        if msg_encoding not in ["8UC1", "8UC3"]:
            # Set Non-uint8 modal file names
            modal_file_path = os.path.join(
                modal_save_dir, "%s_ndarray.npz" % modal
            )
            if os.path.exists(modal_file_path) is False:
                if is_save is True:
                    np.savez(modal_file_path, *npz_data)
            else:
                print("Modal [%s] npz file for file [%s] already exists...! Skipping and Trying next modal..!" % (modal, opts_dict["paths"]["rosbag_filename"].split(".")[0]))
                time.sleep(2)
                continue

        # Sleep for Modal Change
        time.sleep(3)

    set_file.close()
    bag.close()


# Main Function
def main():
    # Parse Modals and Topics First (print)
    for modal, modal_dict in opts_dict["image_topics"].items():
        topic_print_msg = "Modal [%s] TOPIC NAME: ( %s )" % (modal, modal_dict["topic_name"])
        print(topic_print_msg)
    for modal, camerainfo_topic_name in opts_dict["cam_param_topics"].items():
        topic_print_msg = "Camera Info of Modal [%s] TOPIC NAME: ( %s )" % (modal, camerainfo_topic_name)
        print(topic_print_msg)

    # Sleep
    time.sleep(5)

    # Adjoin Bag File Path
    bag_file_path = os.path.join(opts_dict["paths"]["rosbag_file_base_dir"], opts_dict["paths"]["rosbag_filename"])

    # bag2imseq
    bag2imseq(bag_file_path)


# Main Namespace
if __name__ == '__main__':
    main()
