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
        "curr_script": os.path.dirname(__file__),

        # Bag File Root Path
        # "bag_file_base_path": "/home/kyle/bag_files_temp/candidate_bags/0926_snu_result/",
        "bag_file_base_path": "./",

        # Bag File Name
        # "bag_file": "2019-09-27-10-15-00.bag",

        # "bag_file": "kiro_all.bag",
        # "bag_file": "2019-09-26-17-19-40.bag",
        # "bag_file": "2019-09-27-(1).bag",
        # "bag_file": "2019-09-27-(2).bag",

        "bag_file": "2019-11-28-11-55-42_movig_agent_rainy_day.bag",


        # Bag Files (use this format when unbagging multiple bag files in a loop)
        # "bag_files": {
        #     "moving_agent": "2019-09-26-17-19-40.bag",
        #     "depth_exp_not_moving": "2019-09-27-10-15-00.bag",
        #     "depth_exp_moving": "2019-09-27-10-20-12.bag",
        # },
    },

    # Image topics
    # "image_topics": {
    #     "rgb_image": "/osr/snu_result_image",
    #     "thermal_image": "/osr/image_thremal",
    #     # "depth_image": "/osr/image_depth",
    #     # "aligned_depth_image": "/osr/image_aligned_depth",
    # },

    # Camera Parameter Topics
    "cam_param_topics": {
        # "rgb_camera_params": "/osr/image_color_camerainfo",
        # "depth_camera_params": "/osr/image_depth_camerainfo",
    },

    # Image topics (for kiro_all.bag)
    "image_topics": {
        "rgb_image": "/osr/image_color",
        "thermal_image": "/osr/image_thremal",
        # "depth_image": "/osr/image_depth",
    },
    #
    # # Camera Parameter Topics (for kiro_all.bag)
    # "cam_param_topics": {
    #     "rgb_camera_params": "/camera/color/camera_info",
    #     "depth_camera_params": "/camera/color/camera_info",
    # },

    # ETC Options
    "ETC": {},
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
    folder_name = "__image_sequence__[BAG_FILE]_[" + opts_dict["paths"]["bag_file"].split(".")[0] + "]"
    output_save_dir = os.path.join(opts_dict["paths"]["bag_file_base_path"], folder_name)
    if os.path.isdir(output_save_dir) is False:
        os.mkdir(output_save_dir)

    # Read Message from bag file by Camera Parameter Topics
    for modal_cam_param_type, topic_name in opts_dict["cam_param_topics"].items():
        print '[' + str(loop_count) + ''
        # Camera Parameter Save Message
        print "Saving camera parameters: [%s]" % modal_cam_param_type

        # For Loop Count (inside bag file)
        loop_count = 0

        # Set Save File base name (need to save each respective parameters)
        file_base_name = "%s__" % modal_cam_param_type

        # Set Sub-directory for camera parameters
        if modal_cam_param_type.__contains__("rgb") is True:
            output_cam_param_save_dir = os.path.join(output_save_dir, "rgb_cam_params")
            if os.path.isdir(output_cam_param_save_dir) is False:
                os.mkdir(output_cam_param_save_dir)
        elif modal_cam_param_type.__contains__("depth") is True:
            output_cam_param_save_dir = os.path.join(output_save_dir, "depth_cam_params")
            if os.path.isdir(output_cam_param_save_dir) is False:
                os.mkdir(output_cam_param_save_dir)
        else:
            assert 0, "Unknown Camera modal Parameter!"

        # Topic-wise bag file read
        for topic, msg, t in bag.read_messages(topics=topic_name):
            # Break after looped once
            if loop_count > 0:
                continue

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
    for modal, topic_name in opts_dict["image_topics"].items():
        # Frame Index Count Init
        fidx_count = 0

        # Set Current Modal Save Sub-directory (make directory if not exist)
        modal_save_dir = os.path.join(output_save_dir, modal)
        # print(modal_save_dir)
        if os.path.isdir(modal_save_dir) is False:
            os.mkdir(modal_save_dir)

        if modal.__contains__("depth") is True:
            if modal.__contains__("aligned") is True:
                # Aligned
                depth_file_path = os.path.join(modal_save_dir, "aligned_depth_ndarray.npz")
            else:
                # Raw
                depth_file_path = os.path.join(modal_save_dir, "depth_ndarray.npz")
        else:
            depth_file_path = None

        # List for npz data (init)
        npz_data = []

        # Topic-wise bag file read
        for topic, msg, t in bag.read_messages(topics=topic_name):
            # print '[' + str(fidx_count) + ']'
            # Increase Frame Index Count
            fidx_count += 1

            # Convert 'Time' Instance to Seconds
            stamp_secs = t.to_sec()

            # Convert Image Message to cv2
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Save Image (flag ==>> Test)
            if is_save is True:
                # Save Message
                print "Saving modal [%s]...{fidx: %09d}" % (modal, fidx_count)

                # --> RGB : save as png format
                #     Depth : save as numpy "*.npz" file (save as ndarray)
                # if modal.__contains__("rgb") is True:
                # (from bag file)
                # frame = frame[:, :, [2, 1, 0]]

                curr_frame_file_path = os.path.join(modal_save_dir, "%09d.png" % fidx_count)
                print(curr_frame_file_path)
                set_file.write("%09d\n" % fidx_count)
                if os.path.exists(curr_frame_file_path) is False:
                    cv2.imwrite(curr_frame_file_path, frame)
                else:
                    print('error: save img')

                # elif modal.__contains__("depth") is True:
                #     # Store image frame to npz_data list
                #     npz_data.append(frame)
                # else:
                # assert 0, "Unexpected Modal %s" % modal

        # For Depth Image, save npz data
        if is_save is True:
            if modal.__contains__("depth"):
                if modal.__contains__("aligned") is True:
                    print "Processing 'npz' file for aligned depth...!"
                else:
                    print "Processing 'npz' file for raw depth...!"

                if os.path.exists(depth_file_path) is False:
                    np.savez(depth_file_path, *npz_data)

        # Sleep for modal change
        time.sleep(1)
    
    set_file.close()
    # Close bag file
    bag.close()


# Main Function
def main():
    # Iterate through all bag files in the dictionary

    # Adjoin Bag File Path
    bag_file_path = os.path.join(opts_dict["paths"]["bag_file_base_path"], opts_dict["paths"]["bag_file"])

    # bag2imseq
    bag2imseq(bag_file_path)


# Main Namespace
if __name__ == '__main__':
    main()
