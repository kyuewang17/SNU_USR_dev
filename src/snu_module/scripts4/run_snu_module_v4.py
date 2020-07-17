#!/usr/bin/env python
"""
SNU Integrated Module v4.0

- Massive Changes
    - Multimodal Data Synchronization Method Changed
    - (Code received from KIRO)


"""
import os
import time
import argparse
import rospy
import logging
import tf2_ros

import options_v4 as options
import snu_visualizer
import ros_utils_v4 as ros_utils
import snu_algorithms_v4 as snu_algorithms
from utils.profiling import Timer

from module_detection import load_model as load_det_model
from module_action import load_model as load_acl_model

from config import cfg


# Argument Parser
parser = argparse.ArgumentParser(description="SNU Integrated Algorithm", prog="SNU-Integrated-v4.0")
parser.add_argument(
    "--config", "-C",
    default=os.path.join(os.path.dirname(__file__), "config", "curr_agent", "config.yaml"),
    # default=os.path.join(os.path.dirname(__file__), "config", "190823_kiro_lidar_camera_calib.yaml"),
    type=str, help="Configuration YAML file"
)
args = parser.parse_args()


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


# Define SNU Module Class
class snu_module(ros_utils.coverage):
    def __init__(self, opts):
        # Load Options
        super(snu_module, self).__init__(
            opts=opts, is_sensor_param_file=self.sensor_parameter_file_check()
        )

        # Initialize Logger Variable
        self.logger = set_logger(logging_level=logging.INFO)

        # Initialize Frame Index
        self.fidx = 0

        # Synchronized Timestamp of Multimodal Sensors
        self.sync_stamp = None

        # Declare SNU Visualizer
        self.visualizer = snu_visualizer.visualizer(opts=opts)

        # Declare ROS Synchronization Switch Dictionary
        self.ros_sync_switch_dict = {
            "color": True,
            "disparity": False, "aligned_disparity": True,
            "thermal": True,
            "infrared": True,
            "nightvision": True,
        }

    def sensor_parameter_file_check(self):
        return False

    def gather_all_sensor_parameters_via_files(self):
        raise NotImplementedError()

    # Call as Function
    def __call__(self, module_name):
        # Load Detection and Action Classification Models
        frameworks = {
            "det": load_det_model(opts=self.opts),
            "acl": load_acl_model(opts=self.opts),
        }
        self.logger.info("Detector and Action Classifier Neural Network Model Loaded...!")
        time.sleep(0.01)

        # Initialize SNU Algorithm Class
        snu_usr = snu_algorithms.snu_algorithms(frameworks=frameworks, opts=self.opts)
        self.logger.info("SNU Algorithm Loaded...!")
        time.sleep(0.01)

        # ROS Node Initialization
        self.logger.info("ROS Node Initialization")
        rospy.init_node(name=module_name, anonymous=True)

        # Subscribe for tf_static
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(buffer=tf_buffer)

        while self.tf_transform is None:
            try:
                self.tf_transform = tf_buffer.lookup_transform(
                    'rgb_frame', 'velodyne_frame_from_rgb', rospy.Time(0)
                )

            except:
                rospy.logwarn("SNU-MODULE : TF_STATIC Transform Unreadable...!")
                continue

        # Load ROS Synchronized Subscriber
        print("Load ROS Synchronized Subscriber")
        sync_ss = ros_utils.snu_SyncSubscriber(
            ros_sync_switch_dict=self.ros_sync_switch_dict, options=self.opts
        )

        # ROS Loop Starts (Start SNU Integrated Module)
        print("Starting SNU Integrated Module...!")
        try:
            while not rospy.is_shutdown():
                # Make Synchronized Data
                sync_ss.make_sync_data()

                # Get Synchronized Data, Timestamp with Extrinsic Transformation
                sync_data = sync_ss.get_sync_data()
                if sync_data is None:
                    continue
                else:
                    self.update_all_modal_data(sync_data=sync_data)
                self.sync_stamp = sync_data[0]

                # Increase Frame Index
                self.fidx += 1
                rospy.loginfo("FIDX: {}".format(self.fidx))

                # # Visualize Projected LiDAR Data
                # self.lidar.load_pc_xyz_data()
                # lidar_to_color_sensor_data = self.lidar + self.color
                # lidar_to_color_sensor_data.visualize_frame()
                #
                # Gather All Data and Process Disparity Frame
                sync_data_dict = self.gather_all_modal_data()
                sync_data_dict["disparity"].process_data(self.opts.sensors.disparity)

                # SNU USR Integrated Algorithm Call
                tracklets, detections, fps_dict = snu_usr(
                    sync_data_dict=sync_data_dict,
                    logger=self.logger, fidx=self.fidx
                )

                # # Draw Color Image Sequence
                # # self.visualizer.visualize_modal_frames(self.color)
                # self.visualizer.visualize_modal_frames_with_calibrated_pointcloud(
                #     sensor_data=self.color, pc_img_coord=projected_data, color=lidar_color
                # )

                # Draw Results
                result_frame_dict = self.visualizer(
                    sensor_data=self.color, tracklets=tracklets, detections=detections, fidx=self.fidx
                )

                # Publish Tracks
                self.publish_tracks(tracklets=tracklets, odometry_msg=self.odometry_msg)

                # Publish SNU Result Image Results
                self.publish_snu_result_image(result_frame_dict=result_frame_dict)

                # # Rospy Sleep (NOT REQUIRED)
                # rospy.sleep(0.1)

            # Rospy Spin
            rospy.spin()

        except KeyboardInterrupt:
            print("Shutdown SNU Module...!")


def main():
    # Load Configuration FIle
    cfg.merge_from_file(args.config)

    # Load Options
    opts = options.snu_option_class(cfg=cfg)

    # Initialize SNU Module
    snu_usr = snu_module(opts=opts)

    # Run SNU Module
    snu_usr(module_name="snu_module")


if __name__ == "__main__":
    main()
