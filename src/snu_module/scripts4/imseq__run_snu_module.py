#!/usr/bin/env python
"""
SNU Integrated Module v5.0

    - ROS-embedded Code Version

        - [1] Outdoor Surveillance Robot Agents

            - Static (fixed)
            - Dynamic (moving)

        - [2] ROS Bag File


"""
import os
import argparse
import time
import yaml
import rospy
import logging
import tf2_ros
import numpy as np


import utils.loader
from utils.ros.base import backbone
# from utils.ros.sensors import snu_SyncSubscriber
# import snu_visualizer
# from snu_algorithms_v4_5 import snu_algorithms
# from utils.profiling import Timer
#
# from module_detection import load_model as load_det_model
# from module_action import load_model as load_acl_model


# Run Mode
RUN_MODE = "bag"


# Define SNU Module Class
class snu_module(backbone):
    def __init__(self, logger, opts):
        super(snu_module, self).__init__(
            opts=opts, is_sensor_param_file=self.sensor_param_file_check()
        )

        # Initialize Logger Variable
        self.logger = logger

        # Initialize Frame Index
        self.fidx = 0

        # Synchronized Timestamp of Multimodal Sensors
        self.sync_stamp = None

        # Declare SNU Visualizer
        self.visualizer = snu_visualizer.visualizer(opts=opts)

        # Declare ROS Synchronization Switch Dictionary
        self.ros_sync_switch_dict = {
            "color": opts.sensors.color["is_valid"],
            "disparity": False, "aligned_disparity": opts.sensors.disparity["is_valid"],
            "thermal": opts.sensors.thermal["is_valid"],
            "infrared": opts.sensors.infrared["is_valid"],
            "nightvision": opts.sensors.lidar["is_valid"],
        }

    @staticmethod
    def sensor_param_file_check():
        sensor_params_path = os.path.join(os.path.dirname(args.config), "sensor_params")
        if os.path.isdir(sensor_params_path):
            return True
        else:
            return False

    def gather_all_sensor_params_via_files(self):
        config_file_number = int(args.config.split("/")[-1].split(".")[0])
        sensor_params_path = os.path.join(os.path.dirname(args.config), "sensor_params")
        sensor_param_filenames = os.listdir(sensor_params_path)

        for sensor_param_filename in sensor_param_filenames:
            modal = sensor_param_filename.split(".")[0]

            # Get Sensor Parameters
            sensor_param_filepath = os.path.join(sensor_params_path, sensor_param_filename)
            with open(sensor_param_filepath, "r") as stream:
                tmp = yaml.safe_load(stream=stream)
            sensor_param_array = np.asarray(tmp["STATIC_{:02d}".format(config_file_number)]["camera_param"])

            # Update Sensor Parameter
            modal_obj = getattr(self, modal)
            modal_obj.update_sensor_params_file_array(sensor_param_array=sensor_param_array)

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
        snu_usr = snu_algorithms(frameworks=frameworks, opts=self.opts)
        self.logger.info("SNU Algorithm Loaded...!")
        time.sleep(0.01)

        # ROS Node Initialization
        self.logger.info("ROS Node Initialization")
        rospy.init_node(name=module_name, anonymous=True)

        if self.opts.agent_type == "dynamic" or self.opts.agent_type == "rosbagfile":
            # Subscribe for tf_static
            tf_buffer = tf2_ros.Buffer()
            tf_listener = tf2_ros.TransformListener(buffer=tf_buffer)

            # Iterate Loop until "tf_static is heard"
            while self.tf_transform is None:
                try:
                    self.tf_transform = tf_buffer.lookup_transform(
                        "rgb_frame", 'velodyne_frame_from_rgb', rospy.Time(0)
                    )
                except:
                    rospy.logwarn("SNU-MODULE : TF_STATIC Transform Unreadable...!")
                    continue

        elif self.opts.agent_type == "static":
            self.tf_transform = None
            self.gather_all_sensor_params_via_files()

        # Load ROS Synchronized Subscriber
        rospy.loginfo("Load ROS Synchronized Subscriber...!")
        sync_ss = snu_SyncSubscriber(
            ros_sync_switch_dict=self.ros_sync_switch_dict, options=self.opts
        )

        # ROS Loop Starts
        rospy.loginfo("Starting SNU Integrated Module...!")
        try:
            while not rospy.is_shutdown():
                loop_timer = Timer(convert="FPS")
                loop_timer.reset()

                # Make Synchronized Data
                sync_ss.make_sync_data()

                # Get Synchronized Data, Loop Until Synchronized
                sync_data = sync_ss.get_sync_data()
                if sync_data is None:
                    continue
                else:
                    self.update_all_modal_data(sync_data=sync_data)
                self.sync_stamp = sync_data[0]
                sensor_fps = loop_timer.elapsed

                # Increase Frame Index
                self.fidx += 1

                # Update Sensor Image Frame Size
                if self.fidx == 1:
                    self.opts.sensors.update_sensor_image_size(
                        frame=self.color.get_data()
                    )

                # Gather All Data and Process Disparity Frame
                sync_data_dict = self.gather_all_modal_data()
                sync_data_dict["disparity"].process_data(self.opts.sensors.disparity)

                # SNU USR Integrated Algorithm Call
                trajectories, detections, fps_dict = snu_usr(
                    sync_data_dict=sync_data_dict, fidx=self.fidx
                )

                # Algorithm Total FPS
                total_fps = loop_timer.elapsed

                # Log Profile
                rospy.loginfo(
                    "FIDX: {} || # of Trajectories: <{}> || Total SNU Module Speed: {:.2f}fps".format(
                        self.fidx, len(snu_usr), total_fps
                    )
                )
                # rospy.loginfo("FIDX: {} || # of Tracklets: <{}> || [SENSOR: {:.2f}fps | DET: {:.1f}fps | TRK: {:.1f}fps | ACL: {:.1f}fps]".format(
                #     self.fidx, len(snu_usr), sensor_fps, fps_dict["det"], fps_dict["trk"], fps_dict["acl"]
                #     )
                # )

                # Draw Results
                result_frame_dict = self.visualizer(
                    sensor_data=self.color, trajectories=trajectories, detections=detections, fidx=self.fidx
                )

                # Publish Tracks
                self.publish_tracks(trajectories=trajectories, odometry_msg=self.odometry_msg)

                # Publish SNU Result Image Results
                self.publish_snu_result_image(result_frame_dict=result_frame_dict)

                # Draw / Show / Publish Top-view Result
                if self.opts.visualization.top_view["is_draw"] is True:
                    self.visualizer.visualize_top_view_trajectories(trajectories=trajectories)

                    # # Publish Top-view Result
                    # self.top_view_result_pub.publish(
                    #     self.pub_bridge.cv2_to_imgmsg(
                    #         self.visualizer.top_view_map, "rgb8"
                    #     )
                    # )

            # Rospy Spin
            rospy.spin()

        except KeyboardInterrupt:
            rospy.logwarn("ShutDown SNU Module...!")


def main():
    # Set Logger
    logger = utils.loader.set_logger(logging_level=logging.INFO)

    # Argument Parser
    args = utils.loader.argument_parser(logger=logger, dev_version=4.5, mode_selection=RUN_MODE)

    # Load Configuration from File
    cfg = utils.loader.cfg_loader(logger=logger, args=args)

    # Load Options
    opts = utils.loader.load_options(logger=logger, args=args, cfg=cfg)
    opts.visualization.correct_flag_options()

    # Initialize SNU Module
    ros_snu_usr = snu_module(logger=logger, opts=opts)

    # Run SNU Module
    ros_snu_usr(module_name="snu_module")


if __name__ == "__main__":
    main()
