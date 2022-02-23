#!/usr/bin/env python
"""
SNU Integrated Module

    - ROS-embedded Code Version

        - [1] Outdoor Surveillance Robot Agents

            - Static (fixed)
            - Dynamic (moving)

        - [2] ROS Bag File

"""
import os
import copy
import argparse
import time
import yaml
import rospy
import logging
import tf2_ros
import numpy as np
from datetime import datetime

import utils.loader
from registration.tf_object import TF_TRANSFORM
from utils.profiling import Timer
from utils.ros.base import backbone
from utils.ros.sensors import snu_SyncSubscriber
# import snu_visualizer_old as snu_visualizer
import snu_visualizer
from module_bridge import snu_algorithms

# Run Mode (choose btw ==>> bag / imseq / agent)
RUN_MODE = "bag"


# Define SNU Module Class
class snu_module(backbone):
    def __init__(self, logger, opts):
        super(snu_module, self).__init__(opts=opts)

        # Initialize Logger Variable
        self.logger = logger

        # Initialize Frame Index
        self.fidx = 0

        # Initialize Loop Timer
        self.loop_timer = Timer(convert="FPS")

        # Initialize TF Transform
        # NOTE: Tentative Code for testing...!
        self.opts.agent_type = "dynamic"
        orig_agent_id = copy.deepcopy(self.opts.agent_id)
        self.opts.agent_id = "06"

        if opts.agent_type == "dynamic":
            self.tf_transform = TF_TRANSFORM(opts=self.opts)
        else:
            self.tf_transform = None
        self.tf2_transform = None
        self.opts.agent_id = orig_agent_id

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
            "nightvision": opts.sensors.nightvision["is_valid"],
        }

    def gather_all_sensor_params_via_files(self):
        # Get Sensor Parameter File Path
        if self.opts.agent_type in ["static", "imseq"]:
            if self.opts.env_type == "imseq":
                raise NotImplementedError()
                # sensor_params_path = os.path.join(os.path.dirname(__file__), "configs", "imseq", "sensor_params")
            else:
                sensor_params_path = os.path.join(os.path.dirname(__file__), "configs", "agents", "static", "sensor_params")
            if os.path.isdir(sensor_params_path) is True:
                # Collect List of Sensor Parameter for Each Modality
                sensor_param_filenames = os.listdir(sensor_params_path)

                for sensor_param_filename in sensor_param_filenames:
                    modal_type = sensor_param_filename.split(".")[0]

                    # Get Sensor Parameters from YAML file
                    sensor_param_filepath = os.path.join(sensor_params_path, sensor_param_filename)
                    with open(sensor_param_filepath, "r") as stream:
                        tmp = yaml.safe_load(stream=stream)

                    if self.opts.env_type in ["static", "dynamic"]:
                        sensor_param_array = np.asarray(tmp["STATIC_{:02d}".format(int(self.opts.agent_id))]["camera_param"])
                    else:
                        raise NotImplementedError()

                    # Update Sensor Parameter
                    modal_obj = getattr(self, modal_type)
                    modal_obj.update_sensor_params_file_array(sensor_param_array=sensor_param_array)
        else:
            rospy.loginfo("Sensor Parameter Directory Not Found...!")

    # Call as Function
    def __call__(self, module_name, force_set_time=None):
        # Initialize SNU Algorithm Class
        snu_usr = snu_algorithms(opts=self.opts)
        self.logger.info("SNU Algorithm and Neural Network Models Loaded...!")
        time.sleep(0.01)

        # ROS Node Initialization
        self.logger.info("ROS Node Initialization")
        rospy.init_node(name=module_name, anonymous=True)

        # Check for Sensor Parameter Files
        rospy.loginfo("Checking Sensor Parameter Directory...!")
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
                self.loop_timer.reset()

                # Update Time About in 1 minute period
                if self.fidx % 600 == 0 and self.fidx > 0:
                    if self.opts.sunrise <= datetime.now() <= self.opts.sunset:
                        self.opts.time = "day"
                    else:
                        self.opts.time = "night"
                self.opts.time = force_set_time if force_set_time is not None else self.opts.time
                # print(force_set_time)

                # Make Synchronized Data
                sync_ss.make_sync_data()

                # Get Synchronized Data, Loop Until Synchronized
                sync_data = sync_ss.get_sync_data()
                if sync_data is None:
                    # print("LOOPING...!")
                    continue
                else:
                    self.update_all_modal_data(sync_data=sync_data)
                self.sync_stamp = sync_data[0]
                sensor_fps = self.loop_timer.elapsed

                # Increase Frame Index
                self.fidx += 1

                # Update Sensor Image Frame Size
                if self.fidx == 1:
                    self.opts.sensors.update_sensor_image_size(
                        frame=self.color.get_data()
                    )

                # Gather All Data and Process Disparity Frame
                sync_data_dict = self.gather_all_modal_data()
                if self.disparity is not None:
                    sync_data_dict["disparity"].process_data(self.opts.sensors.disparity)

                # SNU USR Integrated Algorithm Call
                trajectories, detections, heatmap, fps_dict = snu_usr(
                    sync_data_dict=sync_data_dict, fidx=self.fidx
                )

                # Algorithm Total FPS
                total_fps = self.loop_timer.elapsed

                # Log Profile
                # rospy.loginfo(
                #     "FIDX: {} || # of Trajectories: <{}> || Total SNU Module Speed: {:.2f}fps".format(
                #         self.fidx, len(snu_usr), total_fps
                #     )
                # )
                rospy.loginfo("FIDX: {} || # of Trajectories: <{}> || [SENSOR: {:.2f}fps | | SEG: {:.1f}fps | DET: {:.1f}fps | TRK: {:.1f}fps | ACL: {:.1f}fps]".format(
                    self.fidx, len(snu_usr), sensor_fps, fps_dict["seg"], fps_dict["det"], fps_dict["trk"], fps_dict["acl"]
                    )
                )

                # Draw Results
                # result_frame_dict = self.visualizer(
                #     sensor_data=self.color, trajectories=trajectories, detections=detections, fidx=self.fidx,
                #     segmentation=heatmap
                # )

                result_frames_dict = self.visualizer(sync_data_dict=sync_data_dict, trajectories=trajectories, detections=detections, segmentation=heatmap, fidx=self.fidx)

                # # Visualize Thermal Image
                # self.visualizer.visualize_modal_frames(sensor_data=self.thermal, precision=np.uint8)

                # # Projection Visualization
                # self.visualizer.visualize_modal_frames_with_calibrated_pointcloud(
                #     sensor_data=self.color, pc_img_coord=
                # )

                # Publish Tracks
                out_tracks = self.publish_tracks(trajectories=trajectories, odometry_msg=self.odometry_msg)

                # Publish SNU Result Image Results
                self.publish_snu_result_image(result_frames_dict=result_frames_dict)

                # NOTE: Publish Evaluator - Tentative Code for MOT Evaluation
                if len(trajectories) != 0:
                    # Search for Trajectory Modalities
                    modals = []
                    for trk in trajectories:
                        trk_modal = trk.modal
                        if len(modals) == 0:
                            modals.append(trk_modal)
                        else:
                            if trk.modal not in modals:
                                modals.append(trk_modal)

                    # Make Modal Trajectories Dict
                    modal_trks_dict = dict.fromkeys(modals)
                    for modal in modal_trks_dict.keys():
                        modal_trks_dict[modal] = []
                        for trk in trajectories:
                            if trk.modal == modal:
                                modal_trks_dict[modal].append(trk)

                    # Publish Evaluators for modals
                    for modal, modal_trks in modal_trks_dict.items():
                        from utils.ros.wrapper import wrap_tracks
                        curr_out_tracks = wrap_tracks(trackers=modal_trks, odometry=self.odometry_msg)
                        try:
                            self.publish_evaluator(out_tracks=out_tracks, modal=modal)
                        except AttributeError:
                            pass

            # Rospy Spin
            rospy.spin()

        except KeyboardInterrupt:
            rospy.logwarn("ShutDown SNU Module...!")


def main():
    # Set Logger
    logger = utils.loader.set_logger(logging_level=logging.INFO)

    # Argument Parser
    args = utils.loader.argument_parser(
        logger=logger, script_name=os.path.basename(__file__),
        dev_version=4.5, mode_selection=RUN_MODE
    )

    # Load Configuration from File
    cfg = utils.loader.cfg_loader(logger=logger, args=args)

    # Load Options
    opts = utils.loader.load_options(logger=logger, args=args, cfg=cfg)
    opts.visualization.correct_flag_options()

    # Initialize SNU Module
    ros_snu_usr = snu_module(logger=logger, opts=opts)

    # Run SNU Module
    ros_snu_usr(module_name="snu_module", force_set_time=args.day_night)


if __name__ == "__main__":
    main()
