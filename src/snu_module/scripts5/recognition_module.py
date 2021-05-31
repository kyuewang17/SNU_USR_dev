#!/usr/bin/env python
"""
Object Recognition Module (SNU Integrated Module)
for Outdoor Surveillance Robots

    - TBA

"""
import os
import time
import rospy
import yaml
import logging
import numpy as np

from module_bridge import osr_recognition_algorithms

import utils.loader
from utils.visualizer import visualizer
from utils.profiling import Timer
from utils.framework import RECOGNITION_BASE_OBJECT

from utils.objects.MULTIMODAL_SENSORS import MULTIMODAL_SENSORS_OBJ


# Run Mode (BAG or AGENT)
__RUN_MODE__ = "BAG"


class OSR_RECOGNITION(RECOGNITION_BASE_OBJECT):
    def __init__(self, logger, opts):
        super(OSR_RECOGNITION, self).__init__(opts=opts)

        # Initialize Frame Index
        self.fidx = 0

        # Initialize Loop Timer
        self.loop_timer = Timer(convert="FPS")

        # Synchronized Timestamp of Multimodal Sensors
        self.sync_timestamp = None

        # Declare Visualizer
        self.visualizer = visualizer(opts=opts)

        """ Private Attributes """
        self.__logger = logger

    def temp_func(self):
        raise NotImplementedError()

    def __call__(self, module_name):
        # Initialize OSR Recognition Algorithm Class
        osr_recognition_module = osr_recognition_algorithms(opts=self.opts)
        self.__logger.info("SNU Algorithm and Neural Network Models Loaded...!")
        time.sleep(0.05)

        # ROS Node Initialization
        self.__logger.info("ROS Node Initialization")
        rospy.init_node(name=module_name, anonymous=True)

        # Initialize Multimodal Sensors Object
        MMS_OBJ = MULTIMODAL_SENSORS_OBJ(sensor_opts=self.opts.sensors)
        rospy.loginfo("Initialize Multimodal Sensor Object")

        # Get TF_TRANSFORM for LiDAR (e.g. check for TF_STATIC node)
        MMS_OBJ.update_tf_transform()

        # Recognition Module Loop Starts
        rospy.loginfo("Starting Recognition Module...!")
        try:
            while not rospy.is_shutdown():
                self.loop_timer.reset()

                # Make Synchronized Multimodal Sensor Data
                MMS_OBJ.make_sync_data()

                # Update Synchronized Multimodal Sensor Data
                MMS_OBJ.update_sensors()

                # Check for FPS
                sensor_fps = self.loop_timer.elapsed

                # Get Synchronized Timestamp and Update Frame Index
                self.fidx += 1
                self.sync_timestamp = MMS_OBJ.sync_timestamp

                # Update Sensor Image Frame Size (Color)
                if self.fidx == 1:
                    self.opts.sensors.update_sensor_image_size(
                        frame=MMS_OBJ.color.get_data()
                    )

                # Get All Modal Data Objects
                trajectories, detections, heatmap, fps_dict = osr_recognition_module(
                    sync_data_dict=MMS_OBJ.get_all_modal_objects(), fidx=self.fidx
                )

                # Log Profile
                _profile_str = "[{}] (FIDX: {}) || # of Trajectories: <{}>".format(
                    self.sync_timestamp, self.fidx, len(osr_recognition_module)
                )
                rospy.loginfo(_profile_str)

                # Draw Results
                results_frame_dict = self.visualizer(
                    sensor_data=MMS_OBJ.color, trajectories=trajectories,
                    detections=detections, segmentation=heatmap, fidx=self.fidx
                )

                # Publish Tracks
                self.publish_tracks(trajectories=trajectories, odom_msg=MMS_OBJ.get_odom_msg())

                # Publish Recognition Result Image Results
                self.publish_recognition_result_image(results_frame_dict=results_frame_dict)

                # Top-view Result is tentatively removed

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
        dev_version=5.0, mode_selection=__RUN_MODE__
    )

    # Load Configuration from File
    cfg = utils.loader.cfg_loader(logger=logger, args=args)

    # Load Options
    opts = utils.loader.load_options(logger=logger, args=args, cfg=cfg)
    opts.visualization.correct_flag_options()

    # Initialize Recognition Module
    recognition_module = OSR_RECOGNITION(logger=logger, opts=opts)

    # Run Recognition Module
    recognition_module(module_name="OSR-RECOGNITION-MODULE")


if __name__ == "__main__":
    main()
