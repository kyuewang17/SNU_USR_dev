#!/usr/bin/env python
"""
Object Recognition Module (SNU Integrated Module)
for Outdoor Surveillance Robots

    - TBA

"""
import os
import time
import yaml
import logging
import numpy as np

from module_bridge import osr_recognition_algorithms

from utils.visualizer import visualizer
from utils.profiling import Timer
from utils.framework import RECOGNITION_BASE_OBJECT


class OSR_RECOGNITION(RECOGNITION_BASE_OBJECT):
    def __init__(self, logger, opts):
        super(OSR_RECOGNITION, self).__init__(opts)

        # Initialize Frame Index
        self.fidx = 0

        # Initialize Loop Timer
        self.loop_timer = Timer(convert="FPS")

        # Synchronized Timestamp of Multimodal Sensors
        self.sync_stamp = None

        # Declare Visualizer
        self.visualizer = visualizer(opts=opts)

    def temp_func(self):
        raise NotImplementedError()

    def __call__(self):
        # Initialize




if __name__ == "__main__":
    pass
