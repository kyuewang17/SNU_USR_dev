#!/usr/bin/env python
"""
- Description PLZ

"""
import time, sys, os
import rosbag
import roslib
import rospy
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


from PIL import ImageFile

import cv2


class MODAL_IMAGES(object):
    def __init__(self, modal, frame_base_path):
        # Modal
        self.modal = modal

        # Get File Lists
        frame_file_list = sorted(os.listdir(frame_base_path))
        self.frame_files = [os.path.join(frame_base_path, frame_file_path)
                            for frame_file_path in frame_file_list]

        # Parse Timestamps
        self.timestamps = []
        for frame_file_name in frame_file_list:
            _date = frame_file_name.split(".")[0].split("_")[2]
            _time = frame_file_name.split(".")[0].split("_")[3]
            _fidx = frame_file_name.split(".")[0].split("_")[4]
            timestamp_dict = {"date": _date, "time": _time, "fidx": _fidx}
            self.timestamps.append(timestamp_dict)

    def __repr__(self):
        return self.modal

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        self.get_data(idx)

    def get_data(self, idx):
        # Load Image via OpenCV
        frame = cv2.cvtColor(cv2.imread(self.frame_files[idx]), cv2.COLOR_BGR2RGB)

        # Load Timestamp Dictionary
        timestamp_dict = self.timestamps[idx]

        return frame, timestamp_dict

class MULTIMODAL_IMAGES(object):
    def __init__(self, modal_obj_dict):
        assert isinstance(modal_obj_dict, dict)

        pre_modal_obj_len = None
        for modal, modal_obj in modal_obj_dict.items():
            if modal_obj is None:
                setattr(self, modal, None)
            else:
                modal_obj_len = len(modal_obj)
                if pre_modal_obj_len is not None:
                    if pre_modal_obj_len != modal_obj_len:
                        raise AssertionError("Modal Images Length Mismatch!")
                pre_modal_obj_len = modal_obj_len

                # Set Class Attributes
                setattr(self, modal, modal_obj)

        # Set FrameLength
        self.__framelength = pre_modal_obj_len

    def __len__(self):
        return self.__framelength

    def __getitem__(self, idx):
        frame_dict, timestamp_dict = {}, {}
        for modal, modal_obj in self.__dict__.iteritems():
            if modal=="_MULTIMODAL_IMAGES__framelength":
                continue
            if modal_obj is None:
                frame_dict[modal], timestamp_dict[modal] = None, None
                continue
            # Get Current Modal Timestamp and Append to Dict
            frame, timestamp = modal_obj.get_data(idx)
            frame_dict[modal], timestamp_dict[modal] = frame, timestamp
        return frame_dict, timestamp_dict


def load_multimodal_images(base_path):
    # Check if Base Path Exists
    if os.path.isdir(base_path) is False:
        raise AssertionError()

    # Get Folder Lists
    modal_lists = os.listdir(base_path)

    # Init Multimodal Image Object and Return
    modal_object_dict = {}
    for modal in modal_lists:
        if modal == "xml":
            continue
        if modal == "thermal":
            modal_object_dict[modal] = None
        else:
            modal_object_dict[modal] = \
                MODAL_IMAGES(modal=modal, frame_base_path=os.path.join(base_path, modal))
    MULTIMODAL_OBJ = MULTIMODAL_IMAGES(modal_obj_dict=modal_object_dict)
    return MULTIMODAL_OBJ


def generate_multimodal_bag_file(multimodal_obj, base_path):
    # Get Bag File Name
    bag_name = os.path.split(base_path)[-1] + ".bag"

    # Create Bag File
    # bag = rosbag.Bag(os.path.join(base_path, bag_name), "w")

    # Iterate for Multimodal Sensors
    for fidx, fidx_data in enumerate(multimodal_obj):
        pass

    pass












if __name__ == "__main__":
    base_path = "/home/snu/DATA/USR_4th_Final_Demo"
    MULTIMODAL_OBJ = load_multimodal_images(base_path=base_path)
    generate_multimodal_bag_file(multimodal_obj=MULTIMODAL_OBJ, base_path=base_path)
