#!/usr/bin/env python
"""


"""
import cv2
import numpy as np
from copy import deepcopy
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo


class CAMERA_INFO_OBJECT(object):
    def __init__(self, modal, timestamp):
        assert isinstance(timestamp, dict)

        self.modal = modal
        self.header = Header()

        t_hour = int(timestamp["time"][0:2])
        t_min = int(timestamp["time"][2:4])
        t_sec = int(timestamp["time"][4:])
        t_seq = float(timestamp["fidx"])
        modal_stamp = rospy.rostime.Time.from_sec(
            float(t_hour * 3600 + t_min * 60 + t_sec) + 0.1 * t_seq
        )
        self.header.stamp = modal_stamp

        # Camera Parameters
        self.D, self.K, self.R, self.P = None, None, None, None

    def update_camera_parameters(self, D, K, R, P):
        self.D, self.K, self.R, self.P = D, K, R, P

    def get_camera_parameters(self):
        return {"D": self.D, "K": self.K, "R": self.R, "P": self.P}

    def get_timestamp(self):
        return self.header.stamp

    def to_CameraInfo(self, width, height, distortion_model="plumb_bob"):
        ros_camera_info = CameraInfo()
        ros_camera_info.header = self.header
        ros_camera_info.width = width
        ros_camera_info.height = height
        ros_camera_info.distortion_model = distortion_model

        ros_camera_info.D = self.D
        ros_camera_info.K = self.K
        ros_camera_info.R = self.R
        ros_camera_info.P = self.P

        ros_camera_info.binning_x, ros_camera_info.binning_y = 1, 1

        return ros_camera_info


class MODAL_BASE_OBJECT(object):
    def __init__(self, modal):
        # Modal
        self.modal = modal

        # Data
        self.data = None

        # Timestamp
        self.timestamp = None

        # Annotations
        self.annos = None

    def __repr__(self):
        return self.modal

    def set_data(self, data, timestamp):
        self.data = data
        self.timestamp = timestamp

    def set_annos(self, annos):
        self.annos = annos

    def get_data(self):
        return {"data": self.data, "timestamp": self.timestamp, "annos": self.annos}

    def get_modal(self):
        return self.modal


class IMAGE_MODAL_OBJ(MODAL_BASE_OBJECT):
    def __init__(self, modal, timestamp):
        super(IMAGE_MODAL_OBJ, self).__init__(modal)

        # CameraInfo
        self.camera_info = CAMERA_INFO_OBJECT(modal=self.modal, timestamp=timestamp)

    def get_camera_info(self):
        return self.camera_info

    def get_dtype(self):
        return self.data.dtype

    def update_camera_parameters(self, D, K, R, P):
        self.camera_info.update_camera_parameters(D, K, R, P)

    def get_camera_parameters(self):
        return self.camera_info.get_camera_parameters()

    def draw_annos(self):
        vis_frame = None
        if self.annos is not None:
            vis_frame = deepcopy(self.data.astype(np.uint8))
            if len(vis_frame.shape) == 2:
                vis_frame = np.dstack((vis_frame, vis_frame, vis_frame))

            # For Annotations,
            for anno in self.annos:
                bbox = anno.bbox.numpify()

                # Draw Rectangle BBOX (Left-Top Right-Bottom)
                if anno.cls == "Car" or anno.cls == "Motorcycle":
                    color = (255, 0, 0)
                elif anno.cls == "Human":
                    color = (0, 255, 0)
                else:
                    color = (120, 120, 120)

                cv2.rectangle(
                    vis_frame,
                    (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                    color, 2
                )

        return vis_frame, self.timestamp


class LIDAR_MODAL_OBJ(MODAL_BASE_OBJECT):
    def __init__(self):
        super(LIDAR_MODAL_OBJ, self).__init__(modal="lidar")

    @staticmethod
    def get_camera_info():
        return None

    @staticmethod
    def get_camera_parameters():
        return None


class MODAL_DATA_OBJ(object):
    def __init__(self, modal_obj_list, modal):
        assert isinstance(modal_obj_list, list)

        # Modal
        self.modal = modal

        # Modal Object List
        self.modal_obj_list = modal_obj_list

    def __repr__(self):
        return self.modal

    def __len__(self):
        return len(self.modal_obj_list)

    def __getitem__(self, idx):
        return self.get_data(idx)

    def get_data(self, idx):
        return self.modal_obj_list[idx].get_data()

    def get_camera_info(self, idx):
        return self.modal_obj_list[idx].get_camera_info()

    def get_camera_parameters(self, idx):
        return self.modal_obj_list[idx].get_camera_parameters()


class MULTIMODAL_DATA_OBJ(object):
    def __init__(self, modal_data_obj_dict):
        assert isinstance(modal_data_obj_dict, dict)

        pre_modal_obj_len = None
        for modal, modal_data_obj in modal_data_obj_dict.items():
            if len(modal_data_obj) > 0:
                # FrameLength
                modal_obj_len = len(modal_data_obj)
                if pre_modal_obj_len is not None:
                    if pre_modal_obj_len != modal_obj_len:
                        raise AssertionError("Modal Images Length Mismatch!")
                pre_modal_obj_len = modal_obj_len

            # Set Class Attributes
            setattr(self, modal, modal_data_obj)

        # Set Data Length
        self.__dataLength = pre_modal_obj_len

    def __len__(self):
        return self.__dataLength

    def __getitem__(self, idx):
        raise NotImplementedError()

    def get_data(self, idx):
        data_dict, timestamp_dict, annos_dict = {}, {}, {}
        for modal, modal_obj in self.__dict__.iteritems():
            if modal == "_MULTIMODAL_DATA_OBJ__dataLength":
                continue
            if len(modal_obj) < idx+1:
                data_dict[modal], timestamp_dict[modal], annos_dict[modal] = None, None, None
                continue
            # Get Current Modal Data and Timestamp, Append to Dictionary
            idx_data_timestamp_dict = modal_obj.get_data(idx)
            data_dict[modal], timestamp_dict[modal], annos_dict[modal] = \
                idx_data_timestamp_dict["data"], idx_data_timestamp_dict["timestamp"], idx_data_timestamp_dict["annos"]
        return data_dict, timestamp_dict, annos_dict

    def get_camera_info(self, idx):
        camera_info_dict = {}
        for modal, modal_obj in self.__dict__.iteritems():
            if modal == "_MULTIMODAL_DATA_OBJ__dataLength":
                continue
            if len(modal_obj) < idx+1:
                camera_info_dict[modal] = None
                continue
            idx_camera_info = modal_obj.get_camera_info(idx)
            camera_info_dict[modal] = idx_camera_info
        return camera_info_dict

    def get_camera_parameters(self, idx=0):
        camera_params_dict = {}
        for modal, modal_obj in self.__dict__.iteritems():
            if modal == "_MULTIMODAL_DATA_OBJ__dataLength":
                continue
            if len(modal_obj) < idx + 1:
                camera_params_dict[modal] = None
                continue
            idx_camera_params = modal_obj.get_camera_parameters(idx)
            camera_params_dict[modal] = idx_camera_params
        return camera_params_dict

    def draw_modal_annos(self, modal):
        assert hasattr(self, modal), "Object does not have attribute [{}]...!".format(modal)
        if modal == "lidar":
            return None

        sel_modal_data_obj = getattr(self, modal)
        sel_modal_obj_list = sel_modal_data_obj.modal_obj_list

        vis_frames, vis_timestamps = [], []
        for sel_modal_obj in sel_modal_obj_list:
            vis_frame, vis_timestamp = sel_modal_obj.draw_annos()
            vis_frames.append(vis_frame)
            vis_timestamps.append(vis_timestamp)

        return vis_frames, vis_timestamps


if __name__ == "__main__":
    pass
