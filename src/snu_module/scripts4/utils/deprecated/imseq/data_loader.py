"""
SNU Integrated Module
    - Multimodal Data Loader for Image Sequence

"""
import os
import copy
import cv2
import numpy as np


class modal_data_loader(object):
    def __init__(self, modal_type, data_base_path):
        # Assertion
        assert isinstance(modal_type, str) and isinstance(data_base_path, str)
        if os.path.isdir(data_base_path) is False:
            raise AssertionError()
        if modal_type not in ["color", "disparity", "infrared", "lidar", "nightvision", "thermal"]:
            raise AssertionError()

        # Modal Type
        self.modal_type = modal_type

        # Data Base Path
        self.data_base_path = data_base_path

        # Initialize Data Variable
        self.data = []

        # Initialize Data Timestamp Variable
        self.timestamp = []

        # Load Data
        self.load_data()

    def __repr__(self):
        return self.modal_type

    def __getitem__(self, idx):
        return {"data": self.data[idx], "timestamp": self.timestamp[idx]}

    def __len__(self):
        return len(self.data)

    def load_data(self):
        # Get Modal Data Path
        modal_data_path = os.path.join(self.data_base_path, self.modal_type)
        if os.path.isdir(modal_data_path) is False:
            raise AssertionError()

        # Get Files and Save to the Object
        modal_data_file_list = sorted(os.listdir(modal_data_path))
        for data_fidx, data_file_name in enumerate(modal_data_file_list):
            data_file_path = os.path.join(modal_data_path, data_file_name)

            if self.modal_type != "lidar":
                # Read Image via OpenCV-Python
                _data = cv2.imread(data_file_path)
                if self.modal_type == "color":
                    _data = cv2.cvtColor(
                        _data, cv2.COLOR_BGR2RGB
                    )
            else:
                # Read PointCloud via Numpy Module
                _data_dict = np.load(data_file_path)
                _data = {
                    "cloud_colors": _data_dict["cloud_colors"],
                    "cloud_distance": _data_dict["cloud_distance"],
                    "uv_cloud": _data_dict["uv_cloud"]
                }

            # Append to Data Variable
            self.data.append(_data)

            # Parse File Name to Get Timestamp
            timestamp_nsec = np.uint64(data_file_name.split(".")[0].split("__")[-1])
            self.timestamp.append(timestamp_nsec)

    def get_data(self, idx):
        return self.data[idx]

    def get_timestamp(self, idx):
        return self.timestamp[idx]


class multimodal_data_loader(object):
    def __init__(self, opts, imseq_base_path):
        # Load Options
        self.opts = opts

        # Get Modal Switch Dictionary Options and Initialize Modal Data Loader Dictionary
        modal_frame_length_list = []
        self.modal_data_loader_dict = copy.deepcopy(opts.modal_switch_dict)
        for modal_type, modal_obj in self.modal_data_loader_dict.items():
            if modal_obj is True:
                curr_modal_data_loader = \
                    modal_data_loader(modal_type=modal_type, data_base_path=imseq_base_path)
                self.modal_data_loader_dict[modal_type] = curr_modal_data_loader
                modal_frame_length_list.append(len(curr_modal_data_loader))
            else:
                self.modal_data_loader_dict[modal_type] = None

        # Check for Frame Length Validity
        if all(elem == modal_frame_length_list[0] for elem in modal_frame_length_list) is False:
            raise AssertionError()
        self.frame_length = modal_frame_length_list[0]

    def __getitem__(self, idx):
        modal_get_item_result_dict = {}
        for modal_type, modal_obj in self.modal_data_loader_dict.items():
            modal_get_item_result_dict[modal_type] = \
                modal_obj[idx] if modal_obj is not None else None
        return modal_get_item_result_dict

    def __len__(self):
        return self.frame_length

    def get_data(self, idx):
        modal_get_item_result_dict = {}
        for modal_type, modal_obj in self.modal_data_loader_dict.items():
            modal_get_item_result_dict[modal_type] = \
                modal_obj.get_data(idx) if modal_obj is not None else None
        return modal_get_item_result_dict

    def get_timestamps(self, idx):
        modal_get_item_result_dict = {}
        for modal_type, modal_obj in self.modal_data_loader_dict.items():
            modal_get_item_result_dict[modal_type] = \
                modal_obj.get_timestamp(idx) if modal_obj is not None else None
        return modal_get_item_result_dict

    def get_repr_timestamp(self, idx):
        timestamps_dict = self.get_timestamps(idx)
        repr_timestamp = 0
        for value in timestamps_dict.values():
            repr_timestamp += value
        repr_timestamp = repr_timestamp / (1.0*len(timestamps_dict)*1e9)
        return repr_timestamp


if __name__ == "__main__":
    pass
