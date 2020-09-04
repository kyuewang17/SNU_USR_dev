"""
SNU Integrated Module
    - Multimodal Data Loader for Image Sequence

"""
import os
import numpy as np


class modal_data_loader(object):
    def __init__(self, modal_type, data_base_path):
        # Assertion
        assert isinstance(modal_type, str) and isinstance(data_base_path, str)
        if os.path.isdir(data_base_path) is False:
            raise AssertionError()
        if modal_type not in ["color", "depth", "infrared", "lidar", "nightvision", "thermal"]:
            raise AssertionError()

        # Modal Type
        self.modal_type = modal_type

        # Data Base Path
        self.data_base_path = data_base_path

        pass

    def __repr__(self):
        return self.modal_type

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


class multimodal_data_loader(object):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass








if __name__ == "__main__":
    pass
