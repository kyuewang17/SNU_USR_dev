"""
Open Numpy *.npy file
"""
import os
import numpy as np


# Open npy file
def open_npy_file(file_path):
    return np.load(file_path)


def main():
    file_base_path = "/home/kyle/bag_files/__image_sequence__[BAG_FILE]_[field_notmoving_190927]"
    sample_npy_file_path = os.path.join(file_base_path, "rgb_cam_params")

    # Sample npy file name
    sample_npy_file_name = "rgb_camera_params__[P].npy"

    # Open npy file
    data_ndarray = open_npy_file(os.path.join(sample_npy_file_path, sample_npy_file_name))

    return


if __name__ == '__main__':
    main()