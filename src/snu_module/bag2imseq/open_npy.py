"""
Open Numpy *.npy file
"""
import os
import numpy as np


# Open npy file
def open_npy_file(file_path):
    return np.load(file_path)


def main():
    file_base_path = os.path.join(os.path.dirname(__file__), "test", "_cvt_data__[test]")
    sample_npy_file_base_path = os.path.join(file_base_path, "camera_params", "color")

    # Sample npy file name
    sample_npy_file_name = "P.npy"

    # Open npy file
    data_ndarray = open_npy_file(os.path.join(sample_npy_file_base_path, sample_npy_file_name))

    return


if __name__ == '__main__':
    main()
