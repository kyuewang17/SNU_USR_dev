#!/usr/bin/env python
"""


"""
import numpy as np


CAMERA_INFO = {
    # RGB Camera Info
    "RGB": {
        # Distortion Matrix
        "D": np.array([0, 0, 0, 0, 0], dtype=np.float64).tolist(),
        # Intrinsic Camera Matrix
        "K": np.array([610.456482, 0, 325.258026, 0, 610.387329, 242.385468, 0, 0, 1],
                      dtype=np.float64).tolist(),
        # Rectification Matrix
        "R": np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64).tolist(),
        # Projection Matrix
        "P": np.array([610.456482, 0, 325.258026, 0, 0, 610.387329, 242.385468, 0, 0, 0, 1, 0],
                      dtype=np.float64).tolist()
    },

    # Depth
    "aligned_depth": {
        # Distortion Matrix
        "D": np.array([0, 0, 0, 0, 0], dtype=np.float64).tolist(),
        # Intrinsic Camera Matrix
        "K": np.array([386.904297, 0, 322.290070, 0, 386.904297, 243.228958, 0, 0, 1],
                      dtype=np.float64).tolist(),
        # Rectification Matrix
        "R": np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64).tolist(),
        # Projection Matrix
        "P": np.array([386.904297, 0, 322.290070, 0, 0, 386.904297, 243.228958, 0, 0, 0, 1, 0],
                      dtype=np.float64).tolist()
    },

    # Infrared
    "infrared": {
        # Distortion Matrix
        "D": np.array([0, 0, 0, 0, 0], dtype=np.float64).tolist(),
        # Intrinsic Camera Matrix
        "K": np.array([386.904297, 0, 322.290070, 0, 386.904297, 243.228958, 0, 0, 1],
                      dtype=np.float64).tolist(),
        # Rectification Matrix
        "R": np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64).tolist(),
        # Projection Matrix
        "P": np.array([386.904297, 0, 322.290070, 0, 0, 386.904297, 243.228958, 0, 0, 0, 1, 0],
                      dtype=np.float64).tolist()
    },

    # Nightvision
    "nightvision": {
        # Distortion Matrix
        "D": [],
        # Intrinsic Camera Matrix
        "K": [],
        # Rectification Matrix
        "R": [],
        # Projection Matrix
        "P": []
    },

    # Thermal
    "thermal": {
        # Distortion Matrix
        "D": np.array([0, 0, 0, 0, 0], dtype=np.float64).tolist(),
        # Intrinsic Camera Matrix
        "K": np.array([436.128240, 0, 342.138120, 0, 406.085860, 222.085950, 0, 0, 1],
                      dtype=np.float64).tolist(),
        # Rectification Matrix
        "R": np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64).tolist(),
        # Projection Matrix
        "P": np.array([436.128240, 0, 342.138120, 0, 0, 406.085860, 222.085950, 0, 0, 0, 1, 0],
                      dtype=np.float64).tolist()
    }
}


if __name__ == "__main__":
    pass
