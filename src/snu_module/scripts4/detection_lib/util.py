import cv2
import numpy as np
import torch
# from config import cfg
# if cfg.detector.name == "RefineDet":
#     from . import _C
#     nms = _C.nms


def sort_boxes_s(boxes_s, confs_s, labels_s=None):
    sorted_confs_s, sorted_idxs = torch.sort(confs_s, dim=0, descending=True)
    # print('\t\t', sorted_idxs.shape)
    if len(sorted_idxs.shape) == 2:
        sorted_idxs = torch.squeeze(sorted_idxs, dim=1)
    # print('\t\t', sorted_idxs.shape)
    sorted_boxes_s = boxes_s[sorted_idxs]
    if labels_s is None:
        return sorted_boxes_s, sorted_confs_s
    else:
        sorted_labels_s = labels_s[sorted_idxs]
        return sorted_boxes_s, sorted_confs_s, sorted_labels_s


def cvt_torch2numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = tensor.detach().numpy()
    elif isinstance(tensor, list) or isinstance(tensor, tuple):
        for i in range(len(tensor)):
            tensor[i] = cvt_torch2numpy(tensor[i])
    # else:
    #     print(type(tensor))
    #     print(tensor)
    return tensor


class ThermalHeuristicCameraParams(object):
    # by Daeho Um
    def __init__(self, thermal_frame_width=640, thermal_frame_height=512):
        self.thermal_frame_width = thermal_frame_width
        self.thermal_frame_height = thermal_frame_height

        self.cam_matrix = np.eye(3, dtype=np.float32)
        self.cam_matrix[0, 0] = 10      # focal length x
        self.cam_matrix[1, 1] = 10      # focal length y
        self.cam_matrix[0, 2] = thermal_frame_width / 2.0
        self.cam_matrix[1, 2] = thermal_frame_height / 2.0

        self.k1 = -1.5e-4
        self.k2 = -8e-10
        self.p1 = 0.0
        self.p2 = 0.0

        self.distCoeff = np.zeros((4, 1), np.float64)
        self.distCoeff[0, 0] = self.k1
        self.distCoeff[1, 0] = self.k2
        self.distCoeff[2, 0] = self.p1
        self.distCoeff[3, 0] = self.p2

    def undistort_thermal_coordinates(self, X_thermal, Y_thermal):
        XY_th_dst = np.array([X_thermal, Y_thermal])
        XY_th_dst = XY_th_dst.reshape(-1, 1, 2)

        XY_th_undst = cv2.undistortPoints(XY_th_dst.astype(np.float32), self.cam_matrix, self.distCoeff)
        X_thermal_undistorted = XY_th_undst[0][0][0] * self.cam_matrix[0, 0] + (self.thermal_frame_width - 1) / 2.0
        Y_thermal_undistorted = XY_th_undst[0][0][1] * self.cam_matrix[1, 1] + (self.thermal_frame_height - 1) / 2.0

        return X_thermal_undistorted, Y_thermal_undistorted

    def undistort_thermal_images(self, thermal_img):
        undistorted_img = cv2.undistort(thermal_img, self.cam_matrix, self.distCoeff)

    def distort_to_thermal_coordinates(self, X, Y):
        # FILL HERE
        pass


def thermal_coord_to_rgb_coord(thermal_camera_params, rgb_size, x_thermal, y_thermal):
    x_slicings, y_slicings = [97, 550], [75, 440]
    # x_slicings, y_slicings = [107, 556], [80, 425]

    # thermal_frame_class = ThermalHeuristicCameraParams(thermal_frame.shape[1], thermal_frame.shape[0])
    x_thermal_undst, y_thermal_undst = thermal_camera_params.undistort_thermal_coordinates(x_thermal, y_thermal)
    x_rgb = int(round((x_thermal_undst-x_slicings[0]) * rgb_size[1] / float(x_slicings[1]-x_slicings[0])))
    y_rgb = int(round((y_thermal_undst-y_slicings[0]) * rgb_size[0] / float(y_slicings[1]-y_slicings[0])))
    return x_rgb, y_rgb


def rgb_coord_to_thermal_coord(rgb_frame, thermal_frame, x_rgb, y_rgb):
    # FILL HERE
    pass

