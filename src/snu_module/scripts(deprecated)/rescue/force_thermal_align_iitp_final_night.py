"""
Code written by Daeho Um

processed by Kyuewang Lee, Daeho Um

"""

import numpy as np
import cv2
import os


class distortion_params(object):
    def __init__(self, k1, k2, p1, p2):
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2

    def get_distCoeff(self, dtype=np.float64):
        dist_coeff = np.zeros((4, 1), dtype=dtype)
        dist_coeff[0, 0] = self.k1
        dist_coeff[1, 0] = self.k2
        dist_coeff[2, 0] = self.p1
        dist_coeff[3, 0] = self.p2
        return dist_coeff


class heuristic_camera_params(object):
    def __init__(self, src_modal, dest_modal):
        # Temp Code
        self.x_slicings = [97, 550]
        self.y_slicings = [75, 440]  # [75, 440]

        # < RGB > to < Thermal >
        if src_modal in ["rgb"] and dest_modal in ["thermal"]:
            self.frame_width = 640
            self.frame_height = 512

            k1 = 2.1e-4
            k2 = 1e-8
            p1 = 0.0
            p2 = 0.0

        # < Thermal > to < RGB >
        elif src_modal in ["thermal"] and dest_modal in ["rgb"]:
            self.frame_width = 640
            self.frame_height = 480

            k1 = -1.5e-4
            k2 = -8e-10
            p1 = 0.0
            p2 = 0.0

            self.x_slicings = [97, 550]
            self.y_slicings = [75, 460] #[75, 440]

        else:
            assert 0, "No Case for this (source)(dest)"

        # Distortion Parameter Class
        self.dist_params = distortion_params(k1, k2, p1, p2)

        # Camera Matrix
        cam_matrix = np.eye(3, dtype=np.float32)
        cam_matrix[0, 2] = (self.frame_width - 1) / 2.0  # center x
        cam_matrix[1, 2] = (self.frame_height - 1) / 2.0  # center y
        cam_matrix[0, 0] = 10.0  # focal length x
        cam_matrix[1, 1] = 10.0  # focal length y

        self.cam_matrix = cam_matrix

    # Undistort Points
    def undistort_coordinates(self, X, Y):
        XY_th_dst = np.array([X, Y])
        XY_th_dst = XY_th_dst.reshape(-1, 1, 2)

        distCoeff = self.dist_params.get_distCoeff()

        XY_undst = cv2.undistortPoints(XY_th_dst.astype(np.float32), self.cam_matrix, distCoeff)
        X_undistorted = XY_undst[0][0][0] * self.cam_matrix[0, 0] + (self.frame_width - 1) / 2.0
        Y_undistorted = XY_undst[0][0][1] * self.cam_matrix[1, 1] + (self.frame_height - 1) / 2.0

        return X_undistorted, Y_undistorted

    # (Code for Test) Undistort Image
    def undistort_image(self, src_img, resize=None, zero_padding=None):
        dst_img = cv2.undistort(src_img, self.cam_matrix, self.dist_params.get_distCoeff())
        dst_img = dst_img[self.y_slicings[0]:self.y_slicings[1], self.x_slicings[0]:self.x_slicings[1]]
        if resize is not None:
            dst_img = cv2.resize(np.copy(dst_img), dsize=resize)
        elif zero_padding is not None:
            vpadding = np.max((zero_padding[1] - dst_img.shape[0], 0))
            hpadding = np.max((zero_padding[0] - dst_img.shape[1], 0))

            # Padding, [top
            top_pad_pxls = int(vpadding/2.0)
            bottom_pad_pxls = vpadding - top_pad_pxls
            left_pad_pxls = int(hpadding/2.0)
            right_pad_pxls = hpadding - left_pad_pxls

            if len(dst_img.shape) == 2:
                pad_value = 0
            elif len(dst_img.shape) == 3:
                pad_value = [0, 0, 0]
            else:
                assert 0, "Unknown channel"

            dst_img = cv2.copyMakeBorder(np.copy(dst_img), top_pad_pxls, bottom_pad_pxls, left_pad_pxls, right_pad_pxls,
                                         cv2.BORDER_CONSTANT, pad_value)
            pass

        return dst_img


# Thermal -> RGB Coordinate Conversion
def thermal_to_rgb_coord(x_thermal, y_thermal):
    th_to_rgb_relation_obj = heuristic_camera_params(src_modal="thermal", dest_modal="rgb")

    x_slicings = th_to_rgb_relation_obj.x_slicings
    y_slicings = th_to_rgb_relation_obj.y_slicings

    rgb_width = th_to_rgb_relation_obj.frame_width
    rgb_height = th_to_rgb_relation_obj.frame_height

    # Undistort Thermal Points
    x_th_undst, y_th_undst = th_to_rgb_relation_obj.undistort_coordinates(x_thermal, y_thermal)
    # return x_th_undst, y_th_undst

    # # Resize to RGB Image Coordinates
    x_rgb = int(round((x_th_undst - x_slicings[0]) * rgb_width / (x_slicings[1] - x_slicings[0])))
    y_rgb = int(round((y_th_undst - y_slicings[0]) * rgb_height / (y_slicings[1] - y_slicings[0])))
    return x_rgb, y_rgb


# RGB -> Thermal Coordinate Conversion
def rgb_to_thermal_coord(x_rgb, y_rgb):
    """
    *** Manually Find Position in the Action Recognition Module
    """
    rgb_to_th_relation_obj = heuristic_camera_params(src_modal="rgb", dest_modal="thermal")

    # Back-ward Resize (Test)
    x_slicings = rgb_to_th_relation_obj.x_slicings
    y_slicings = rgb_to_th_relation_obj.y_slicings

    # thermal_width = rgb_to_th_relation_obj.frame_width
    # thermal_height = rgb_to_th_relation_obj.frame_height

    x_rgb_temp = x_rgb * (x_slicings[1]-x_slicings[0])/640.0 + x_slicings[0]
    y_rgb_temp = y_rgb * (y_slicings[1]-y_slicings[0])/480.0 + y_slicings[0]

    # Undistort RGB Points to Thermal Coordinates
    x_rgb_undst, y_rgb_undst = rgb_to_th_relation_obj.undistort_coordinates(x_rgb_temp, y_rgb_temp)
    return x_rgb_undst, y_rgb_undst

    # # # Resize to Thermal Image Coordinates
    # x_thermal = int(round((x_rgb_undst - x_slicings[0]) * thermal_width / (x_slicings[1] - x_slicings[0])))
    # y_thermal = int(round((y_rgb_undst - y_slicings[0]) * thermal_height / (y_slicings[1] - y_slicings[0])))
    # return x_thermal, y_thermal


# Test Code
def test_code():
    img_base_path = "/home/kyle/USR_SNU_MODULE/SNU_Integrated_v2/src/snu_module/DATA/US_2019_POHANG/01/MV12_0604_d_fin/"

    rgb_frame_path = os.path.join(img_base_path, "rgbdepth", "color", "MV12_RGB_190604_220322_08.png")
    thermal_frame_path = os.path.join(img_base_path, "thermal", "MV12_THERMAL_190604_220322_08.png")

    img_gray = cv2.cvtColor(cv2.imread(rgb_frame_path), cv2.COLOR_BGR2GRAY)
    img_thermal = cv2.imread(thermal_frame_path, cv2.IMREAD_UNCHANGED)

    th_to_rgb_obj = heuristic_camera_params(src_modal="thermal", dest_modal="rgb")
    rgb_to_th_obj = heuristic_camera_params(src_modal="rgb", dest_modal="thermal")

    # Thermal to RGB Conversion Test
    img_cvt_gray = th_to_rgb_obj.undistort_image(img_thermal, resize=(640, 480))
    img_cvt_th = rgb_to_th_obj.undistort_image(img_gray, resize=(640, 480))

    # # Sum Images (check thermal to rgb)
    # sum_img = 0.5*img_gray + 0.5*img_cvt_gray
    # # View
    # cv2.imshow("Compare", sum_img.astype(np.uint8))
    # cv2.waitKey(0)

    # Horizontal Stack Images (check rgb to thermal)
    vertical_bar = np.zeros((4, img_gray.shape[1]))
    cmp_img = np.vstack((img_gray, vertical_bar, img_cvt_th, vertical_bar, img_thermal))
    cmp_img = cv2.resize(cmp_img, dsize=(cmp_img.shape[1] // 2, cmp_img.shape[0] // 2))

    # View
    cv2.imshow("Compare", cmp_img.astype(np.uint8))
    cv2.waitKey(0)

    # Test Points
    x_thermal_test, y_thermal_test = 100, 100

    # T2RGB
    x_rgb_conversion, y_rgb_conversion = thermal_to_rgb_coord(x_thermal_test, y_thermal_test)

    # Re-transform to check
    x_thermal_reconversion, y_thermal_reconversion = rgb_to_thermal_coord(x_rgb_conversion, y_rgb_conversion)


if __name__ == '__main__':
    test_code()
