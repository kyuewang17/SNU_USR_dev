"""
Code from below GitHub URL, code adapted to our integrated module
- https://github.com/yubbit/kcf/blob/master/kcf.py

SNU Integrated Module 5.0
    - KCF Tracker as Module
    - Papers
        - [ECCV 2012] Exploiting the Circulant Structure of Tracking-by-detection with Kernels

"""
import copy
import cv2
import numpy as np

from module_lib.v4_5._TRK.objects import bbox


# Single-Object Tracker Class
class SOT(object):
    def __init__(self, init_frame, init_bbox):
        assert isinstance(init_bbox, bbox.BBOX)

        # Initial Target ROI
        self.roi = init_bbox.convert_bbox_fmt("XYWH")

        # Initial Target Patch
        self.patch = init_bbox.get_patch(frame=init_frame)


# KCF SOT Class
class KCF(SOT):
    def __init__(self, init_frame, init_bbox, init_fidx, kcf_params):
        super(KCF, self).__init__(init_frame, init_bbox)

        # KCF Class Call Index
        self.call_idx = 0

        # Module Frame Index
        self.fidx = init_fidx

        # Parameters
        self._lambda = kcf_params["lambda"]
        self._padding = kcf_params["padding"]
        self._sigma = kcf_params["sigma"]
        self._osf = kcf_params["output_sigma_factor"]
        self._interp_factor = kcf_params["interp_factor"]
        self._is_resize = kcf_params["resize"]["flag"]
        self._resize_sz = kcf_params["resize"]["size"]
        self._cell_size = kcf_params["cell_size"]
        self._is_window = kcf_params["is_cos_window"]

        # Initialize KCF Incremental Parameter
        self.xhat_f, self.ahat_f = None, None

        """ Pre-process """
        # Copy Original ROI
        self.orig_roi = copy.deepcopy(self.roi)

        # Pad ROI and Moves BBOX Accordingly
        self.roi[2] = np.floor(self.roi[2] * (1 + self._padding))
        self.roi[3] = np.floor(self.roi[3] * (1 + self._padding))
        self.roi.adjust_coordinates()

        # ROI Difference
        self.orig_diff = self.roi - self.orig_roi

        # ROI Resize (if option is True)
        if self._is_resize is False:
            window_sz = np.floor(self.roi[2:4] / )






if __name__ == "__main__":
    pass
