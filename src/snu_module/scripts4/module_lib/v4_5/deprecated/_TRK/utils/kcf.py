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


def gen_label(sigma, sz):
    """Generates a matrix representing the penalty for shifts in the image
    Keyword arguments:
    sigma -- The standard deviation of the Gaussian model
    sz -- An array of the form [x_sz, y_sz] representing the size of the feature array
    """
    sz = np.array(sz).astype(int)
    rs, cs = np.meshgrid(np.asarray(range(1, sz[0]+1)) - np.floor(sz[0]/2), np.asarray(range(1, sz[1]+1)) - np.floor(sz[1]/2))
    labels = np.exp(-0.5 / sigma ** 2 * (rs ** 2 + cs ** 2))

    # The [::-1] reverses the sz array, since it is of the form [x_sz, y_sz] by default
    labels = np.roll(labels, np.floor(sz[::-1] / 2).astype(int), axis=(0,1))

    return labels


def gaussian_correlation(x_f, y_f, sigma):
    """Calculates the Gaussian correlation between two images in the Fourier domain.
    Keyword arguments:
    x_f -- The representation of x in the Fourier domain
    y_f -- The representation of y in the Fourier domain
    sigma -- The variance to be used in the calculation
    """
    N = x_f.shape[0] * x_f.shape[1]
    xx = np.real(x_f.flatten().conj().dot(x_f.flatten()) / N)
    yy = np.real(y_f.flatten().conj().dot(y_f.flatten()) / N)

    xy_f = np.multiply(x_f, y_f.conj())
    if len(xy_f.shape) == 2:
        xy = np.real(np.fft.ifft2(xy_f, axes=(0, 1)))
    else:
        xy = np.sum(np.real(np.fft.ifft2(xy_f, axes=(0, 1))), 2)

    k_f = np.fft.fft2(np.exp(-1 / (sigma ** 2) * ((xx + yy - 2 * xy) / x_f.size).clip(min=0)))

    return k_f


def compute_psr_value(corr_response):
    # Flatten Response
    corr_resp_vec = corr_response.flatten()

    # Get Max Value
    max_response = np.max(corr_resp_vec)

    # Get Mean Value
    mean_response = np.mean(corr_resp_vec)

    # Get Standard Deviation Value
    std_response = np.std(corr_resp_vec)

    # Compute PSR Value and Return
    psr_value = (max_response - mean_response) / std_response
    return psr_value


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
            window_sz = np.floor(self.roi[2:4] / self._cell_size)
            self._resize_diff = None
        else:
            window_sz = np.floor(np.array(self._resize_sz) / self._cell_size)
            self._resize_diff = self.roi[2:4] / np.array(self._resize_sz)

        # Cosine Window
        if self._is_window is True:
            y_hann = np.hanning(window_sz[1]).reshape(-1, 1)
            x_hann = np.hanning(window_sz[0]).reshape(-1, 1)
            self.cos_window = y_hann.dot(x_hann.T)
        else:
            self.cos_window = None

        # Generates the output matrix to be used for training. "Penalizes" every shift in the image
        output_sigma = np.sqrt(np.prod(self.roi[2:4])) * self._osf
        self.y_f = np.fft.fft2(gen_label(output_sigma, window_sz))

    def _extract_feature(self, frame, roi):
        # Assertion
        assert isinstance(roi, bbox.BBOX)

        # Get Patch
        x = roi.get_patch(frame=frame)

        # Resize
        if self._is_resize is True:
            x = cv2.resize(x, tuple(self._resize_sz))

        # Cosine Window (Hann Window)
        if self._is_window is True:
            x = (x.T * self.cos_window.T).T

        # Get ROI Feature
        x_f = np.fft.fft2(x, axes=(0, 1))
        return x_f

    def track(self, frame):
        raise NotImplementedError()

    def update(self, frame, roi):
        # Get Current ROI Training Feature
        x_f = self._extract_feature(frame=frame, roi=roi)

        # Get Training Parameters
        k_f = gaussian_correlation(x_f, x_f, self._sigma)
        a_f = self.y_f / (k_f + self._lambda)

        # Stores the model's parameters. x_hat is kept for the calculation of kernel correlations
        if self.call_idx == 1:
            self.ahat_f = a_f.copy()
            self.xhat_f = x_f.copy()
        else:
            self.ahat_f = (1 - self._interp_factor) * self.ahat_f + self._interp_factor * a_f
            self.xhat_f = (1 - self._interp_factor) * self.xhat_f + self._interp_factor * x_f


# KCF Tracker as BBOX Predictor
class KCF_PREDICTOR(KCF):
    def __init__(self, init_frame, init_bbox, init_fidx, kcf_params):
        super(KCF_PREDICTOR, self).__init__(init_frame, init_bbox, init_fidx, kcf_params)

        # ROI as Original Size
        self.bbox = None

        # Initial Update
        self.update(frame=init_frame, roi_bbox=self.roi)

    def update(self, frame, roi_bbox):
        # Get Current ROI Training Feature
        x_f = self._extract_feature(frame=frame, roi=roi_bbox)

        # Get Training Parameters
        # Get Train Parameters
        self.k_f = gaussian_correlation(x_f, x_f, self._sigma)
        self.a_f = self.y_f / (self.k_f + self._lambda)

        # Stores the model's parameters. x_hat is kept for the calculation of kernel correlations
        if self.ahat_f is None:
            self.ahat_f, self.xhat_f = copy.deepcopy(self.a_f), copy.deepcopy(x_f)
        else:
            self.ahat_f = (1 - self._interp_factor) * self.ahat_f + self._interp_factor * self.a_f
            self.xhat_f = (1 - self._interp_factor) * self.xhat_f + self._interp_factor * x_f

    def predict(self, frame, roi_bbox):
        assert isinstance(roi_bbox, bbox.BBOX)

        # Copy ROI to Predicted ROI BBOX
        predicted_roi_bbox = copy.deepcopy(roi_bbox)

        # Convert roi_bbox type to "XYWH"
        roi_bbox.convert_bbox_fmt("XYWH")

        # Pad ROI
        roi_bbox[2] = np.floor(roi_bbox[2] * (1 + self._padding))
        roi_bbox[3] = np.floor(roi_bbox[3] * (1 + self._padding))
        roi_bbox.adjust_coordinates()

        # Get Current ROI Test Feature
        z_f = self._extract_feature(frame=frame, roi=roi_bbox)

        # Get Test Parameter
        kz_f = gaussian_correlation(z_f, self.x_f, self._sigma)

        # Searches for the shift with the greatest response to the model
        # shift is of the form [y_pos, x_pos]
        resp = np.real(np.fft.ifft2(kz_f * self.a_f))
        shift = np.unravel_index(resp.argmax(), resp.shape)
        shift = np.array(shift) + 1

        # Compute PSR Value
        psr_value = compute_psr_value(resp)

        # If the shift is higher than halfway, then it is interpreted as being a shift
        # in the opposite direction (i.e., right by default, but becomes left if too high)
        if shift[0] > z_f.shape[0] / 2:
            shift[0] -= z_f.shape[0]
        if shift[1] > z_f.shape[1] / 2:
            shift[1] -= z_f.shape[1]

        if self._is_resize is True:
            shift = np.ceil(shift * self._resize_diff)

        # Move ROI Center by Shift
        roi_bbox[0] += shift[1] - 1
        roi_bbox[1] += shift[0] - 1
        roi_bbox.adjust_coordinates()

        # Move Original ROI
        predicted_roi_bbox[0] = roi_bbox[0] - self.orig_diff
        predicted_roi_bbox[1] = roi_bbox[1] - self.orig_diff
        predicted_roi_bbox.adjust_coordinates()
        return predicted_roi_bbox, psr_value


if __name__ == "__main__":
    pass
