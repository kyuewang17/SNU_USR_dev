"""
Code from below GitHub URL, code adapted to our integrated module
- https://github.com/yubbit/kcf/blob/master/kcf.py

SNU Integrated Module 4.5
    - KCF Tracker as Module
    - Papers
        - [ECCV 2012] Exploiting the Circulant Structure of Tracking-by-detection with Kernels

"""
import cv2
import numpy as np
import bounding_box as snu_bbox


# Single-Object-Tracker Class
class SOT(object):
    def __init__(self, init_frame, init_zx_bbox):
        # Target ROI
        self.roi = init_zx_bbox

        # Target Patch
        self.patch = get_subwindow(frame=init_frame, roi=init_zx_bbox.astype(int))

        # ROI List
        self.roi_list = [init_zx_bbox]

    def get_bbox(self):
        raise NotImplementedError()

    def track(self, *args, **kwargs):
        raise NotImplementedError()


# KCF Single-Object-Tracker Class
class KCF(SOT):
    def __init__(self, init_frame, init_zx_bbox, init_fidx, kcf_params):
        assert isinstance(init_zx_bbox, np.ndarray)
        super(KCF, self).__init__(init_frame=init_frame, init_zx_bbox=init_zx_bbox)

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
        self.orig_roi = self.roi.copy()

        # Pad ROI and Moves BBOX Accordingly
        roi_center = np.array([self.roi[0] + self.roi[2] * 0.5, self.roi[1] + self.roi[3] * 0.5])
        self.roi[2] = np.floor(self.roi[2] * (1 + self._padding))
        self.roi[3] = np.floor(self.roi[3] * (1 + self._padding))
        self.roi[0] = np.floor(roi_center[0] - self.roi[2] / 2)
        self.roi[1] = np.floor(roi_center[1] - self.roi[3] / 2)

        # ROI Difference
        self.orig_diff = self.roi[0:2] - self.orig_roi[0:2]

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

    def get_bbox(self):
        return self.orig_roi

    def _extract_feature(self, frame, roi):
        # Get Subwindow (Previous Patch)
        x = get_subwindow(frame=frame, roi=roi.astype(int))
        if self._is_resize is True:
            x = cv2.resize(x, tuple(self._resize_sz))
        if self._is_window is True:
            x = (x.T * self.cos_window.T).T

        # Get ROI Feature
        x_f = np.fft.fft2(x, axes=(0, 1))
        return x_f

    def track(self, frame):
        # Update KCF Call Index
        self.call_idx += 1

        # For First Frame, Define

        # Test KCF Regressor on Current Frame ROI
        if self.call_idx > 1:
            # Get Current ROI Test Feature
            z_f = self._extract_feature(frame=frame, roi=self.roi)

            # Get Test Parameter
            kz_f = gaussian_correlation(z_f, self.xhat_f, self._sigma)

            # Searches for the shift with the greatest response to the model
            # shift is of the form [y_pos, x_pos]
            resp = np.real(np.fft.ifft2(kz_f * self.ahat_f))
            shift = np.unravel_index(resp.argmax(), resp.shape)
            shift = np.array(shift) + 1

            # If the shift is higher than halfway, then it is interpreted as being a shift
            # in the opposite direction (i.e., right by default, but becomes left if too high)
            if shift[0] > z_f.shape[0] / 2:
                shift[0] -= z_f.shape[0]
            if shift[1] > z_f.shape[1] / 2:
                shift[1] -= z_f.shape[1]

            if self._is_resize is True:
                shift = np.ceil(shift * self._resize_diff)

            # Move ROI Center by Shift
            self.roi[0] += shift[1] - 1
            self.roi[1] += shift[0] - 1

            # Append to ROI List
            self.roi_list.append(self.roi)

            # Move Original ROI
            self.orig_roi[0:2] = self.roi[0:2] - self.orig_diff

        # Train KCF Regressor on Current Frame ROI
        x_f = self._extract_feature(frame=frame, roi=self.roi)

        # Get Train Parameters
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
    def __init__(self, init_frame, init_zx_bbox, init_fidx, kcf_params):
        super(KCF_PREDICTOR, self).__init__(init_frame, init_zx_bbox, init_fidx, kcf_params)

        # ROI as Original Size
        self.bbox = None

        # Train KCF Regressor
        self.x_f = self._extract_feature(frame=init_frame, roi=self.roi)

        # Get Train Parameters
        self.k_f = gaussian_correlation(self.x_f, self.x_f, self._sigma)
        self.a_f = self.y_f / (self.k_f + self._lambda)

    def get_bbox(self):
        pass

    def predict_bbox(self, frame, roi_bbox):
        # Convert BBOX to ZX (ROI)
        roi = np.zeros(4)
        roi[0] = roi_bbox[0]
        roi[1] = roi_bbox[1]
        roi[2] = roi_bbox[2] - roi_bbox[0]
        roi[3] = roi_bbox[3] - roi_bbox[1]

        # Copy ROI
        orig_roi = roi.copy()

        # Adopt Padding to ROI
        roi_center = np.array([roi[0] + roi[2] * 0.5, roi[1] + roi[3] * 0.5])
        roi[2] = np.floor(roi[2] * (1 + self._padding))
        roi[3] = np.floor(roi[3] * (1 + self._padding))
        roi[0] = np.floor(roi_center[0] - roi[2] / 2)
        roi[1] = np.floor(roi_center[1] - roi[3] / 2)

        # Get Current ROI Test Feature
        z_f = self._extract_feature(frame=frame, roi=roi)

        # Get Test Parameter
        kz_f = gaussian_correlation(z_f, self.x_f, self._sigma)

        # Searches for the shift with the greatest response to the model
        # shift is of the form [y_pos, x_pos]
        resp = np.real(np.fft.ifft2(kz_f * self.a_f))
        shift = np.unravel_index(resp.argmax(), resp.shape)
        shift = np.array(shift) + 1

        # If the shift is higher than halfway, then it is interpreted as being a shift
        # in the opposite direction (i.e., right by default, but becomes left if too high)
        if shift[0] > z_f.shape[0] / 2:
            shift[0] -= z_f.shape[0]
        if shift[1] > z_f.shape[1] / 2:
            shift[1] -= z_f.shape[1]

        if self._is_resize is True:
            shift = np.ceil(shift * self._resize_diff)

        # Move ROI Center by Shift
        roi[0] += shift[1] - 1
        roi[1] += shift[0] - 1

        # Move Original ROI
        orig_roi[0:2] = roi[0:2] - self.orig_diff

        # Convert Original ROI to BBOX
        predicted_bbox = np.zeros(4)
        predicted_bbox[0:2] = orig_roi[0:2]
        predicted_bbox[2] = orig_roi[0] + orig_roi[2]
        predicted_bbox[3] = orig_roi[1] + orig_roi[3]

        return predicted_bbox


# Get Sub-window
def get_subwindow(frame, roi):
    """Returns an array representing the subwindow of a frame, as defined by the ROI.
        Keyword arguments:
        frame -- The array representing the image. By default, must be of the form [y, x, ch]
        roi -- The array representing the ROI to be extracted. By default, is of the form
               [x_pos, y_pos, x_sz, y_sz]
        NOTE: This has the side effect of ensuring that the values of roi do not go beyond the image
              boundaries
        """
    if roi[0] < 0:
        roi[0] = 0
    if roi[1] < 0:
        roi[1] = 0

    if roi[0] + roi[2] > frame.shape[1]:
        roi[0] = frame.shape[1] - roi[2]

    if roi[1] + roi[3] > frame.shape[0]:
        roi[1] = frame.shape[0] - roi[3]

    return frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]


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
    xy = np.sum(np.real(np.fft.ifft2(xy_f, axes=(0,1))), 2)

    k_f = np.fft.fft2(np.exp(-1 / (sigma ** 2) * ((xx + yy - 2 * xy) / x_f.size).clip(min=0)))

    return k_f


if __name__ == "__main__":
    pass
