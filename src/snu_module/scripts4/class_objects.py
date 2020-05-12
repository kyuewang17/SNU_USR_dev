"""
SNU Integrated Module v3.0
    - Code which defines object classes
    [1] Tracklet Candidate Class
    [2] Tracklet Class

"""

# Import Modules
import numpy as np
import filterpy.kalman.kalman_filter as kalmanfilter

# Import Custom Modules
import snu_utils.bounding_box as snu_bbox
import kalman_params as kparams
import snu_utils.patch as snu_patch
import snu_utils.histogram as snu_hist


# Tracklet Candidate Class
class TrackletCandidate(object):
    # Initialization
    def __init__(self, bbox, conf, label, init_fidx):
        # Frame Index
        self.fidxs = [init_fidx]

        # Detection BBOX, Confidence, Label Lists
        self.asso_dets = [bbox]
        self.asso_confs = [conf]
        self.label = label

        # Association Flag List
        self.is_associated = [True]

        # z (observation bbox type: {u, v, du, dv, w, h})
        self.z = [snu_bbox.bbox_to_zx(bbox, np.zeros(2))]

        # Depth (tentative)
        self.depth = None

    # Destructor
    def __del__(self):
        pass

    # TrackletCandidate Length
    def __len__(self):
        """
        :return: {Number of Frame Indices Current TrackletCandidate Object Existed}
        """
        return len(self.fidxs)

    # Get Rough Depth from Disparity Map (Histogram-based)
    def get_rough_depth(self, disparity_frame, opts):
        # Get TrackletCandidate Patch
        patch = snu_patch.get_patch(
            img=disparity_frame, bbox=self.asso_dets[-1]
        )

        # Get Histogram
        disparity_hist, disparity_hist_idx = \
            snu_hist.histogramize_patch(
                sensor_patch=patch, dhist_bin=opts.tracker.disparity_params["rough_hist_bin"],
                min_value=opts.sensors.disparity["clip_distance"]["min"],
                max_value=opts.sensors.disparity["clip_distance"]["max"]
            )

        # Get Max-bin and Representative Depth Value of Disparity Histogram
        max_bin = disparity_hist.argmax()
        depth_value = ((disparity_hist_idx[max_bin] + disparity_hist_idx[max_bin + 1]) / 2.0) / 1000.0


    # Update
    def update(self, fidx, bbox=None, conf=None):
        # Assertion
        if bbox is None and conf is not None:
            assert 0, "Input Method [bbox] cannot be <None> value while [conf] is not <None>!"

        # Update Current Frame Index
        self.fidxs.append(fidx)

        # Update Detection BBOX
        if bbox is None:
            self.asso_dets.append(None)
            self.asso_confs.append(None)
            self.is_associated.append(False)
            self.z.append(None)
        else:
            z_bbox = snu_bbox.bbox_to_zx(bbox)
            velocity = (z_bbox[0:2] - self.z[-1][0:2]).reshape(2)
            self.asso_dets.append(bbox)
            self.asso_confs.append(conf)
            self.is_associated.append(True)
            self.z.append(snu_bbox.bbox_to_zx(bbox, velocity))

    # Initialize Tracklet Class from TrackletCandidate
    def init_tracklet(self):
        pass


# Tracklet Class
class Tracklet(object):
    # Initialization
    def __init__(self, bbox, conf, label, init_fidx, init_depth, trk_id):
        """
        < Some Notations > (self)
        - Observation (Detection) [u,v,du,dv,w,h] : z2 < 6x1 >
        - Observation BBOX [l,t,w,h] : z_bbox
        - 3D Observation [u,v,D,du,dv,dD,w,h] : z3 < 8x1 >

        - 3D State on Camera Coordinates [X,Y,Z,dX,dY,dZ,w,h] : x3 < 8x1 >
            * (w,h) is the weight and height on Image Coordinates

        """
        # Tracklet ID
        self.id = trk_id

        # Tracklet Frame Indices
        self.fidxs = [init_fidx]

        # Associated Detections (bbox, confidence, label)
        self.asso_dets = [bbox]
        self.det_confs = [conf]
        self.label = label

        # Association Flag List
        self.is_associated = [True]

        # Tracklet Visualization Color

        # Tracklet Kalman Parameter Initialization
        # self.A = kparams.

        # Kalman States (Initial States)
        init_state = None
        # TODO: Add this!!

        # Initialize Depth
        self.depth = init_depth

        # Observations
        self.z2 = snu_bbox.bbox_to_zx(
            bbox=bbox, velocity=np.zeros(2)
        )
        self.z3 = snu_bbox.bbox_to_zx(
            bbox=bbox, velocity=np.zeros(2), depth=init_depth, ddepth=0.0
        )

        # Convert to "x3"
        self.x3 = None


        pass

    # Destructor
    def __del__(self):
        pass

    # Tracklet Length
    def __len__(self):
        """
        :return: {Number of Frame Indices Current Tracklet Object Existed}
        """
        return len(self.fidxs)

    # Tracklet Update
    def update(self, bbox=None, conf=None):
        """
        Kalman Update on 3D camera coordinates, if possible
        """
        pass

    # Tracklet Predict
    def predict(self):
        """
        Kalman Prediction on 3D camera coordinates, if possible
        """
        pass

    # Get Tracklet state in a specific frame index
    def get_state(self, fidx):
        pass

    # Get Image Coordinate State [Back-project x3 to image coordinates]
    def get_img_coord_state(self):
        pass

    # Update Tracklet Depth
    def update_depth(self):
        pass

    # Image Coordinates(2D) to Camera Coordinates(3D)
    def img_coord_to_cam_coord(self):
        pass

    # Camera Coordinates(3D) to Image Coordinates(2D)
    def cam_coord_to_img_coord(self):
        pass

    # Get 3D Position in Camera Coordinates (in meters)
    def get_3d_cam_coord(self):
        pass

    # Get Roll-Pitch-Yaw
    def compute_rpy(self):
        pass

    # Update Action Classification Results
    def update_action(self):
        pass













