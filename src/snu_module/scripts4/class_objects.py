"""
SNU Integrated Module v3.0
    - Code which defines object classes
    [1] Tracklet Candidate Class
    [2] Tracklet Class

"""

# Import Modules
import cv2
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
        self.asso_det_confs = [conf]
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

        return depth_value

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
            self.asso_det_confs.append(None)
            self.is_associated.append(False)
            self.z.append(None)
        else:
            z_bbox = snu_bbox.bbox_to_zx(bbox)
            velocity = (z_bbox[0:2] - self.z[-1][0:2]).reshape(2)
            self.asso_dets.append(bbox)
            self.asso_det_confs.append(conf)
            self.is_associated.append(True)
            self.z.append(snu_bbox.bbox_to_zx(bbox, velocity))

    # Initialize Tracklet Class from TrackletCandidate
    def init_tracklet(self, disparity_frame, trk_id, fidx, opts):
        # Get Rough Depth
        depth = self.get_rough_depth(disparity_frame, opts)

        # Initialize Tracklet
        tracklet = Tracklet(
            asso_dets=self.asso_dets, asso_det_confs=self.asso_det_confs, label=self.label,
            is_associated=self.is_associated, init_fidx=fidx, init_depth=depth, trk_id=trk_id,
            colorbar=opts.tracker.tracklet_colors,
            colorbar_refresh_period=opts.tracker.trk_color_refresh_period
        )

        return tracklet


# Tracklet Class
class Tracklet(object):
    # Initialization
    def __init__(self, asso_dets, asso_det_confs, label, is_associated, init_fidx, init_depth, trk_id, colorbar, colorbar_refresh_period):
        """
        * Initialization from "TrackletCandidate" Class Object

        < Some Notations > (self)
        - Observation (Detection) [u,v,du,dv,w,h] : z2 < 6x1 >
        - Observation BBOX [u,v,w,h] : z2_bbox
        - 3D Observation [u,v,D,du,dv,w,h] : z3 < 7x1 >

        - 3D State on Image Coordinates : x3 < 7x1 >
            - Kalman Filtering of "z3"

        - 3D State on Camera Coordinates [X,Y,Z,dX,dY,dZ] : c3 < 6x1 >
            - Projected via "x3"

        NOTE: If the 3D State on Camera Coordinate is not stable, then consider
              conducting Kalman Filter on "c3"
                - in this case, "z3" is projected directly to "c3" and "c3" acts as
                  an observation

        """
        # Tracklet ID
        self.id = trk_id

        # Tracklet Frame Indices
        self.fidxs = [init_fidx]

        # Associated Detections
        self.asso_dets = asso_dets  # (bbox)
        self.asso_det_confs = asso_det_confs
        self.label = label

        # Association Flag
        self.is_associated = is_associated

        # Tracklet Depth Value
        self.depth = [init_depth]

        # Tracklet Visualization Color
        self.color = colorbar[self.id % colorbar_refresh_period, :] * 255

        # Initialize Tracklet Kalman Parameters
        self.A = kparams.A  # State Transition Matrix (Motion Model)
        self.H = kparams.H  # Unit Transformation Matrix
        self.P = kparams.P  # Error Covariance Matrix
        self.Q = kparams.Q  # State Covariance Matrix
        self.R = kparams.R  # Measurement Covariance Matrix

        # Initialize Image Coordinate Observation Vector
        curr_z2_bbox = snu_bbox.bbox_to_zx(asso_dets[-1])
        prev_z2_bbox = snu_bbox.bbox_to_zx(asso_dets[-2])
        init_observation = snu_bbox.bbox_to_zx(
            bbox=asso_dets[-1], velocity=(curr_z2_bbox - prev_z2_bbox)[0:2].reshape(2),
            depth=init_depth
        )

        # Kalman State (initial state)
        self.x3 = init_observation
        self.x3p, self.Pp = kalmanfilter.predict(self.x3, self.P, self.A, self.Q)

        # Tracklet States
        self.states = [self.x3]

        # Tracklet Predicted States
        self.pred_states = [self.x3p]

        # 3D Tracklet State on Camera Coordinates
        self.c3 = None

        # Roll, Pitch, Yaw
        self.roll, self.pitch, self.yaw = None, None, None

        # Action Classification Results
        self.pose_list = []
        self.pose = None

    # Get 2D Image Coordinate Tracklet State

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
    def update(self, fidx, bbox=None, conf=None):
        """
        Make sure that the depth value (self.depth) is updated prior to this code
        """
        # Assertion
        if bbox is None and conf is not None:
            assert 0, "Input Argument 'bbox' cannot be <None> while 'conf' is not <None>!"

        # Append Frame Index
        self.fidxs.append(fidx)

        # If Tracklet is unassociated, replace detection with the previous Kalman Prediction
        if bbox is None:
            self.asso_dets.append(None)
            self.asso_det_confs.append(None)
            self.is_associated.append(False)
            z3 = self.x3p
        else:
            self.asso_dets.append(bbox)
            self.asso_det_confs.append(conf)
            self.is_associated.append(True)

            # Get Velocity
            c = np.array([(bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0])
            velocity = c - self.x3p[0:2].reshape(2)

            # Make sure to Update Depth Prior to this code
            assert (len(self.fidxs) == len(self.depth)), "Depth Not Updated!"
            z3 = snu_bbox.bbox_to_zx(bbox=bbox, velocity=velocity, depth=self.depth[-1])

        # Kalman Update
        self.x3, self.P = kalmanfilter.update(self.x3p, self.Pp, z3, self.R, self.H)

        # Append to Tracklet States
        self.states.append(self.x3)

    # Tracklet Predict
    def predict(self):
        # Kalman Prediction
        self.x3p, self.Pp = kalmanfilter.predict(self.x3, self.P, self.A, self.Q)
        self.pred_states.append(self.x3p)

    # Get Tracklet 2D state on Image Coordinates
    def get_2d_img_coord_state(self):
        return snu_bbox.zx3_to_zx2(self.x3)

    # Get Tracklet Depth (as an Observation)
    def get_depth(self, sync_data_dict, opts):
        # Get Observation Patch bbox
        if self.asso_dets[-1] is not None:
            patch_bbox = self.asso_dets[-1]
        else:
            patch_bbox, _ = snu_bbox.zx_to_bbox(self.x3p)

        # Get Disparity Frame
        disparity_frame = sync_data_dict["disparity"].get_data()

        # Get and Process LiDAR Frame
        if sync_data_dict["lidar"].get_data() is not None:
            lidar_frame = sync_data_dict["lidar"].get_data() / 255.0
            # NOTE: Need to convert units into < millimeters >
            # lidar_frame = sync_data_dict["lidar"].get_data_in_mm()

            # Bilateral Filtering on LiDAR Frame
            # TODO: Implement Bilateral Filtering on LiDAR Frame
            pass
        else:
            lidar_frame = \
                np.zeros((disparity_frame.shape[0], disparity_frame.shape[1]), dtype=np.uint16)

        # Sum Frames
        depth_frame = (disparity_frame + lidar_frame) * 0.5

        # Get Patch
        trk_patch = snu_patch.get_patch(
            img=depth_frame, bbox=patch_bbox
        )

        # Get Histogram
        depth_hist, depth_hist_idx = snu_hist.histogramize_patch(
            sensor_patch=trk_patch, dhist_bin=opts.tracker.disparity_params["hist_bin"],
            min_value=opts.sensors.disparity["clip_distance"]["min"],
            max_value=opts.sensors.disparity["clip_distance"]["max"]
        )

        # Get Max-bin and Representative Depth Value of Disparity Histogram
        max_bin = depth_hist.argmax()
        depth_value = ((depth_hist_idx[max_bin] + depth_hist_idx[max_bin + 1]) / 2.0) / 1000.0

        self.depth.append(depth_value)


        # # Get Frames
        # if sync_data_dict["lidar"].get_data() is not None:
        #     disparity_frame = sync_data_dict["disparity"].get_normalized_data(0.6, 0.8)
        #     lidar_frame = sync_data_dict["lidar"].get_data() / 255.0
        #
        #     fusion_frame = 0.25*disparity_frame + 0.75*lidar_frame
        #
        #     comparison_frame = np.hstack((disparity_frame, fusion_frame))
        #
        #     # DEBUG
        #     cv2.imshow("comparison", comparison_frame)
        #     cv2.waitKey(1)
        #
        #     # Get Associated Disparity Patch
        #     if self.asso_dets[-1] is not None:
        #         patch_bbox = self.asso_dets[-1]
        #     else:
        #         patch_bbox, _ = snu_bbox.zx_to_bbox(self.x3p)
        #
        #     # Get Disparity Patch
        #     patch = snu_patch.get_patch(
        #         img=disparity_frame, bbox=patch_bbox
        #     )
        #
        #     print(1)
        # self.depth.append(0)

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


if __name__ == "__main__":
    tt = np.array([1, 2, 3, 4, 5])
    np.delete(tt, 0)

