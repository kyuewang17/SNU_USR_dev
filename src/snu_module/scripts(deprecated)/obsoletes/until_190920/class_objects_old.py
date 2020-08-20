"""
SNU Integrated Module v2.0
  - Code that defines object classes
    [1] Detection Object
    [2] Tracklet Candidate Class
    [3] Tracklet Class
"""

# Import Modules
import numpy as np
import math
import filterpy.kalman.kalman_filter as kalmanfilter

# Import Source Libraries
import kalman_params as kparams
import utils.bounding_box as fbbox
import utils.patch as ptch


# [1] Tracklet Candidate Class
class TrackletCandidate(object):
    # Initialization
    def __init__(self, bbox, conf, label):
        self.age = 1
        self.asso_dets = [bbox]
        self.is_associated = [True]
        self.z = [fbbox.bbox_to_zx(bbox, np.zeros(2))]
        self.conf = conf
        self.label = label

    # Destructor
    def __del__(self):
        print("Tracklet Candidate Destroyed")

    # Update
    def update(self, bbox=None):
        self.age += 1
        if bbox is None:
            self.asso_dets.append(None)
            self.is_associated.append(False)
            self.z.append(None)
        else:
            z_bbox = fbbox.bbox_to_zx(bbox)
            velocity = (z_bbox[0:2] - self.z[-1][0:2]).reshape(2)
            self.asso_dets.append(bbox)
            self.is_associated.append(True)
            self.z.append(fbbox.bbox_to_zx(bbox, velocity))


# [2] Tracklet Class
class Tracklet(object):
    # Initialization
    def __init__(self, bbox, conf, label, fidx, trk_id, colorbar):
        # Tracklet ID
        self.id = trk_id
        # Tracklet Age
        self.age = 1
        # Tracklet Birth Frame
        self.birth_fidx = fidx
        # Associated Detections (bbox, confidence, label)
        self.asso_dets = [bbox]
        self.conf = conf
        self.label = label

        # Association Counter
        self.is_associated = [True]
        # Tracklet Visualization Color
        self.color = colorbar[(self.id % 3), :] * 255

        # Tracklet Kalman Parameter Initialization
        self.A = kparams.A      # State Transition Matrix (Motion Model)
        self.H = kparams.H      # Unit Transformation Matrix
        self.P = kparams.P      # Error Covariance Matrix
        self.Q = kparams.Q      # State Covariance Matrix
        self.R = kparams.R      # Measurement Covariance Matrix

        # Camera Coordinate Kalman Parameter Initialization
        self.cam_A = kparams.cam_A  # State Transition Matrix (Model Model)
        self.cam_H = kparams.cam_H  # Unit Transformation Matrix
        self.cam_P = kparams.cam_P  # Error Covariance Matrix
        self.cam_Q = kparams.cam_Q  # State Covariance Matrix
        self.cam_R = kparams.cam_R  # Measurement Covariance Matrix
        self.cam_Pp = None

        # Kalman States (Initial States)
        init_state = fbbox.bbox_to_zx(bbox, np.zeros(2))
        self.x = init_state
        self.xp, self.Pp = kalmanfilter.predict(self.x, self.P, self.A, self.Q)

        # Tracklet States
        self.states = [self.x]

        # 3D Camera Coordinate Position
        self.cam_coord_raw_state = None
        self.cam_coord_predicted_state = None
        self.cam_coord_estimated_state = None

        # Roll, Pitch, Yaw
        self.roll, self.pitch, self.yaw = None, None, None

        # Tracklet Predicted States (Next State Prediction)
        self.pred_states = [self.xp]

        # Depth of the Tracklet
        self.depth_hist, self.depth = None, None

        # Action Recognition Result of the Tracklet Object
        self.poselist = []
        self.pose = None

    # Destructor
    def __del__(self):
        print("Tracklet [id: " + str(self.id) + "] Destroyed!")

    # Tracklet Update
    def update(self, bbox=None):
        # Increase Tracklet Age
        self.age += 1

        # Get Detection (Measurement)
        if bbox is None:
            # If Tracklet is unassociated, replace detection with the previous Kalman prediction
            self.asso_dets.append(None)
            self.is_associated.append(False)
            z = self.pred_states[-1]
            z = np.array([z[0], z[1], np.zeros(1), np.zeros(1), z[4], z[5]])
        else:
            z_bbox = fbbox.bbox_to_zx(bbox)
            if self.asso_dets[-1] is None:
                prev_z_bbox = self.states[-1]
            else:
                prev_z_bbox = fbbox.bbox_to_zx(self.asso_dets[-1])
            velocity = (z_bbox[0:2] - prev_z_bbox[0:2]).reshape(2)
            self.asso_dets.append(bbox)
            self.is_associated.append(True)
            z = fbbox.bbox_to_zx(bbox, velocity)

        # Kalman Update
        self.x, self.P = kalmanfilter.update(self.xp, self.Pp, z, self.R, self.H)

        # Append Tracklet State
        self.states.append(self.x)

    # Tracklet Prediction
    def predict(self):
        # Kalman Prediction
        self.xp, self.Pp = kalmanfilter.predict(self.x, self.P, self.A, self.Q)
        self.pred_states.append(self.xp)

    # Get State in the Specific Frame Index
    def get_state(self, fidx):
        states_idx = (fidx - self.birth_fidx)
        return self.states[states_idx]

    # Update Depth Information
    def update_depth(self, depth_hist, depth_idx):
        self.depth_hist = depth_hist

        max_bin = depth_hist.argmax()
        # Choose Depth Value between the "argmax()" and "argmax()+1" and convert the value into "meters"
        self.depth = ((depth_idx[max_bin] + depth_idx[max_bin+1]) / 2.0) / 1000.0

    # Get 3D Position in camera coordinates (m)
    def get_3d_cam_coord(self, inverse_projection_matrix, is_camera_static=False):
        if is_camera_static is False:
            img_coord = np.array([self.x[0][0], self.x[1][0], 1.0]).reshape((3, 1))
        else:
            # If the camera type is "static", the camera coordinate reference is the bottom-centre of the bbox
            img_coord = np.array([self.x[0][0], (self.x[1][0] + 0.5*self.x[5][0]), 1.0]).reshape((3, 1))
        cam_coord = np.matmul(inverse_projection_matrix, img_coord)
        cam_coord *= self.depth

        img_coord_vel = np.array([self.x[2][0], self.x[3][0], 1.0]).reshape((3, 1))
        cam_coord_vel = np.matmul(inverse_projection_matrix, img_coord_vel)
        cam_coord_vel *= self.depth

        # Consider Robot Coordinates
        # x ==>> camera +z
        # y ==>> camera -x
        # z ==>> camera -y
        self.cam_coord_raw_state = np.array([cam_coord[2][0], -cam_coord[0][0], -cam_coord[1][0],
                                             cam_coord_vel[2][0], -cam_coord_vel[0][0], -cam_coord_vel[1][0]]).reshape(6, 1)

    # Predict 3D Position in camera coordinates via Kalman Filter (Prediction)
    def predict_3d_cam_coord(self):
        self.cam_coord_predicted_state, self.cam_Pp = kalmanfilter.predict(self.cam_coord_estimated_state, self.cam_P,
                                                                           self.cam_A, self.cam_Q)

    # Update 3D Position in camera coordinates via Kalman Filter (Update)
    def update_3d_cam_coord(self):
        self.cam_coord_estimated_state, self.cam_P = kalmanfilter.update(self.cam_coord_predicted_state, self.cam_Pp,
                                                                         self.cam_coord_raw_state, self.cam_R, self.cam_H)

    # Get Roll-Pitch-Yaw
    def compute_rpy(self):
        direction_vector = self.cam_coord_estimated_state[2:-1]

        # Roll needs additional information
        self.roll = 0.0

        # Pitch
        denum = np.sqrt(direction_vector[0][0]*direction_vector[0][0] + direction_vector[1][0]*direction_vector[1][0])
        self.pitch = math.atan2(direction_vector[2][0], denum)

        # Yaw
        self.yaw = math.atan2(direction_vector[1][0], direction_vector[0][0])

    # Update Action Recognition Result
    def update_action(self, action_pred, action_probs):
        pose_list = self.pose_list
        pose_probs = self.pose_probs

        pose_list.append(action_pred)
        pose_probs.append(action_probs)

        if len(pose_list) > 5:
            pose_list.pop(0)
            pose_probs.pop(0)

        self.pose_list = pose_list
        self.pose_probs = pose_probs





















