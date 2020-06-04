"""
SNU Integrated Module v2.05
  - Code that defines object classes
    [1] Tracklet Candidate Class
    [2] Tracklet Class

    [Update Logs]
    - 190916
        (1)
"""

# Import Modules
import copy
import numpy as np
import math
import filterpy.kalman.kalman_filter as kalmanfilter

# Import Source Libraries
import kalman_params as kparams
import snu_utils.bounding_box as fbbox
import snu_utils.patch as ptch

# import rescue.force_thermal_align_iitp_final_night as rgb_t_align


# [1] Tracklet Candidate Class
class TrackletCandidate(object):
    # Initialization
    def __init__(self, bbox, conf, label):
        self.age = 1
        self.asso_dets = [bbox]
        self.conf = [conf]
        self.label = label
        self.is_associated = [True]
        self.z = [fbbox.bbox_to_zx(bbox, np.zeros(2))]
        self.label = int(label[0])

    # Destructor
    def __del__(self):
        pass
        # print("Tracklet Candidate Destroyed")

    # Update
    def update(self, bbox=None, conf=None):
        # Assertion
        if bbox is None and conf is not None:
            assert 0, "Input Method [bbox] cannot be <None> Value!"

        self.age += 1
        if bbox is None:
            self.asso_dets.append(None)
            self.conf.append(None)
            self.is_associated.append(False)
            self.z.append(None)
        else:
            z_bbox = fbbox.bbox_to_zx(bbox)
            velocity = (z_bbox[0:2] - self.z[-1][0:2]).reshape(2)
            self.asso_dets.append(bbox)
            self.conf.append(conf)
            self.is_associated.append(True)
            self.z.append(fbbox.bbox_to_zx(bbox, velocity))


# [2] Tracklet Class
class Tracklet(object):
    # Initialization
    def __init__(self, bbox, conf, label, fidx, trk_id, colorbar, colorbar_refresh_period):
        # Tracklet ID
        self.id = trk_id
        # Tracklet Age
        self.age = 1
        # Tracklet Birth Frame
        self.birth_fidx = fidx
        # Associated Detections (bbox, confidence, label)
        self.asso_dets = [bbox]
        self.conf = [conf]
        self.label = label

        # Association Counter
        self.is_associated = [True]
        # Tracklet Visualization Color
        self.color = colorbar[self.id % colorbar_refresh_period, :] * 255

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

        # Temporary State (RGB ==>> Thermal // for action classification, iitp final demo night version)
        bbox_from_x, _ = fbbox.zx_to_bbox(self.x)
        bbox_from_x[0], bbox_from_x[1] = \
            rgb_t_align.rgb_to_thermal_coord(bbox_from_x[0], bbox_from_x[1])
        bbox_from_x[2], bbox_from_x[3] = \
            rgb_t_align.rgb_to_thermal_coord(bbox_from_x[2], bbox_from_x[3])
        self.x_bbox_thermal_aclassify = bbox_from_x

        # Tracklet States
        self.states = [self.x]

        # 3D Camera Coordinate Position
        self.cam_coord = None
        self.cam_coord_vel = None

        # Roll, Pitch, Yaw
        self.roll, self.pitch, self.yaw = None, None, None

        # Tracklet Predicted States (Next State Prediction)
        self.pred_states = [self.xp]

        # Depth of the Tracklet
        self.depth_hist, self.depth = None, None
        self.raw_depths = []

        # Action Recognition Result of the Tracklet Object
        self.pose_list, self.pose_probs = [], []
        self.pose = None

    # Destructor
    def __del__(self):
        pass
        # print("Tracklet [id: " + str(self.id) + "] Destroyed!")

    # Tracklet Update
    def update(self, bbox=None, conf=None):
        # Assertion
        if bbox is None and conf is not None:
            assert 0, "Input Method [bbox] cannot be <None> Value!"

        # Increase Tracklet Age
        self.age += 1

        # Get Detection (Measurement)
        if bbox is None:
            # If Tracklet is unassociated, replace detection with the previous Kalman prediction
            self.asso_dets.append(None)
            self.conf.append(None)
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
            self.conf.append(conf)
            self.is_associated.append(True)
            z = fbbox.bbox_to_zx(bbox, velocity)

        # Kalman Update
        self.x, self.P = kalmanfilter.update(self.xp, self.Pp, z, self.R, self.H)

        # Temporary State (RGB ==>> Thermal // for action classification, iitp final demo night version)
        bbox_from_x, _ = fbbox.zx_to_bbox(self.x)
        bbox_from_x[0], bbox_from_x[1] = \
            rgb_t_align.rgb_to_thermal_coord(bbox_from_x[0], bbox_from_x[1])
        bbox_from_x[2], bbox_from_x[3] = \
            rgb_t_align.rgb_to_thermal_coord(bbox_from_x[2], bbox_from_x[3])
        self.x_bbox_thermal_aclassify = bbox_from_x

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
    def update_depth(self, depth_hist, depth_idx, depth_update_weights):
        self.depth_hist = depth_hist

        if depth_hist == []:
            pass
        else:
            max_bin = depth_hist.argmax()
            # Choose Depth Value between the "argmax()" and "argmax()+1" and convert the value into "meters"
            # self.depth = ((depth_idx[max_bin] + depth_idx[max_bin+1]) / 2.0) / 1000.0
            hist_depth_value = ((depth_idx[max_bin] + depth_idx[max_bin + 1]) / 2.0) / 1000.0

            self.raw_depths.append(hist_depth_value)

        # Pop Raw Depths
        while len(self.raw_depths) > len(depth_update_weights):
            self.raw_depths.pop(0)

        # Weight Multiplication
        if depth_hist == []:
            self.depth = -1.0
        else:
            _depth_update_weights = depth_update_weights[0:len(self.raw_depths)]
            weighted_depths = [a * b for a, b in zip(self.raw_depths, _depth_update_weights)]
            valid_weighted_depths = [(idx, weighted_depth) for idx, weighted_depth in enumerate(weighted_depths) if weighted_depth >= 0]

            if len(valid_weighted_depths) == 0:
                self.depth = -1.0
            else:
                _valid_weight_idx = np.array(valid_weighted_depths, dtype=int)[:, 0].tolist()
                self.depth = sum(np.array(valid_weighted_depths)[:, 1]) / sum([_depth_update_weights[idx] for idx in _valid_weight_idx])

        # if len(self.raw_depths) == len(depth_update_weights):
        #     weighted_depths = [a * b for a, b in zip(self.raw_depths, depth_update_weights)]
        # else:
        #     # Less than weight list number
        #     weighted_depths =
        #
        #
        # # Get Valid Depths
        # valid_depths = [i for i in self.raw_depths if i >= 0]
        #
        # # Pop Until Weight Length Match
        # if len(valid_depths) < len(depth_update_weights):
        #     if valid_depths == []:
        #         self.depth = -1.0
        #     else:
        #         self.depth = valid_depths[-1]
        # else:
        #     # Pop Until Length Matches
        #     while len(valid_depths) > len(depth_update_weights):
        #         valid_depths.pop(0)
        #
        #     # Weighted Sum
        #     self.depth = sum([a * b for a, b in zip(valid_depths, depth_update_weights)]) / sum(depth_update_weights)
        #
        # # Pop Raw Depths
        # while len(self.raw_depths) > len(depth_update_weights):
        #     self.raw_depths.pop(0)

    # Get 3D Position in camera coordinates (m)
    def get_3d_cam_coord(self, inverse_projection_matrix, is_camera_static=False):
        if is_camera_static is True:
            # If the camera type is "static", the camera coordinate reference is the bottom-centre of the bbox
            img_coord = np.array([self.x[0][0], (self.x[1][0] + 0.5*self.x[5][0]), 1.0]).reshape((3, 1))
            cam_coord = np.matmul(inverse_projection_matrix, img_coord)
        else:
            img_coord = np.array([self.x[0][0], self.x[1][0], 1.0]).reshape((3, 1))
            cam_coord = np.matmul(inverse_projection_matrix, img_coord)
            cam_coord *= self.depth

        img_coord_vel = np.array([self.x[2][0], self.x[3][0], 1.0]).reshape((3, 1))
        cam_coord_vel = np.matmul(inverse_projection_matrix, img_coord_vel)
        cam_coord_vel *= self.depth

        # Consider Robot Coordinates
        if is_camera_static is not True:
            # x ==>> camera +z
            # y ==>> camera -x
            # z ==>> camera -y
            self.cam_coord = np.array([cam_coord[2][0], -cam_coord[0][0], -cam_coord[1][0]]).reshape((3, 1))
            self.cam_coord_vel = np.array([cam_coord_vel[2][0], -cam_coord_vel[0][0], -cam_coord_vel[1][0]]).reshape((3, 1))

    # Get 3D Position in camera coordinates (m)
    def get_3d_cam_coord_2(self, intrinsic_matrix, rotation_matrix, translation_vector, is_camera_static=False):
        if is_camera_static is False:
            img_coord = np.array([self.x[0][0], self.x[1][0], 1.0]).reshape((3, 1))
        else:
            # If the camera type is "static", the camera coordinate reference is the bottom-centre of the bbox
            img_coord = np.array([self.x[0][0], (self.x[1][0] + 0.5 * self.x[5][0]), 1.0]).reshape((3, 1))

        # dfocal_matrix = copy.deepcopy(intrinsic_matrix)
        # dfocal_matrix[0, 2], dfocal_matrix[1, 2] = 0, 0
        # c_vector = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2], 1]).reshape((3, 1))
        # int_inv = np.linalg.inv(intrinsic_matrix)
        fx = intrinsic_matrix[0][0]
        fy = intrinsic_matrix[1][1]
        cx = intrinsic_matrix[0][2]
        cy = intrinsic_matrix[1][2]
        nx = (img_coord[0][0] - cx) / fx
        ny = (img_coord[1][0] - cy) / fy
        Pc = np.array([nx, ny, 1]).reshape(3, 1)
        cam_coord = np.matmul(rotation_matrix.transpose(), (Pc + np.matmul(rotation_matrix, translation_vector)))
        if self.id == 2:
            print("x", self.x.reshape(1, -1).astype(int))
            print("box bottom", img_coord.reshape(1, -1).astype(int))

            # print(self.x.reshape(1, -1), cam_coord.reshape(1, -1))
        # cam_coord = np.matmul(np.matmul(rotation_matrix.transpose(), int_inv), img_coord)
        # dfocal_inv = np.linalg.inv(dfocal_matrix)
        # cam_coord = \
        #     np.matmul(rotation_matrix.transpose(), np.matmul(dfocal_inv, (img_coord - c_vector))) + \
        #     np.matmul(rotation_matrix, translation_vector)
        self.cam_coord = cam_coord

        # cam_coord *= self.depth

        img_coord_vel = np.array([self.x[2][0], self.x[3][0], 1.0]).reshape((3, 1))
        # cam_coord_vel = np.matmul(inverse_projection_matrix, img_coord_vel)
        # cam_coord_vel *= self.depth
        cam_coord_vel = np.ones((3, 1)).reshape(3, 1)

        # Consider Robot Coordinates
        # x ==>> camera +z
        # y ==>> camera -x
        # z ==>> camera -y
        # self.cam_coord_raw_state = np.array([cam_coord[2][0], -cam_coord[0][0], -cam_coord[1][0],
        #                                      cam_coord_vel[2][0], -cam_coord_vel[0][0], -cam_coord_vel[1][0]]).reshape(6, 1)
        self.cam_coord_raw_state = np.concatenate((cam_coord[0:3], cam_coord_vel[0:3]), axis=1).reshape((6, 1))
        # self.cam_coord_raw_state = np.zeros(())

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
        direction_vector = self.cam_coord_vel

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





















