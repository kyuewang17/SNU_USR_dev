"""
SNU Integrated Module v4.0
    - Classes for ROS Multimodal Sensors, with Synchronization (sync part from KIRO)
    - Message Wrapper for Publishing Message to ETRI Agent

"""
# Import Modules
import cv2
import random
import matplotlib.cm
import numpy as np
import rospy
import ros_numpy
import image_geometry
import pyquaternion
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_euler

# Import ROS Messages
from osr_msgs.msg import Track, Tracks, BoundingBox
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud, transform_to_kdl
from rospy.numpy_msg import numpy_msg

# Import KIRO's Synchronized Subscriber
from utils.sync_subscriber import SyncSubscriber


# RUN SNU Module Coverage Class Source
class coverage(object):
    def __init__(self, opts, is_sensor_param_file=True):
        # Load Options
        self.opts = opts

        # Initialize Point Cloud ROS Message Variable
        self.lidar_msg = None

        # Odometry Message (Pass-through Variable)
        self.odometry_msg = None

        # TF Static-related Variables
        self.tf_transform = None

        # Initialize Modal Classes
        self.color = ros_sensor_image(modal_type="color")
        self.disparity = ros_sensor_image(modal_type="disparity")
        self.thermal = ros_sensor_image(modal_type="thermal")
        self.infrared = ros_sensor_image(modal_type="infrared")
        self.nightvision = ros_sensor_image(modal_type="nightvision")
        self.lidar = ros_sensor_lidar(modal_type="lidar")

        # CvBridge for Publisher
        self.pub_bridge = CvBridge()

        # Subscriber for Point Cloud
        self.pc_sub = rospy.Subscriber(
            opts.sensors.lidar["rostopic_name"], PointCloud2, self.point_cloud_callback
        )

        # Subscriber for Odometry
        self.odom_sub = rospy.Subscriber(
            opts.sensors.odometry["rostopic_name"], Odometry, self.odometry_callback
        )

        if is_sensor_param_file is False:
            # Subscriber for Color CameraInfo
            self.color_camerainfo_sub = rospy.Subscriber(
                opts.sensors.color["camerainfo_rostopic_name"], (CameraInfo), self.color_camerainfo_callback
            )

            # Subscriber for Disparity CameraInfo
            self.disparity_camerainfo_sub = rospy.Subscriber(
                opts.sensors.disparity["camerainfo_rostopic_name"], (CameraInfo), self.disparity_camerainfo_callback
            )

            # Subscriber for Infrared CameraInfo
            self.infrared_camerainfo_sub = rospy.Subscriber(
                opts.sensors.infrared["camerainfo_rostopic_name"], (CameraInfo), self.infrared_camerainfo_callback
            )

        # ROS Publisher
        self.tracks_pub = rospy.Publisher(
            opts.publish_mesg["tracks"], Tracks, queue_size=1
        )

        # ROS SNU Result Publisher
        self.det_result_pub = rospy.Publisher(
            opts.publish_mesg["det_result_rostopic_name"], Image, queue_size=1
        )
        self.trk_acl_result_pub = rospy.Publisher(
            opts.publish_mesg["trk_acl_result_rostopic_name"], Image, queue_size=1
        )
        self.top_view_result_pub = rospy.Publisher(
            opts.publish_mesg["trk_top_view_rostopic_name"], Image, queue_size=1
        )

    # Point Cloud Callback Function
    def point_cloud_callback(self, msg):
        self.lidar_msg = msg

    # Odometry Callback Function
    def odometry_callback(self, msg):
        self.odometry_msg = msg

    # Color CameraInfo Callback Function
    def color_camerainfo_callback(self, msg):
        if self.color.sensor_params is None:
            self.color.update_sensor_params_rostopic(msg=msg)

    # Disparity Camerainfo Callback Function
    def disparity_camerainfo_callback(self, msg):
        if self.disparity.sensor_params is None:
            self.disparity.update_sensor_params_rostopic(msg=msg)

    # Infrared Camerainfo Callback Function
    def infrared_camerainfo_callback(self, msg):
        if self.infrared.sensor_params is None:
            self.infrared.update_sensor_params_rostopic(msg=msg)

    # Publish Tracks
    def publish_tracks(self, tracklets, odometry_msg):
        out_tracks = wrap_tracks(trackers=tracklets, odometry=odometry_msg)
        self.tracks_pub.publish(out_tracks)

    # Publish SNU Result Image ( DET / TRK + ACL )
    def publish_snu_result_image(self, result_frame_dict):
        for module, result_frame in result_frame_dict.items():
            if result_frame is not None:
                if module == "det":
                    if self.opts.detector.is_result_publish is True:
                        self.det_result_pub.publish(
                            self.pub_bridge.cv2_to_imgmsg(
                                result_frame, "rgb8"
                            )
                        )
                elif module == "trk_acl":
                    if self.opts.tracker.is_result_publish is True:
                        self.trk_acl_result_pub.publish(
                            self.pub_bridge.cv2_to_imgmsg(
                                result_frame, "rgb8"
                            )
                        )
                else:
                    assert 0, "Undefined Module!"

    def update_all_modal_data(self, sync_data):
        sync_stamp = sync_data[0]
        sync_frame_dict = sync_data[1]

        # Update Modal Frames
        self.color.update_data(frame=sync_frame_dict["color"], stamp=sync_stamp)

        self.disparity.update_data(frame=sync_frame_dict["aligned_disparity"], stamp=sync_stamp)
        self.disparity.update_raw_data(raw_data=sync_frame_dict["disparity"])

        self.thermal.update_data(frame=sync_frame_dict["thermal"], stamp=sync_stamp)

        self.infrared.update_data(frame=sync_frame_dict["infrared"], stamp=sync_stamp)

        self.nightvision.update_data(frame=sync_frame_dict["nightvision"], stamp=sync_stamp)

        self.lidar.update_data(
            lidar_pc_msg=self.lidar_msg, stamp=self.lidar_msg.header.stamp,
            tf_transform=self.tf_transform
        )

    def gather_all_modal_data(self):
        sensor_data = {
            "color": self.color,
            "disparity": self.disparity,
            "thermal": self.thermal,
            "infrared": self.infrared,
            "nightvision": self.nightvision,
            "lidar": self.lidar
        }
        return sensor_data


# Sensor Parameter Base Class
class sensor_params(object):
    def __init__(self, param_precision):
        # Set Parameter Precision
        self.param_precision = param_precision

        # Set Projection Matrix and its Pseudo-inverse Matrix
        self.projection_matrix = None
        self.pinv_projection_matrix = None

    # Update Parameters
    def update_params(self, param_argument):
        raise NotImplementedError()


# Sensor Parameter Class (Rostopic)
class sensor_params_rostopic(sensor_params):
    def __init__(self, param_precision=np.float32):
        super(sensor_params_rostopic, self).__init__(param_precision)

        """ Initialize Camera Parameter Matrices
        ----------------------------------------
        D: Distortion Matrix (5x1)
        K: Intrinsic Matrix (3x3)
        R: Rotation Matrix (3x3)
        P: Projection Matrix (3x4)
        ----------------------------------------
        """
        self.D, self.K, self.R, self.P = None, None, None, None

    def update_params(self, msg):
        self.D = np.asarray(msg.D).reshape((5, 1))  # Distortion Matrix
        self.K = np.asarray(msg.K).reshape((3, 3))  # Intrinsic Matrix
        self.R = np.asarray(msg.R).reshape((3, 3))  # Rotation Matrix
        self.P = np.asarray(msg.P).reshape((3, 4))  # Projection Matrix

        self.projection_matrix = self.P
        self.pinv_projection_matrix = np.linalg.pinv(self.P)


# Sensor Parameter Class (File)
class sensor_params_file(sensor_params):
    def __init__(self, param_precision=np.float32):
        super(sensor_params_file, self).__init__(param_precision)

        # Initialize Intrinsic-related Variables
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        self.w = None

        # Initialize Translation-related Variables
        self.x, self.y, self.z = None, None, None

        # Initialize Pan(yaw) / Tilt(pitch) / Roll Variables
        self.pan, self.tilt, self.roll = None, None, None

        # Set Camera Parameter Matrices
        self.intrinsic_matrix, self.extrinsic_matrix, self.rotation_matrix = None, None, None

    # Update Parameter Variables
    def update_params(self, param_array):
        # Intrinsic-related
        self.fx, self.fy, self.cx, self.cy = \
            param_array[0], param_array[1], param_array[2], param_array[3]
        self.w = param_array[4]

        # Translation-related
        self.x, self.y, self.z = param_array[5], param_array[6], param_array[7]

        # Pan / Tilt / Roll
        self.pan, self.tilt, self.roll = param_array[8], param_array[9], param_array[10]

        # Intrinsic Matrix < 3 x 4 >
        self.intrinsic_matrix = np.array([[self.fx, self.w, self.cx, 0],
                                          [0, self.fy, self.cy, 0],
                                          [0, 0, 1, 0]], dtype=self.param_precision)

        # Rotation Matrix
        self.rotation_matrix = self.convert_ptr_to_rotation()

        # Extrinsic Matrix < 4 x 4 >
        translation_vector = np.matmul(
            self.rotation_matrix,
            np.array([self.x, self.y, self.z], dtype=self.param_precision).reshape((3, 1))
        )
        self.extrinsic_matrix = np.block(
            [np.vstack((self.rotation_matrix, np.zeros((1, 3)))), np.append(translation_vector, 1).reshape(-1, 1)]
        )

        # Get Projection Matrix and its Pseudo-inverse
        self.projection_matrix = np.matmul(self.intrinsic_matrix, self.extrinsic_matrix)
        self.pinv_projection_matrix = np.linalg.pinv(self.projection_matrix)

    # Convert PTR to Rotation Matrix
    def convert_ptr_to_rotation(self):
        r11 = np.sin(self.pan) * np.cos(self.roll) - np.cos(self.pan) * np.sin(self.tilt) * np.sin(self.roll)
        r12 = -np.cos(self.pan) * np.cos(self.roll) - np.sin(self.pan) * np.sin(self.tilt) * np.sin(self.roll)
        r13 = np.cos(self.tilt) * np.sin(self.roll)
        r21 = np.sin(self.pan) * np.sin(self.roll) + np.sin(self.tilt) * np.cos(self.pan) * np.cos(self.roll)
        r22 = -np.cos(self.pan) * np.sin(self.roll) + np.sin(self.tilt) * np.sin(self.pan) * np.cos(self.roll)
        r23 = -np.cos(self.tilt) * np.cos(self.roll)
        r31 = np.cos(self.tilt) * np.cos(self.pan)
        r32 = np.cos(self.tilt) * np.sin(self.pan)
        r33 = np.sin(self.tilt)

        rotation_matrix = np.array([[r11, r12, r13],
                                    [r21, r22, r23],
                                    [r31, r32, r33]], dtype=self.param_precision)
        return rotation_matrix


# Multimodal ROS Sensor Managing Class
class ros_sensor(object):
    def __init__(self, modal_type, stamp):
        # Modal Type (becomes the class representation)
        self.modal_type = modal_type

        # Raw Data
        self.raw_data = None

        # Modal Timestamp
        self.timestamp = stamp

        # Modal Sensor Parameters
        self.camerainfo_msg = None
        self.sensor_params = None

    def __repr__(self):
        return self.modal_type

    def update_data(self, data, stamp):
        raise NotImplementedError()

    def update_raw_data(self, raw_data):
        """
        In-built Function for Disparity
        """
        self.raw_data = raw_data

    def get_data(self):
        raise NotImplementedError()

    def update_stamp(self, stamp):
        self.timestamp = stamp

    def update_sensor_params_rostopic(self, msg):
        if msg is not None:
            # Save CameraInfo Message
            self.camerainfo_msg = msg

            # Initialize Sensor Parameter Object (from rostopic)
            self.sensor_params = sensor_params_rostopic(param_precision=np.float32)

            # Update Parameters
            self.sensor_params.update_params(msg=msg)


class ros_sensor_image(ros_sensor):
    def __init__(self, modal_type, frame=None, stamp=None):
        super(ros_sensor_image, self).__init__(modal_type=modal_type, stamp=stamp)

        # Modal Frame
        self.frame = frame
        self.processed_frame = None

        # Initialize Frame Shape
        self.WIDTH, self.HEIGHT = None, None

        # Variable for Moving Visualization Window
        self.__vis_window_moved = False

    def __add__(self, other):
        assert isinstance(other, ros_sensor_image), "Argument 'other' must be the same type!"

        # Concatenate Channel-wise
        concat_frame = np.dstack((self.get_data(), other.get_data()))

        # Get Concatenated Modal Type
        concat_modal_type = "{}+{}".format(self, other)

        # Return Concatenated ROS Sensor Image Class
        return ros_sensor_image(concat_modal_type, concat_frame, self.timestamp)

    def get_data(self, is_processed=True):
        if self.processed_frame is not None and is_processed is True:
            return self.processed_frame
        else:
            return self.frame

    def update_data(self, frame, stamp):
        self.frame = frame
        self.update_stamp(stamp=stamp)

        if self.WIDTH is None:
            self.WIDTH = frame.shape[1]
        if self.HEIGHT is None:
            self.HEIGHT = frame.shape[0]

    def process_data(self, disparity_sensor_opts):
        assert "{}".format(self) == "disparity"

        frame = self.frame.astype(np.float32)
        self.processed_frame = np.where(
            (frame < disparity_sensor_opts["clip_distance"]["min"]) |
            (frame > disparity_sensor_opts["clip_distance"]["max"]),
            disparity_sensor_opts["clip_value"], frame
        )

    # Get Normalized Data
    def get_normalized_data(self, min_value=0.0, max_value=1.0):
        frame = self.get_data()
        frame_max_value, frame_min_value = frame.max(), frame.min()
        minmax_normalized_frame = (frame - frame_min_value) / (frame_max_value - frame_min_value)
        normalized_frame = min_value + (max_value - min_value) * minmax_normalized_frame
        return normalized_frame

    # Visualize Frame
    def visualize_frame(self, is_processed=True):
        # Get Frame
        vis_frame = self.get_data(is_processed=is_processed)

        if vis_frame is not None:
            # Get Modal Type Name
            modal_type = "{}".format(self)

            # OpenCV Window Name
            winname = "[{}]".format(modal_type)

            # Make NamedWindow
            cv2.namedWindow(winname)

            # Move Window
            cv2.moveWindow(winname=winname, x=1000, y=500)

            # IMSHOW
            if modal_type.__contains__("color") is True:
                cv2.imshow(winname, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            else:
                cv2.imshow(winname, vis_frame)

            cv2.waitKey(1)

    def _empty_frame(self):
        self.frame, self.processed_frame = None, None

    def _empty_stamp(self):
        self.timestamp = None


class ros_sensor_lidar(ros_sensor):
    def __init__(self, modal_type, lidar_pc_msg=None, stamp=None):
        super(ros_sensor_lidar, self).__init__(modal_type=modal_type, stamp=stamp)

        # Modal LiDAR PointCloud Message (format: ROS "PointCloud2")
        self.raw_pc_msg = lidar_pc_msg

        # Initialize Rotation and Translation Matrices btw (Color and LiDAR) Camera
        self.R__color, self.T__color = None, None
        self.projected_cloud = None

        # Define Camera Model Function Variable (differs according to the input camerainfo)
        self.CAMERA_MODEL = image_geometry.PinholeCameraModel()

        # LiDAR Point Cloud XYZ, Distance and Colors
        self.cloud = None
        self.pc_distance, self.pc_colors = None, None

    def __add__(self, other):
        assert isinstance(other, ros_sensor_image)
        # Get Added Sensor Data Frame
        other_frame = other.get_data(is_processed=False)

        # Initialize Projected LiDAR Sensor Data
        projected_sensor_data = ros_sensor_image(modal_type="{}+{}".format(other, "LiDAR"))

        # Project LiDAR Sensor Data to the Added Sensor Data
        tmp = self.project_xyz_to_uv_by_sensor_data(sensor_data=other)
        if tmp is not None:
            uv_arr, distances, colors = tmp[0], tmp[1], tmp[2]
            for idx in range(len(uv_arr)):
                uv_point = tuple(uv_arr[idx])
                cv2.circle(
                    img=other_frame, center=uv_point, radius=2, color=colors[idx], thickness=-1
                )

            # Update Projected LiDAR Sensor Data
            projected_sensor_data.update_data(frame=other_frame, stamp=self.timestamp)

        else:
            projected_sensor_data = None

        return projected_sensor_data

    def get_data(self):
        raise NotImplementedError()

    def load_pc_xyz_data(self):
        if self.projected_cloud is not None:
            # # Convert ROS PointCloud2 to Cloud Data (XYZRGB)
            # cloud = ros_numpy.point_cloud2.pointcloud2_to_array(self.tf_pc_msg)
            # cloud = np.asarray(cloud.tolist())
            cloud = self.projected_cloud

            # Filer-out Points in Front of Camera
            inrange = np.where((cloud[:, 0] > -8) &
                               (cloud[:, 0] < 8) &
                               (cloud[:, 1] > -5) &
                               (cloud[:, 1] < 5) &
                               (cloud[:, 2] > -0) &
                               (cloud[:, 2] < 30))
            max_intensity = np.max(cloud[:, -1])
            cloud = cloud[inrange[0]]
            self.cloud = cloud

            # Straight Distance From Camera
            self.pc_distance = np.sqrt(cloud[:, 0] * cloud[:, 0] + cloud[:, 1] * cloud[:, 1] + cloud[:, 2] * cloud[:, 2])

            # Color map for the points
            cmap = matplotlib.cm.get_cmap('jet')
            self.pc_colors = cmap(cloud[:, -1] / max_intensity) * 255  # intensity color view

    def update_data(self, lidar_pc_msg, stamp=None, tf_transform=None):
        # kdl = transform_to_kdl(tf_transform)
        # self.tf_pc_msg = do_transform_cloud(
        #     cloud=self.raw_pc_msg, transform=tf_transform
        # )

        # Update LiDAR Message
        self.raw_pc_msg = lidar_pc_msg

        # Update Stamp
        if stamp is not None:
            self.update_stamp(stamp=stamp)
        else:
            self.update_stamp(stamp=lidar_pc_msg.header.stamp)

        # Update Rotation and Translation Matrices
        if self.R__color is None:
            self.R__color = pyquaternion.Quaternion(
                tf_transform.transform.rotation.w,
                tf_transform.transform.rotation.x,
                tf_transform.transform.rotation.y,
                tf_transform.transform.rotation.z,
            ).rotation_matrix
        if self.T__color is None:
            self.T__color = np.array([
                tf_transform.transform.translation.x,
                tf_transform.transform.translation.y,
                tf_transform.transform.translation.z
            ]).reshape(3, 1)

        # Project Point Cloud
        if self.R__color is not None and self.T__color is not None:
            pc = np.array(ros_numpy.numpify(self.raw_pc_msg).tolist())
            self.projected_cloud = np.dot(pc[:, 0:3], self.R__color.T) + self.T__color.T
        else:
            self.projected_cloud = None

    def project_xyz_to_uv_by_sensor_data(self, sensor_data, random_sample_number=0):
        """
        Project XYZ PointCloud Numpy Array Data using Input Sensor Data's CameraInfo
        """
        assert isinstance(sensor_data, ros_sensor_image)

        return self.project_xyz_to_uv(
            camerainfo_msg=sensor_data.camerainfo_msg,
            frame_width=sensor_data.WIDTH, frame_height=sensor_data.HEIGHT,
            random_sample_number=random_sample_number
        )

    def project_xyz_to_uv(self, camerainfo_msg, frame_width, frame_height, random_sample_number=0):
        if self.cloud is not None:
            # Update Camera Parameter to Pinhole Camera Model
            self.CAMERA_MODEL.fromCameraInfo(msg=camerainfo_msg)

            # Get Camera Parameters
            fx, fy = self.CAMERA_MODEL.fx(), self.CAMERA_MODEL.fy()
            cx, cy = self.CAMERA_MODEL.cx(), self.CAMERA_MODEL.cy()
            Tx, Ty = self.CAMERA_MODEL.Tx(), self.CAMERA_MODEL.Ty()

            px = (fx * self.cloud[:, 0] + Tx) / self.cloud[:, 2] + cx
            py = (fy * self.cloud[:, 1] + Ty) / self.cloud[:, 2] + cy

            # Stack UV Image Coordinate Points
            uv = np.column_stack((px, py))
            inrange = np.where((uv[:, 0] >= 0) & (uv[:, 1] >= 0) &
                               (uv[:, 0] < frame_width) & (uv[:, 1] < frame_height))
            uv_array = uv[inrange[0]].round().astype('int')
            pc_distances = self.pc_distance[inrange[0]]
            pc_colors = self.pc_colors[inrange[0]]

            if random_sample_number > 0:
                rand_indices = sorted(random.sample(range(len(uv_array)), random_sample_number))
                uv_array = uv_array[rand_indices]
                pc_distances = pc_distances[rand_indices]
                pc_colors = pc_colors[rand_indices]

            return uv_array, pc_distances, pc_colors

        else:
            return None

    def project_xyz_to_uv_inside_bbox(self, camerainfo_msg, bbox, random_sample_number=0):
        if self.cloud is not None:
            # Update Camera Parameter to Pinhole Camera Model
            self.CAMERA_MODEL.fromCameraInfo(msg=camerainfo_msg)

            # Get Camera Parameters
            fx, fy = self.CAMERA_MODEL.fx(), self.CAMERA_MODEL.fy()
            cx, cy = self.CAMERA_MODEL.cx(), self.CAMERA_MODEL.cy()
            Tx, Ty = self.CAMERA_MODEL.Tx(), self.CAMERA_MODEL.Ty()

            px = (fx * self.cloud[:, 0] + Tx) / self.cloud[:, 2] + cx
            py = (fy * self.cloud[:, 1] + Ty) / self.cloud[:, 2] + cy

            # Stack UV Image Coordinate Points
            uv = np.column_stack((px, py))
            inrange = np.where((uv[:, 0] >= bbox[0]) & (uv[:, 1] >= bbox[1]) &
                               (uv[:, 0] < bbox[2]) & (uv[:, 1] < bbox[3]))
            uv_array = uv[inrange[0]].round().astype('int')
            pc_distances = self.pc_distance[inrange[0]]
            pc_colors = self.pc_colors[inrange[0]]

            if random_sample_number > 0:
                random_sample_number = min(random_sample_number, len(uv_array))
                rand_indices = sorted(random.sample(range(len(uv_array)), random_sample_number))
                uv_array = uv_array[rand_indices]
                pc_distances = pc_distances[rand_indices]
                pc_colors = pc_colors[rand_indices]

            return uv_array, pc_distances, pc_colors

        else:
            return None


# Synchronized Subscriber (from KIRO, SNU Adaptation)
class snu_SyncSubscriber(SyncSubscriber):
    def __init__(self, ros_sync_switch_dict, options):
        super(snu_SyncSubscriber, self).__init__(ros_sync_switch_dict, options)

    def get_sync_data(self):
        self.lock_flag.acquire()
        if self.sync_flag is False:
            self.lock_flag.release()
            return None
        else:
            result_sync_frame_dict = {
                "color": self.sync_color, "disparity": self.sync_depth, "aligned_disparity": self.sync_aligned_depth,
                "thermal": self.sync_thermal, "infrared": self.sync_ir, "nightvision": self.sync_nv1
            }
            self.lock_flag.release()
            return self.sync_stamp, result_sync_frame_dict


# Function for Publishing SNU Module Result to ETRI Module
def wrap_tracks(trackers, odometry):
    out_tracks = Tracks()

    # For the header stamp, record current time
    out_tracks.header.stamp = rospy.Time.now()

    # Odometry Information Passes Through SNU Module
    if odometry is not None:
        out_tracks.odom = odometry

    # For each Tracklets,
    for _, tracker in enumerate(trackers):
        # Get Tracklet State
        track_state = tracker.states[-1]
        if len(tracker.states) > 1:
            track_prev_state = tracker.states[-2]
        else:
            # [x,y,dx,dy,w,h]
            track_prev_state = np.zeros(7).reshape((7, 1))
        track_cam_coord_state = tracker.c3

        # Initialize Track
        track = Track()

        # Tracklet ID
        # important: set the Tracklet ID to modulus of 256 since the type is < uint8 >
        track.id = tracker.id % 256

        # Tracklet Object Type (1: Person // 2: Car)
        track.type = tracker.label

        # Tracklet Action Class (Posture)
        # 1: Stand, 2: Sit, 3: Lie
        # (publish if only tracklet object type is person)
        if tracker.label == 1:
            track.posture = (tracker.pose if tracker.pose is not None else 0)
        else:
            track.posture = 0

        # Bounding Box Position [bbox_pose]
        track_bbox = BoundingBox()
        track_bbox.x = np.uint32(track_state[0][0])
        track_bbox.y = np.uint32(track_state[1][0])
        track_bbox.height = np.uint32(track_state[6][0])
        track_bbox.width = np.uint32(track_state[5][0])
        track.bbox_pose = track_bbox

        # Bounding Box Velocity [bbox_velocity]
        track_d_bbox = BoundingBox()
        track_d_bbox.x = np.uint32(track_state[3][0])
        track_d_bbox.y = np.uint32(track_state[4][0])
        track_d_bbox.height = np.uint32((track_state - track_prev_state)[6][0])
        track_d_bbox.width = np.uint32((track_state - track_prev_state)[5][0])
        track.bbox_velocity = track_d_bbox

        # [pose]
        cam_coord_pose = Pose()
        cam_coord_position = Point()
        cam_coord_orientation = Quaternion()

        cam_coord_position.x = np.float64(track_cam_coord_state[0][0])
        cam_coord_position.y = np.float64(track_cam_coord_state[1][0])
        cam_coord_position.z = np.float64(track_cam_coord_state[2][0])

        # Convert to Quaternion
        q = quaternion_from_euler(tracker.roll, tracker.pitch, tracker.yaw)
        cam_coord_orientation.x = np.float64(q[0])
        cam_coord_orientation.y = np.float64(q[1])
        cam_coord_orientation.z = np.float64(q[2])
        cam_coord_orientation.w = np.float64(q[3])

        cam_coord_pose.position = cam_coord_position
        cam_coord_pose.orientation = cam_coord_orientation
        track.pose = cam_coord_pose

        # [twist]
        cam_coord_twist = Twist()
        cam_coord_linear = Vector3()
        cam_coord_angular = Vector3()

        cam_coord_linear.x = np.float64(track_cam_coord_state[3][0])
        cam_coord_linear.y = np.float64(track_cam_coord_state[4][0])
        cam_coord_linear.z = np.float64(track_cam_coord_state[5][0])

        cam_coord_angular.x = np.float64(0)
        cam_coord_angular.y = np.float64(0)
        cam_coord_angular.z = np.float64(0)

        cam_coord_twist.linear = cam_coord_linear
        cam_coord_twist.angular = cam_coord_angular
        track.twist = cam_coord_twist

        # Append to Tracks
        out_tracks.tracks.append(track)

    return out_tracks
