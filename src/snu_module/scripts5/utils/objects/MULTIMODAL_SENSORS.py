#!/usr/bin/env python
"""
Object Recognition Module (SNU Integrated Module)
for Outdoor Surveillance Robots

    - Object Class Python Script for Multimodal Sensors

"""
import tf2_ros
import rospy
import SENSORS
from SYNC_SUBSCRIBER import SyncSubscriber

from sensor_msgs.msg import PointCloud2, CameraInfo
from nav_msgs.msg import Odometry

""" TF_TRANSFORM related Object """


# TODO: Check for "ros__run_snu_module.py" script from <script4>


class TF_TRANSLATION(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class TF_ROTATION(object):
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class _TF_TRANSFORM(object):
    def __init__(self, translation, rotation):
        self.translation = translation
        self.rotation = rotation


class TF_TRANSFORM(object):
    def __init__(self):
        translation = TF_TRANSLATION(
            x=0.44415, y=0.128996, z=0.238593
        )
        rotation = TF_ROTATION(
            x=0.482089, y=-0.501646, z=0.526684, w=0.488411
        )
        self.transform = _TF_TRANSFORM(
            translation=translation, rotation=rotation
        )


"""'''''''''''''''''''''''''''''"""


class MULTIMODAL_SENSORS_OBJ(object):
    def __init__(self, sensor_opts):
        # Initialize Synchronized Frame Index
        self.fidx = 0

        # Initialize Synchronized Timestamp
        self.sync_timestamp = None

        # Initialize Each Modal Sensor Objects
        self.color = SENSORS.COLOR_SENSOR_OBJ(color_opts=sensor_opts.color)
        self.depth = SENSORS.DEPTH_SENSOR_OBJ(depth_opts=sensor_opts.depth)
        self.infrared = SENSORS.INFRARED_SENSOR_OBJ(infrared_opts=sensor_opts.infrared)
        self.thermal = SENSORS.THERMAL_SENSOR_OBJ(thermal_opts=sensor_opts.thermal)
        self.nightvision = SENSORS.NIGHTVISION_SENSOR_OBJ(nightvision_opts=sensor_opts.nightvision)
        self.lidar = SENSORS.LIDAR_SENSOR_OBJ(lidar_opts=sensor_opts.lidar)

        # Initialize Synchronized Subscriber
        sync_switch_dict = self.__get_multimodal_sensor_valid_dict(sensor_type="img")
        self.sync_subscriber = SyncSubscriber(
            sync_switch_dict=sync_switch_dict,
            rostopic_info_dict=self.__get_multimodal_sensor_rostopic_info_dict(sensor_type="img")
        )

        # Initialize LiDAR Subscriber
        self.lidar_sub = rospy.Subscriber(
            sensor_opts.lidar["rostopic_name"], PointCloud2, self.__lidar_callback
        )

        # Initialize TF_TRANSFORM (LiDAR-to-Color Transform)
        self.tf_transform = None

        # Initialize Odometry Subscriber
        self.odom_sub = rospy.Subscriber(
            sensor_opts.odometry["rostopic_name"], Odometry, self.__odom_callback
        )

        # Initialize CameraInfo Subscribers
        camerainfo_rostopic_name_dict = self.__get_multimodal_sensor_rostopic_info_dict(sensor_type="img")
        for modal, topic_name in camerainfo_rostopic_name_dict.items():
            if topic_name is not None:
                assert isinstance(topic_name, str)
                modal_camerainfo_callback_fn = getattr(self, "__{}_camerainfo_callback".format(modal))
                setattr(self, "{}".format(modal),
                        rospy.Subscriber(topic_name, CameraInfo, modal_camerainfo_callback_fn))


        """ Private Attributes """
        # Get Synchronized Frame Rate of All Sensors
        self.__sync_fps = sensor_opts.sync_fps

        # Initialize LiDAR Message
        self.__lidar_msg = None

        # Initialize Odometry Message
        self.__odom_msg = None

    def __repr__(self):
        sensor_valid_dict = self.__get_multimodal_sensor_valid_dict()
        repr_str = ""
        for key, value in sensor_valid_dict.items():
            if value is True:
                repr_str = repr_str + "+{}".format(key)
        return repr_str

    def update_sensors(self, update_method="synchronous"):
        if update_method == "synchronous":
            while True:
                sync_data = self.__synchronous_update()
                if sync_data is not None:
                    sync_timestamp, sync_frame_dict = sync_data[0], sync_data[1]

                    # Update Synchronized Timestamp and Frame Index
                    self.fidx += 1
                    self.sync_timestamp = sync_timestamp

                    # Update Multimodal Sensor Data (except LiDAR)
                    self.color.update(frame=sync_frame_dict["color"], timestamp=sync_timestamp)
                    self.depth.update(frame=sync_frame_dict["depth"], timestamp=sync_timestamp)
                    self.infrared.update(frame=sync_frame_dict["infrared"], timestamp=sync_timestamp)
                    self.thermal.update(frame=sync_frame_dict["thermal"], timestamp=sync_timestamp)
                    self.nightvision.update(frame=sync_frame_dict["nightvision"], timestamp=sync_timestamp)
                    self.lidar.update_data(lidar_pc_msg=self.__lidar_msg, tf_transform=self.tf_transform)

        elif update_method == "force":
            self.__force_update()
        else:
            raise AssertionError("Input Argument 'update_method' must be either 'synchronous' or 'force'...!")

    def update_tf_transform(self, attempt_thresh=10):
        if self.tf_transform is None:
            # Subscribe for TF_STATIC
            tf_buffer = tf2_ros.Buffer()
            tf_listener = tf2_ros.TransformListener(buffer=tf_buffer)

            tf_static_check_attempts = 0
            while self.tf_transform is None:
                try:
                    self.tf_transform = tf_buffer.lookup_transform(
                        "rgb_frame", "velodyne_frame_from_rgb", rospy.Time(0)
                    )
                except:
                    if tf_static_check_attempts == 0:
                        print("SNU-MODULE : TF_STATIC Transform Unreadable...! >> WAIT FOR A MOMENT...")
                    tf_static_check_attempts += 1

                    if tf_static_check_attempts >= attempt_thresh:
                        print("TF_STATIC: Custom TF Static Transform Loaded...!")
                        self.tf_transform = TF_TRANSFORM()

    def __get_multimodal_sensor_valid_dict(self, sensor_type="all"):
        assert (sensor_type == "all" or sensor_type == "img")
        return {
            "color": self.color.is_valid, "depth": self.depth.is_valid,
            "infrared": self.infrared.is_valid, "thermal": self.thermal.is_valid,
            "nightvision": self.nightvision.is_valid,
            "lidar": self.lidar.is_valid if sensor_type == "all" else False,
        }

    def __get_multimodal_sensor_rostopic_info_dict(self, sensor_type="all"):
        assert (sensor_type == "all" or sensor_type == "img")
        return {
            "color": {"rostopic_name": self.color.rostopic_name, "imgmsg_to_cv2_encoding": self.color.imgmsg_to_cv2_encoding},
            "depth": {"rostopic_name": self.depth.rostopic_name, "imgmsg_to_cv2_encoding": self.depth.imgmsg_to_cv2_encoding},
            "infrared": {"rostopic_name": self.infrared.rostopic_name, "imgmsg_to_cv2_encoding": self.infrared.imgmsg_to_cv2_encoding},
            "thermal": {"rostopic_name": self.thermal.rostopic_name, "imgmsg_to_cv2_encoding": self.thermal.imgmsg_to_cv2_encoding},
            "nightvision": {"rostopic_name": self.nightvision.rostopic_name, "imgmsg_to_cv2_encoding": self.nightvision.imgmsg_to_cv2_encoding},
            "lidar": {"rostopic_name": self.lidar.rostopic_name, "imgmsg_to_cv2_encoding": self.lidar.imgmsg_to_cv2_encoding}
            if sensor_type == "all" else None,
        }

    def __get_multimodal_camerainfo_rostopic_name_dict(self, sensor_type="all"):
        assert (sensor_type == "all" or sensor_type == "img")
        return {
            "color": None if (self.color.camerainfo_rostopic_name == "NULL" or self.color.is_valid is False) else self.color.camerainfo_rostopic_name,
            "depth": None if (self.depth.camerainfo_rostopic_name == "NULL" or self.depth.is_valid is False) else self.depth.camerainfo_rostopic_name,
            "infrared": None if (self.infrared.camerainfo_rostopic_name == "NULL" or self.infrared.is_valid is False) else self.infrared.camerainfo_rostopic_name,
            "thermal": None if (self.thermal.camerainfo_rostopic_name == "NULL" or self.thermal.is_valid is False) else self.thermal.camerainfo_rostopic_name,
            "nightvision": None if (self.nightvision.camerainfo_rostopic_name == "NULL" or self.nightvision.is_valid is False) else self.nightvision.camerainfo_rostopic_name,
            "lidar": None if (self.lidar.camerainfo_rostopic_name == "NULL" or self.lidar.is_valid is False or sensor_type == "all") else self.lidar.camerainfo_rostopic_name
        }

    def __synchronous_update(self):
        return self.sync_subscriber.get_sync_data()

    def __lidar_callback(self, msg):
        self.__lidar_msg = msg

    def __odom_callback(self, msg):
        self.__odom_msg = msg

    def __color_camerainfo_callback(self, msg):
        self.color.sensor_params.update_sensor_params(mode="ROS", msg=msg)

    def __depth_camerainfo_callback(self, msg):
        self.depth.sensor_params.update_sensor_params(mode="ROS", msg=msg)

    def __infrared_camerainfo_callback(self, msg):
        self.infrared.sensor_params.update_sensor_params(mode="ROS", msg=msg)

    def __thermal_camerainfo_callback(self, msg):
        self.thermal.sensor_params.update_sensor_params(mode="ROS", msg=msg)

    def __nightvision_camerainfo_callback(self, msg):
        self.nightvision.sensor_params.update_sensor_params(mode="ROS", msg=msg)


    def __force_update(self, *args, **kwargs):
        raise NotImplementedError()



    # def


if __name__ == "__main__":
    pass
