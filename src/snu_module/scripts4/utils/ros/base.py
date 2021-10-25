"""
SNU Integrated Module v4.5
    - Coverage Class Module for ROS-embedded Integrated Algorithm

"""
from rospy import Subscriber, Publisher
from cv_bridge import CvBridge

from wrapper import wrap_tracks
from sensors import ros_sensor_image, ros_sensor_disparity, ros_sensor_lidar

# Import ROS Message Types
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from osr_msgs.msg import Tracks, Annotations, Evaluator
from nav_msgs.msg import Odometry


class backbone(object):
    def __init__(self, opts):
        # Load Options
        self.opts = opts

        # Initialize Point Cloud ROS Message Variable
        self.lidar_msg = None

        # Odometry Message (Pass-through Variable)
        self.odometry_msg = None

        # NOTE: Annotations Message (Pass-through Variable - Tentative for MOT Evaluation)
        self.annos_msg = {
            "color": None, "thermal": None
        }

        # TF Static-related Variables
        self.tf_transform = None

        # Initialize Modal Classes
        self.color = ros_sensor_image(modal_type="color") if opts.modal_switch_dict["color"] is True else None
        self.disparity = ros_sensor_disparity() if opts.modal_switch_dict["disparity"] is True else None
        self.thermal = ros_sensor_image(modal_type="thermal") if opts.modal_switch_dict["thermal"] is True else None
        self.infrared = ros_sensor_image(modal_type="infrared") if opts.modal_switch_dict["infrared"] is True else None
        self.nightvision = ros_sensor_image(modal_type="nightvision") if opts.modal_switch_dict["nightvision"] is True else None
        self.lidar = ros_sensor_lidar() if opts.modal_switch_dict["lidar"] is True else None

        # CvBridge for Publisher
        self.pub_bridge = CvBridge()

        # Subscriber for Point Cloud
        self.pc_sub = Subscriber(
            opts.sensors.lidar["rostopic_name"], PointCloud2, self.point_cloud_callback
        )

        # Subscriber for Odometry
        self.odom_sub = Subscriber(
            opts.sensors.odometry["rostopic_name"], Odometry, self.odometry_callback
        )

        # CameraInfo Subscribers
        # Color CameraInfo Subscriber
        if self.color is not None:
            self.color_camerainfo_sub = Subscriber(
                opts.sensors.color["camerainfo_rostopic_name"], CameraInfo,
                self.color_camerainfo_callback
            )

        # Disparity CameraInfo Subscriber
        if self.disparity is not None:
            self.disparity_camerainfo_sub = Subscriber(
                opts.sensors.disparity["camerainfo_rostopic_name"], CameraInfo,
                self.disparity_camerainfo_callback
            )

        # Infrared CameraInfo Subscriber
        if self.infrared is not None:
            self.infrared_camerainfo_sub = Subscriber(
                opts.sensors.infrared["camerainfo_rostopic_name"], CameraInfo,
                self.infrared_camerainfo_callback
            )

        # Thermal CameraInfo Subscriber
        if self.thermal is not None:
            self.thermal_camerainfo_sub = Subscriber(
                opts.sensors.thermal["camerainfo_rostopic_name"], CameraInfo,
                self.thermal_camerainfo_callback
            )

        # NOTE: Annotations Subscriber (RGB/Thermal) - Tentative Code for MOT Evaluation
        self.rgb_annos_sub = Subscriber("/osr/annos_RGB", Annotations, self.rgb_annos_callback)
        self.thermal_annos_sub = Subscriber("/osr/annos_thermal", Annotations, self.thermal_annos_callback)

        # ROS Publisher
        self.tracks_pub = Publisher(
            opts.publish_mesg["tracks"], Tracks, queue_size=1
        )

        # NOTE: Evaluator Publisher (Tentative Code for MOT Evaluation)
        self.evaluator_pub = Publisher("/osr/eval", Evaluator, queue_size=1)

        # ROS SNU Result Publisher
        # NOTE: Tentative Code
        self.det_results_pub = {
            "color": Publisher(
                opts.publish_mesg["visualizations"]["det"]["color"],
                Image, queue_size=1
            ),
            "thermal": Publisher(
                opts.publish_mesg["visualizations"]["det"]["thermal"],
                Image, queue_size=1
            )
        }

        self.trk_acl_result_pub = {
            "color": Publisher(
                opts.publish_mesg["visualizations"]["trk_acl"]["color"],
                Image, queue_size=1
            ),
            "thermal": Publisher(
                opts.publish_mesg["visualizations"]["trk_acl"]["thermal"],
                Image, queue_size=1
            )
        }

        # self.det_result_pub = Publisher(
        #     opts.publish_mesg["det_result_rostopic_name"], Image, queue_size=1
        # )
        # self.trk_acl_result_pub = Publisher(
        #     opts.publish_mesg["trk_acl_result_rostopic_name"], Image, queue_size=1
        # )

    # Point Cloud Callback Function
    def point_cloud_callback(self, msg):
        self.lidar_msg = msg

    # Odometry Callback Function
    def odometry_callback(self, msg):
        self.odometry_msg = msg

    # NOTE: Annotations Callback Function (Tentative)
    def rgb_annos_callback(self, msg):
        self.annos_msg["color"] = msg

    def thermal_annos_callback(self, msg):
        self.annos_msg["thermal"] = msg

    # Color CameraInfo Callback Function
    def color_camerainfo_callback(self, msg):
        if self.color.get_sensor_params() is None:
            self.color.update_sensor_params_rostopic(msg=msg)

    # Disparity CameraInfo Callback Function
    def disparity_camerainfo_callback(self, msg):
        if self.disparity.get_sensor_params() is None:
            self.disparity.update_sensor_params_rostopic(msg=msg)

    # Infrared CameraInfo Callback Function
    def infrared_camerainfo_callback(self, msg):
        if self.infrared.get_sensor_params() is None:
            self.infrared.update_sensor_params_rostopic(msg=msg)

    # Thermal CameraInfo Callback Function
    def thermal_camerainfo_callback(self, msg):
        if self.thermal.get_sensor_params() is None:
            self.thermal.update_sensor_params_rostopic(msg=msg)

    # Publish Tracks
    def publish_tracks(self, trajectories, odometry_msg):
        out_tracks = wrap_tracks(trackers=trajectories, odometry=odometry_msg)
        self.tracks_pub.publish(out_tracks)
        return out_tracks

    # NOTE: Publish Evaluator - Tentative Code for MOT Evaluation
    def publish_evaluator(self, out_tracks, modal):
        if modal in self.annos_msg:
            # Get Annotations Msg
            annos = self.annos_msg[modal]

            # Initialize Evaluator
            ros_evaluator = Evaluator()
            ros_evaluator.header = annos.header

            # Get "Annotation" and "Track" Data and Input to Evaluator
            for anno in annos.annotations:
                ros_evaluator.annos.append(anno)

            for track in out_tracks.tracks:
                ros_evaluator.tracks.append(track)

            # Publish Evaluator
            self.evaluator_pub.publish(ros_evaluator)

        else:
            raise AssertionError()

    # Publish SNU Result Image ( DET / TRK + ACL )
    def publish_snu_result_image(self, result_frames_dict):
        for module, result_frames in result_frames_dict.items():
            if module == "det":
                if result_frames is not None:
                    for modal, result_frame in result_frames.items():
                        if self.opts.detector.is_result_publish is True:
                            if modal == "color":
                                pub_fmt = "rgb8"
                            elif modal == "thermal":
                                # pub_fmt = "mono8"
                                pub_fmt = "rgb8"
                            else:
                                raise NotImplementedError()
                            self.det_results_pub[modal].publish(
                                self.pub_bridge.cv2_to_imgmsg(result_frame, pub_fmt)
                            )
            elif module == "trk_acl":
                if result_frames is not None:
                    for modal, result_frame in result_frames.items():
                        if self.opts.tracker.is_result_publish is True:
                            if modal == "color":
                                pub_fmt = "rgb8"
                            elif modal == "thermal":
                                pub_fmt = "rgb8"
                            else:
                                raise NotImplementedError()
                            self.trk_acl_result_pub[modal].publish(
                                self.pub_bridge.cv2_to_imgmsg(result_frame, pub_fmt)
                            )
            else:
                raise NotImplementedError("Undefined Module: {}".format(module))

    # Update All Modal Data
    def update_all_modal_data(self, sync_data):
        sync_stamp = sync_data[0]
        sync_frame_dict = sync_data[1]

        # Update Modal Frames
        if self.color is not None:
            self.color.update_data(frame=sync_frame_dict["color"], stamp=sync_stamp)

        if self.disparity is not None:
            self.disparity.update_data(frame=sync_frame_dict["aligned_disparity"], stamp=sync_stamp)
            self.disparity.update_raw_data(raw_data=sync_frame_dict["disparity"])

        if self.thermal is not None:
            self.thermal.update_data(frame=sync_frame_dict["thermal"], stamp=sync_stamp)

        if self.infrared is not None:
            self.infrared.update_data(frame=sync_frame_dict["infrared"], stamp=sync_stamp)

        if self.nightvision is not None:
            self.nightvision.update_data(frame=sync_frame_dict["nightvision"], stamp=sync_stamp)

        if self.lidar is not None:
            self.lidar.update_data(
                lidar_pc_msg=self.lidar_msg, tf_transform=self.tf_transform
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

    def gather_all_modal_sensor_params(self):
        sensor_params_dict = {
            "color": self.color.get_sensor_params(),
            "disparity": self.disparity.get_sensor_params(),
            "thermal": self.thermal.get_sensor_params(),
            "infrared": self.infrared.get_sensor_params(),
            "nightvision": self.nightvision.get_sensor_params(),
            "lidar": self.lidar.get_sensor_params()
        }
        return sensor_params_dict


if __name__ == "__main__":
    pass
