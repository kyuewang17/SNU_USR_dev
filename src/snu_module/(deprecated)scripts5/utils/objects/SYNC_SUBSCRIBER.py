#!/usr/bin/env python

import sys
import threading
import roslib
import rospy
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2


class SyncSubscriber(object):
    def __init__(self, sync_switch_dict, rostopic_info_dict):
        self.rostopic_info_dict = rostopic_info_dict

        # Unpack Synchronization Switch Dictionary
        self.enable_color = sync_switch_dict["color"]
        self.enable_depth = sync_switch_dict["depth"]
        self.enable_ir = sync_switch_dict["infrared"]
        self.enable_nv1 = sync_switch_dict["nightvision"]
        self.enable_thermal = sync_switch_dict["thermal"]

        self.bridge = CvBridge()

        self.lock_flag = threading.Lock()
        self.lock_color = threading.Lock()
        self.lock_depth = threading.Lock()
        self.lock_ir = threading.Lock()
        self.lock_nv1 = threading.Lock()
        self.lock_thermal = threading.Lock()

        self.dict_color = {}
        self.dict_depth = {}
        self.dict_ir = {}
        self.dict_nv1 = {}
        self.dict_thermal = {}

        self.sub_color = rospy.Subscriber(rostopic_info_dict["color"]["rostopic_name"], Image, self.callback_image_color) if self.enable_color else None
        self.sub_depth = rospy.Subscriber(rostopic_info_dict["depth"]["rostopic_name"], Image, self.callback_image_depth) if self.enable_depth else None
        self.sub_ir = rospy.Subscriber(rostopic_info_dict["infrared"]["rostopic_name"], Image, self.callback_image_ir) if self.enable_ir else None
        self.sub_nv1 = rospy.Subscriber(rostopic_info_dict["nightvision"]["rostopic_name"], Image, self.callback_image_nv1) if self.enable_nv1 else None
        self.sub_thermal = rospy.Subscriber(rostopic_info_dict["thermal"]["rostopic_name"], Image, self.callback_image_thermal) if self.enable_thermal else None

        # self.sub_color = rospy.Subscriber(opts.sensors.color["rostopic_name"], Image, self.callback_image_color) if self.enable_color else None
        # self.sub_depth = rospy.Subscriber("/osr/image_depth", Image, self.callback_image_depth) if self.enable_depth else None
        # self.sub_ir = rospy.Subscriber(opts.sensors.infrared["rostopic_name"], Image, self.callback_image_ir) if self.enable_ir else None
        # self.sub_nv1 = rospy.Subscriber(opts.sensors.nightvision["rostopic_name"], Image, self.callback_image_nv1) if self.enable_nv1 else None
        # self.sub_thermal = rospy.Subscriber(opts.sensors.thermal["rostopic_name"], Image, self.callback_image_thermal) if self.enable_thermal else None

        # self.sub_color = rospy.Subscriber("/osr/image_color", Image, self.callback_image_color) if self.enable_color else None
        # self.sub_depth = rospy.Subscriber("/osr/image_depth", Image, self.callback_image_depth) if self.enable_depth else None
        # self.sub_ir = rospy.Subscriber("/osr/image_ir", Image, self.callback_image_ir) if self.enable_ir else None
        # self.sub_aligned_depth = rospy.Subscriber("/osr/image_aligned_depth", Image, self.callback_image_aligned_depth) if self.enable_aligned_depth else None
        # self.sub_nv1 = rospy.Subscriber("/osr/image_nv1", Image, self.callback_image_nv1) if self.enable_nv1 else None
        # self.sub_thermal = rospy.Subscriber("/osr/image_thermal", Image, self.callback_image_thermal) if self.enable_thermal else None
        # self.sub_color_camerainfo = rospy.Subscriber("/osr/image_color_camerainfo", CameraInfo, self.callback_image_color_camerainfo) if self.enable_color_camerainfo else None
        # self.sub_depth_camerainfo = rospy.Subscriber("/osr/image_depth_camerainfo", CameraInfo, self.callback_image_depth_camerainfo) if self.enable_depth_camerainfo else None
        # self.sub_ir_camerainfo = rospy.Subscriber("/osr/image_ir_camerainfo", CameraInfo, self.callback_image_ir_camerainfo) if self.enable_ir_camerainfo else None
        # self.sub_pointcloud = rospy.Subscriber("/osr/lidar_pointcloud", PointCloud2, self.callback_pointcloud) if self.enable_pointcloud else None
        # self.sub_odometry = rospy.Subscriber("/robot_odom", Odometry, self.callback_odometry) if self.enable_odometry else None

        self.sync_flag = False
        self.sync_stamp = 0
        
        self.sync_color = None
        self.sync_depth = None
        self.sync_ir = None
        self.sync_nv1 = None
        self.sync_thermal = None

    def __del__(self):
        del self.dict_color
        del self.dict_depth
        del self.dict_ir
        del self.dict_nv1
        del self.dict_thermal

    def callback_image_color(self, data):
        try:
            if data.encoding.__contains__("bgr") is True:
                cv_img_color = self.bridge.imgmsg_to_cv2(data, self.rostopic_info_dict["color"]["imgmsg_to_cv2_encoding"])
                cv_img_color = cv2.cvtColor(cv_img_color, cv2.COLOR_BGR2RGB)
            elif data.encoding.__contains__("rgb") is True:
                cv_img_color = self.bridge.imgmsg_to_cv2(data, self.rostopic_info_dict["color"]["imgmsg_to_cv2_encoding"])
            elif data.encoding.__contains__("mono") is True:
                cv_img_color = self.bridge.imgmsg_to_cv2(data, "8UC1")
            else:
                assert 0, "Undefined Data Type for Color Modal"

            if cv_img_color is not None:
                self.lock_color.acquire()
                self.dict_color[data.header.stamp] = cv_img_color
                self.lock_color.release()
        except CvBridgeError as e:
            print(e)

    def callback_image_depth(self, data):
        try:
            cv_img_depth = self.bridge.imgmsg_to_cv2(data, self.rostopic_info_dict["depth"]["imgmsg_to_cv2_encoding"])
            if cv_img_depth is not None:
                self.lock_depth.acquire()
                self.dict_depth[data.header.stamp] = cv_img_depth
                self.lock_depth.release()
        except CvBridgeError as e:
            print(e)

    def callback_image_ir(self, data):
        try:
            cv_img_ir = self.bridge.imgmsg_to_cv2(data, self.rostopic_info_dict["infrared"]["imgmsg_to_cv2_encoding"])
            if cv_img_ir is not None:
                self.lock_ir.acquire()
                self.dict_ir[data.header.stamp] = cv_img_ir
                self.lock_ir.release()
        except CvBridgeError as e:
            print(e)

    def callback_image_nv1(self, data):
        try:
            cv_img_nv1 = self.bridge.imgmsg_to_cv2(data, self.rostopic_info_dict["nightvision"]["imgmsg_to_cv2_encoding"])
            if cv_img_nv1 is not None:
                self.lock_nv1.acquire()
                self.dict_nv1[data.header.stamp] = cv_img_nv1
                self.lock_nv1.release()
        except CvBridgeError as e:
            print(e)

    def callback_image_thermal(self, data):
        try:
            cv_img_thermal = self.bridge.imgmsg_to_cv2(data, self.rostopic_info_dict["thermal"]["imgmsg_to_cv2_encoding"])
            if cv_img_thermal is not None:
                self.lock_thermal.acquire()
                self.dict_thermal[data.header.stamp] = cv_img_thermal
                self.lock_thermal.release()
        except CvBridgeError as e:
            print(e)

    def make_sync_data(self):
        if self.enable_color:
            self.lock_color.acquire()
            keys_color = self.dict_color.keys()
            self.lock_color.release()
        else:
            keys_color = []

        if self.enable_depth:
            self.lock_depth.acquire()
            keys_depth = self.dict_depth.keys()
            self.lock_depth.release()
        else:
            keys_depth = []

        if self.enable_ir:
            self.lock_ir.acquire()
            keys_ir = self.dict_ir.keys()
            self.lock_ir.release()
        else:
            keys_ir = []

        if self.enable_nv1:
            self.lock_nv1.acquire()
            keys_nv1 = self.dict_nv1.keys()
            self.lock_nv1.release()
        else:
            keys_nv1 = []

        if self.enable_thermal:
            self.lock_thermal.acquire()
            keys_thermal = self.dict_thermal.keys()
            self.lock_thermal.release()
        else:
            keys_thermal = []

        mergeset = list(set(keys_color) | set(keys_depth) | set(keys_ir) | set(keys_nv1) | set(keys_thermal))
        keys_color = mergeset if not self.enable_color else keys_color
        keys_depth = mergeset if not self.enable_depth else keys_depth
        keys_ir = mergeset if not self.enable_ir else keys_ir
        keys_nv1 = mergeset if not self.enable_nv1 else keys_nv1
        keys_thermal = mergeset if not self.enable_thermal else keys_thermal

        common_keys = list(set(keys_color) & set(keys_depth) & set(keys_ir) & set(keys_nv1) & set(keys_thermal))
        if common_keys is not None and len(common_keys) > 0:
            common_keys.sort()
            key_value = common_keys[-1]

            if self.enable_color:
                self.lock_color.acquire()
                self.sync_color = self.dict_color[key_value]
                self.lock_color.release()
            else:
                self.sync_color = None

            if self.enable_depth:
                self.lock_depth.acquire()
                self.sync_depth = self.dict_depth[key_value]
                self.lock_depth.release()
            else:
                self.sync_depth = None

            if self.enable_ir:
                self.lock_ir.acquire()
                self.sync_ir = self.dict_ir[key_value]
                self.lock_ir.release()
            else:
                self.sync_ir = None

            if self.enable_nv1:
                self.lock_nv1.acquire()
                self.sync_nv1 = self.dict_nv1[key_value]
                self.lock_nv1.release()
            else:
                self.sync_nv1 = None

            if self.enable_thermal:
                self.lock_thermal.acquire()
                self.sync_thermal = self.dict_thermal[key_value]
                self.lock_thermal.release()
            else:
                self.sync_thermal = None

            self.lock_flag.acquire()
            self.sync_stamp = key_value
            self.sync_flag = True
            self.lock_flag.release()

            [self.dict_color.pop(v) for v in keys_color] if self.enable_color else None
            [self.dict_depth.pop(v) for v in keys_depth] if self.enable_depth else None
            [self.dict_ir.pop(v) for v in keys_ir] if self.enable_ir else None
            [self.dict_nv1.pop(v) for v in keys_nv1] if self.enable_nv1 else None
            [self.dict_thermal.pop(v) for v in keys_thermal] if self.enable_thermal else None
        else:
            self.lock_flag.acquire()
            self.sync_stamp = -1
            self.sync_flag = False
            self.lock_flag.release()

    def get_sync_data(self):
        self.lock_flag.acquire()
        if self.sync_flag is False:
            self.lock_flag.release()
            return None
        else:
            result_sync_frame_dict = {
                "color": self.sync_color, "depth": self.sync_depth,
                "thermal": self.sync_thermal, "infrared": self.sync_ir, "nightvision": self.sync_nv1
            }
            self.lock_flag.release()
            return self.sync_stamp, result_sync_frame_dict

    # def get_sync_data(self):
    #     self.lock_flag.acquire()
    #     result = (self.sync_flag, self.sync_stamp, self.sync_color, self.sync_depth, self.sync_ir, self.sync_aligned_depth,
    #               self.sync_nv1, self.sync_thermal)
    #     self.lock_flag.release()
    #     return result


##########################################################################################
def main(args):
    rospy.init_node('sub_test_node', anonymous=True)

    params = {'enable_color': False, 'enable_depth': False, 'enable_ir': False, 'enable_aligned_depth': False,
              'enable_nv1': False, 'enable_thermal': False, 'enable_color_camerainfo': True,
              'enable_depth_camerainfo': False, 'enable_ir_camerainfo': False, 'enable_pointcloud': False, 'enable_odometry': False}
    ss = SyncSubscriber(**params)

    try:
        while not rospy.is_shutdown():
            ss.make_sync_data()
            sync_data = ss.get_sync_data()
            print("get sync data... {} - {}".format(sync_data[0], sync_data[1]))

            if sync_data[0] is True:
                if sync_data[2] is not None:
                    img_color = cv2.resize(sync_data[2], dsize=(320, 240), interpolation=cv2.INTER_LINEAR)
                else:
                    img_color = np.zeros((240, 320, 3)).astype('uint8')

                if sync_data[3] is not None:
                    img_depth = (sync_data[3] / 256).astype('uint8')
                    img_depth = cv2.resize(img_depth, dsize=(320, 240), interpolation=cv2.INTER_LINEAR)
                    img_depth = cv2.applyColorMap(img_depth, cv2.COLORMAP_JET)
                else:
                    img_depth = np.zeros((240, 320, 3)).astype('uint8')

                if sync_data[4] is not None:
                    img_ir = cv2.resize(sync_data[4], dsize=(320, 240), interpolation=cv2.INTER_LINEAR)
                    img_ir = cv2.cvtColor(img_ir, cv2.COLOR_GRAY2RGB)
                else:
                    img_ir = np.zeros((240, 320, 3)).astype('uint8')

                if sync_data[5] is not None:
                    img_aligned_depth = (sync_data[5] / 256).astype('uint8')
                    img_aligned_depth = cv2.resize(img_aligned_depth, dsize=(320, 240), interpolation=cv2.INTER_LINEAR)
                    img_aligned_depth = cv2.applyColorMap(img_aligned_depth, cv2.COLORMAP_JET)
                else:
                    img_aligned_depth = np.zeros((240, 320, 3)).astype('uint8')

                if sync_data[7] is not None:
                    img_thermal = np.zeros((480, 640)).astype('uint8')
                    cv2.normalize(sync_data[7], img_thermal, 0.0, 255.0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                    img_thermal = cv2.resize(img_thermal, dsize=(320, 240), interpolation=cv2.INTER_LINEAR)
                    img_thermal = cv2.cvtColor(img_thermal, cv2.COLOR_GRAY2RGB)
                else:
                    img_thermal = np.zeros((240, 320, 3)).astype('uint8')

                img_sync_data = np.hstack([img_color, img_ir, img_depth, img_aligned_depth, img_thermal])
                cv2.imshow("sync_data", img_sync_data)
                cv2.waitKey(1)

            rospy.sleep(0.1)
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down...")

    cv2.destroyAllWindows()
    del ss


if __name__ == '__main__':
    main(sys.argv)
