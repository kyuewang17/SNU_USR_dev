#!/usr/bin/env python

import sys
import threading
import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2


class SyncSubscriber:
    def __init__(self, enable_color=True, enable_depth=True, enable_ir=True, enable_aligned_depth=True, enable_nv1=True, enable_thermal=True,
                 enable_color_camerainfo=True, enable_depth_camerainfo=True, enable_ir_camerainfo=True, enable_pointcloud=True, enable_odometry=True):
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_ir = enable_ir
        self.enable_aligned_depth = enable_aligned_depth
        self.enable_nv1 = enable_nv1
        self.enable_thermal = enable_thermal
        self.enable_color_camerainfo = enable_color_camerainfo
        self.enable_depth_camerainfo = enable_depth_camerainfo
        self.enable_ir_camerainfo = enable_ir_camerainfo
        self.enable_pointcloud = enable_pointcloud
        self.enable_odometry = enable_odometry

        self.bridge = CvBridge()

        self.sub_color = rospy.Subscriber("/osr/image_color", Image, self.callback_image_color) if self.enable_color else None
        self.sub_depth = rospy.Subscriber("/osr/image_depth", Image, self.callback_image_depth) if self.enable_depth else None
        self.sub_ir = rospy.Subscriber("/osr/image_ir", Image, self.callback_image_ir) if self.enable_ir else None
        self.sub_aligned_depth = rospy.Subscriber("/osr/image_aligned_depth", Image, self.callback_image_aligned_depth) if self.enable_aligned_depth else None
        self.sub_nv1 = rospy.Subscriber("/osr/image_nv1", Image, self.callback_image_nv1) if self.enable_nv1 else None
        self.sub_thermal = rospy.Subscriber("/osr/image_thermal", Image, self.callback_image_thermal) if self.enable_thermal else None
        self.sub_color_camerainfo = rospy.Subscriber("/osr/image_color_camerainfo", CameraInfo, self.callback_image_color_camerainfo) if self.enable_color_camerainfo else None
        self.sub_depth_camerainfo = rospy.Subscriber("/osr/image_depth_camerainfo", CameraInfo, self.callback_image_depth_camerainfo) if self.enable_depth_camerainfo else None
        self.sub_ir_camerainfo = rospy.Subscriber("/osr/image_ir_camerainfo", CameraInfo, self.callback_image_ir_camerainfo) if self.enable_ir_camerainfo else None
        self.sub_pointcloud = rospy.Subscriber("/osr/lidar_pointcloud", PointCloud2, self.callback_pointcloud) if self.enable_pointcloud else None
        self.sub_odometry = rospy.Subscriber("/osr/robot_odometry", Odometry, self.callback_odometry) if self.enable_odometry else None

        self.lock_flag = threading.Lock()
        self.lock_color = threading.Lock()
        self.lock_depth = threading.Lock()
        self.lock_ir = threading.Lock()
        self.lock_aligned_depth = threading.Lock()
        self.lock_nv1 = threading.Lock()
        self.lock_thermal = threading.Lock()
        self.lock_color_camerainfo = threading.Lock()
        self.lock_depth_camerainfo = threading.Lock()
        self.lock_ir_camerainfo = threading.Lock()
        self.lock_pointcloud = threading.Lock()
        self.lock_odometry = threading.Lock()

        self.dict_color = {}
        self.dict_depth = {}
        self.dict_ir = {}
        self.dict_aligned_depth = {}
        self.dict_nv1 = {}
        self.dict_thermal = {}
        self.dict_color_camerainfo = {}
        self.dict_depth_camerainfo = {}
        self.dict_ir_camerainfo = {}
        self.dict_pointcloud = {}
        self.dict_odometry = {}

        self.sync_flag = False
        self.sync_stamp = 0
        self.sync_color = None
        self.sync_depth = None
        self.sync_ir = None
        self.sync_aligned_depth = None
        self.sync_nv1 = None
        self.sync_thermal = None
        self.sync_color_camerainfo = None
        self.sync_depth_camerainfo = None
        self.sync_ir_camerainfo = None
        self.sync_pointcloud = None
        self.sync_odometry = None

    def __del__(self):
        del self.dict_color
        del self.dict_depth
        del self.dict_ir
        del self.dict_aligned_depth
        del self.dict_nv1
        del self.dict_thermal
        del self.dict_color_camerainfo
        del self.dict_depth_camerainfo
        del self.dict_ir_camerainfo
        del self.dict_pointcloud
        del self.dict_odometry

    def callback_image_color(self, data):
        try:
            cv_img_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
            if cv_img_color is not None:
                self.lock_color.acquire()
                self.dict_color[data.header.stamp] = cv_img_color
                self.lock_color.release()
        except CvBridgeError as e:
            print(e)

    def callback_image_depth(self, data):
        try:
            cv_img_depth = self.bridge.imgmsg_to_cv2(data, "mono16")
            if cv_img_depth is not None:
                self.lock_depth.acquire()
                self.dict_depth[data.header.stamp] = cv_img_depth
                self.lock_depth.release()
        except CvBridgeError as e:
            print(e)

    def callback_image_ir(self, data):
        try:
            cv_img_ir = self.bridge.imgmsg_to_cv2(data, "mono8")
            if cv_img_ir is not None:
                self.lock_ir.acquire()
                self.dict_ir[data.header.stamp] = cv_img_ir
                self.lock_ir.release()
        except CvBridgeError as e:
            print(e)

    def callback_image_aligned_depth(self, data):
        try:
            cv_img_aligned_depth = self.bridge.imgmsg_to_cv2(data, "mono16")
            if cv_img_aligned_depth is not None:
                self.lock_aligned_depth.acquire()
                self.dict_aligned_depth[data.header.stamp] = cv_img_aligned_depth
                self.lock_aligned_depth.release()
        except CvBridgeError as e:
            print(e)

    def callback_image_nv1(self, data):
        try:
            cv_img_nv1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            if cv_img_nv1 is not None:
                self.lock_nv1.acquire()
                self.dict_nv1[data.header.stamp] = cv_img_nv1
                self.lock_nv1.release()
        except CvBridgeError as e:
            print(e)

    def callback_image_thermal(self, data):
        try:
            cv_img_thermal = self.bridge.imgmsg_to_cv2(data, "mono16")
            if cv_img_thermal is not None:
                self.lock_thermal.acquire()
                self.dict_thermal[data.header.stamp] = cv_img_thermal
                self.lock_thermal.release()
        except CvBridgeError as e:
            print(e)

    def callback_image_color_camerainfo(self, data):
        self.lock_color_camerainfo.acquire()
        self.dict_color_camerainfo[data.header.stamp] = data
        self.lock_color_camerainfo.release()

    def callback_image_depth_camerainfo(self, data):
        self.lock_depth_camerainfo.acquire()
        self.dict_depth_camerainfo[data.header.stamp] = data
        self.lock_depth_camerainfo.release()

    def callback_image_ir_camerainfo(self, data):
        self.lock_ir_camerainfo.acquire()
        self.dict_ir_camerainfo[data.header.stamp] = data
        self.lock_ir_camerainfo.release()

    def callback_pointcloud(self, data):
        self.lock_pointcloud.acquire()
        self.dict_pointcloud[data.header.stamp] = data
        self.lock_pointcloud.release()

    def callback_odometry(self, data):
        self.lock_odometry.acquire()
        self.dict_odometry[data.header.stamp] = data
        self.lock_odometry.release()

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

        if self.enable_aligned_depth:
            self.lock_aligned_depth.acquire()
            keys_aligned_depth = self.dict_aligned_depth.keys()
            self.lock_aligned_depth.release()
        else:
            keys_aligned_depth = []

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

        if self.enable_color_camerainfo:
            self.lock_color_camerainfo.acquire()
            keys_color_camerainfo = self.dict_color_camerainfo.keys()
            self.lock_color_camerainfo.release()
        else:
            keys_color_camerainfo = []

        if self.enable_depth_camerainfo:
            self.lock_depth_camerainfo.acquire()
            keys_depth_camerainfo = self.dict_depth_camerainfo.keys()
            self.lock_depth_camerainfo.release()
        else:
            keys_depth_camerainfo = []

        if self.enable_ir_camerainfo:
            self.lock_ir_camerainfo.acquire()
            keys_ir_camerainfo = self.dict_ir_camerainfo.keys()
            self.lock_ir_camerainfo.release()
        else:
            keys_ir_camerainfo = []

        if self.enable_pointcloud:
            self.lock_pointcloud.acquire()
            keys_pointcloud = self.dict_pointcloud.keys()
            self.lock_pointcloud.release()
        else:
            keys_pointcloud = []

        if self.enable_odometry:
            self.lock_odometry.acquire()
            keys_odometry = self.dict_odometry.keys()
            self.lock_odometry.release()
        else:
            keys_odometry = []

        mergeset = list(set(keys_color) | set(keys_depth) | set(keys_ir) | set(keys_aligned_depth) | set(keys_nv1) | set(keys_thermal) |
                        set(keys_color_camerainfo) | set(keys_depth_camerainfo) | set(keys_ir_camerainfo) | set(keys_pointcloud) | set(keys_odometry))
        keys_color = mergeset if not self.enable_color else keys_color
        keys_depth = mergeset if not self.enable_depth else keys_depth
        keys_ir = mergeset if not self.enable_ir else keys_ir
        keys_aligned_depth = mergeset if not self.enable_aligned_depth else keys_aligned_depth
        keys_nv1 = mergeset if not self.enable_nv1 else keys_nv1
        keys_thermal = mergeset if not self.enable_thermal else keys_thermal
        keys_color_camerainfo = mergeset if not self.enable_color_camerainfo else keys_color_camerainfo
        keys_depth_camerainfo = mergeset if not self.enable_depth_camerainfo else keys_depth_camerainfo
        keys_ir_camerainfo = mergeset if not self.enable_ir_camerainfo else keys_ir_camerainfo
        keys_pointcloud = mergeset if not self.enable_pointcloud else keys_pointcloud
        keys_odometry = mergeset if not self.enable_odometry else keys_odometry

        common_keys = list(set(keys_color) & set(keys_depth) & set(keys_ir) & set(keys_aligned_depth) & set(keys_nv1) & set(keys_thermal) &
                           set(keys_color_camerainfo) & set(keys_depth_camerainfo) & set(keys_ir_camerainfo) & set(keys_pointcloud) & set(keys_odometry))
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

            if self.enable_aligned_depth:
                self.lock_aligned_depth.acquire()
                self.sync_aligned_depth = self.dict_aligned_depth[key_value]
                self.lock_aligned_depth.release()
            else:
                self.sync_aligned_depth = None

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

            if self.enable_color_camerainfo:
                self.lock_color_camerainfo.acquire()
                self.sync_color_camerainfo = self.dict_color_camerainfo[key_value]
                self.lock_color_camerainfo.release()
            else:
                self.sync_color_camerainfo = None

            if self.enable_depth_camerainfo:
                self.lock_depth_camerainfo.acquire()
                self.sync_depth_camerainfo = self.dict_depth_camerainfo[key_value]
                self.lock_depth_camerainfo.release()
            else:
                self.sync_depth_camerainfo = None

            if self.enable_ir_camerainfo:
                self.lock_ir_camerainfo.acquire()
                self.sync_ir_camerainfo = self.dict_ir_camerainfo[key_value]
                self.lock_ir_camerainfo.release()
            else:
                self.sync_ir_camerainfo = None

            if self.enable_pointcloud:
                self.lock_pointcloud.acquire()
                self.sync_pointcloud = self.dict_pointcloud[key_value]
                self.lock_pointcloud.release()
            else:
                self.sync_pointcloud = None

            if self.enable_odometry:
                self.lock_odometry.acquire()
                self.sync_odometry = self.dict_odometry[key_value]
                self.lock_odometry.release()
            else:
                self.sync_odometry = None

            self.lock_flag.acquire()
            self.sync_stamp = key_value
            self.sync_flag = True
            self.lock_flag.release()

            [self.dict_color.pop(v) for v in keys_color] if self.enable_color else None
            [self.dict_depth.pop(v) for v in keys_depth] if self.enable_depth else None
            [self.dict_ir.pop(v) for v in keys_ir] if self.enable_ir else None
            [self.dict_aligned_depth.pop(v) for v in keys_aligned_depth] if self.enable_aligned_depth else None
            [self.dict_nv1.pop(v) for v in keys_nv1] if self.enable_nv1 else None
            [self.dict_thermal.pop(v) for v in keys_thermal] if self.enable_thermal else None
            [self.dict_color_camerainfo.pop(v) for v in keys_color_camerainfo] if self.enable_color_camerainfo else None
            [self.dict_depth_camerainfo.pop(v) for v in keys_depth_camerainfo] if self.enable_depth_camerainfo else None
            [self.dict_ir_camerainfo.pop(v) for v in keys_ir_camerainfo] if self.enable_ir_camerainfo else None
            [self.dict_pointcloud.pop(v) for v in keys_pointcloud] if self.enable_pointcloud else None
            [self.dict_odometry.pop(v) for v in keys_odometry] if self.enable_odometry else None
        else:
            self.lock_flag.acquire()
            self.sync_stamp = -1
            self.sync_flag = False
            self.lock_flag.release()

    def get_sync_data(self):
        self.lock_flag.acquire()
        result = (self.sync_flag, self.sync_stamp, self.sync_color, self.sync_depth, self.sync_ir, self.sync_aligned_depth,
                  self.sync_nv1, self.sync_thermal, self.sync_color_camerainfo, self.sync_depth_camerainfo, self.sync_ir_camerainfo,
                  self.sync_pointcloud, self.sync_odometry)
        self.lock_flag.release()
        return result


##########################################################################################
def main(args):
    rospy.init_node('sub_test_node', anonymous=True)

    params = {'enable_color': True, 'enable_depth': True, 'enable_ir': True, 'enable_aligned_depth': True,
              'enable_nv1': False, 'enable_thermal': True, 'enable_color_camerainfo': False,
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
