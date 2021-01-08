"""
Object Recognition Module (SNU Integrated Module)
for Outdoor Surveillance Robots

    - TBA

"""
from rospy import Subscriber, Publisher
from cv_bridge import CvBridge

from wrapper import wrap_tracks
from sensors import ros_sensor_image, ros_sensor_disparity, ros_sensor_lidar

# Import ROS Message Types
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from osr_msgs.msg import Tracks
from nav_msgs.msg import Odometry


class RECOGNITION_FRAMEWORK(object):
    def __init__(self, opts):
        # Options
        self.opts = opts

        # Initialize Point Cloud ROS Message Variable
        self.lidar_msg = None

        # Odometry Message (Pass-through Variable)
        self.odometry_msg = None

        # TF Static-related Variables
        self.tf_transform = None




























if __name__ == "__main__":
    pass
