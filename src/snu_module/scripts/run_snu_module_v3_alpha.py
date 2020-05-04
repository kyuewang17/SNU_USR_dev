#!/usr/bin/env python
"""
SNU Integrated Module v3.0-alpha version
    - Main Execution Code (Run Code)

    [Updates on v2.0] (~190908)
    - Code Structure modified
    - Subscribe multi-modal sensor input in a <struct>-like data structure

    [Updates on v2.05] (~190920)
    - Improved <struct>-like multi-modal sensor data structure

    [Updates on v2.5] (~191018)_(iitp interim version)
    - <class>-like multi-modal sensor data structure
    - separate "static" and "dynamic" agent modes
    - minor improvements on execution modes (dictionary-based)

    [Updates expected on v3.0]
    - remove global variables for multi-modal data
        : class-based rospy Subscribe/Publish
    - make each module executable
        : now, people can directly push their modification to the GitHub
        --> This is done by using python "main" namespace
    - [VERY IMPORTANT ENVIRONMENT MODIFICATION]
        : upgrade pyTorch and torchVision
            - pyTorch: 1.0.0 ==>> 1.1.0 (??)
            - torchVision: 0.2.0 ==>> 0.3.0
    - TBA

    << Project Information >>
    - "Development of multimodal sensor-based intelligent systems for outdoor surveillance robots"
    - Project Total Period : 2017-04-01 ~ 2021-12-31
    - Current Project Period : The 3rd year (2019-01-01 ~ 2019-12-31)

    << Institutions & Researchers >>
    - Perception and Intelligence Laboratory (PIL)
        [1] Kyuewang Lee (kyuewang@snu.ac.kr)
        [2] Daeho Um (umdaeho1@gmail.com)
    - Machine Intelligence and Pattern Recognition Laboratory (MIPAL)
        [1] Jae-Young Yoo (yoojy31@snu.ac.kr)
        [2] Hojun Lee (hojun815@snu.ac.kr)
        [3] Inseop Chung (jis3613@snu.ac.kr)

    << Code Environments >>
        [ Key Environments ]
        - python == 2.7
        - torch == 1.0.0
            - torchvision == 0.2.0
        - CUDA == 10.0 (greater than 9.0)
            - cuDNN == 7.5.0
        - ROS-kinetics
            - (need rospkg inside the python virtualenv)
        - opencv-python
        - empy
            - (need this to prevent "-j4 -l4" error)

        [ Other Dependencies ] - (will be updated continuously)
        - numpy, numba, scipy, opencv-python, FilterPy, yaml, rospkg, sklearn

    << Memo >>
        Watch out for OpenCV imshow!!!
            -> RGB is shown in "BGR" format
        ** Notes about RGB Image Input
            -> rosbag file from pohang agents : BGR format
            -> rostopic from D435i Camera : RGB format

"""

# Import Modules
import cv2
import socket
import datetime

# Import SNU Algorithm Modules
import module_detection as snu_det
import module_tracking as snu_mmt
import module_action as snu_acl

########### Switches ###########
switches = {
    # True if ROS mode, False if image sequence mode
    # (deprecated...since valid multimodal DB is acquired from bag file or ROS topics)
    "ROS": True,

    # True if ssh mode, False if visualization mode
    # (ssh mode: turn off all visualization regardless of options)
    "SSH": False,

    # TBA

}
################################

###### Agent Type and Name ######
agent_type = "dynamic"
agent_name = socket.gethostname()
#################################




























