## SNU Unmanned Surveillance Robot (USR) Project

---
#### Project Information
- Project Name

    - Development of multimodal sensor-based intelligent systems for outdoor surveillance robots
    
    - Project Total Period : 2017-04-01 ~ 2021-12-31
    
    - Institutions
        - LG Electronics
        - ETRI
        - KIRO
        - SNU

---
#### Seoul National University (SNU) Researchers
- **Perception and Intelligence Laboratory (PIL)**
    - Professor
        - [Jin Young Choi](http://pil.snu.ac.kr/about/view.do?idx=1)
    - Ph.D. Candidate
        - [Kyuewang Lee](http://pil.snu.ac.kr/member/view.do?idx=15)
        - [Daeho Um](http://pil.snu.ac.kr/member/view.do?idx=66)
        
- **Machine Intelligence and Pattern Recognition Laboratory (MIPAL)**
    - Professor
        - [Nojun Kwak](http://mipal.snu.ac.kr/index.php/Nojun_Kwak)
    - Ph.D. Candidate
        - [Jae-Young Yoo](http://mipal.snu.ac.kr/index.php/Jae-Young_Yoo)
        - [Jee-soo Kim](http://mipal.snu.ac.kr/index.php/Jee-soo_Kim)
        - [Hojun Lee](http://mipal.snu.ac.kr/index.php/Hojun_Lee)
        - [Inseop Chung](http://mipal.snu.ac.kr/index.php/Inseop_Chung)

---
#### Code Instructions
- Development System Information
    - Developed on **Ubuntu 16.04**
    - GPU: **_GeForce GTX 1070_** (also tested on **_GTX 1080Ti_**)

- Dependencies (use **Anaconda Environment**)
    - python 2.7
    - PyTorch 1.1.0
        - torchvision 0.3.0
    - CUDA 10.0
        - cuDNN 7.5.0
    - ROS-kinetic (**Install on Ubuntu Main System**)
        - need "rospkg" module, install via *pip*\
        (**rospkg module is needed in the Anaconda Environment**, don't install it via pip on the system)
        - For **LiDAR** support, do install additional module "_ros\_numpy_" as follows:\
        `>> sudo apt install ros-kinetic-ros-numpy`\
        (**Currently, NOT USING!!**)
        - for "pycharm" IDE, refer to [**THIS**](https://stackoverflow.com/questions/24197970/pycharm-import-external-library/24206781#24206781)
            - import-(1): **/opt/ros/\<distro\>/lib/python2.7/dist-packages**\
              also refer to [THIS](https://developpaper.com/ros-python-libraries-such-as-import-rospy-are-not-available-in-sublime-text-3-and-pycharm/)
            - import-(2): **/devel/lib/python2.7/dist-packages**\
              **\[Note\]** : "**catkin\_make**" is necessary\
              (check-out for step-02 in "_How to use ROS-embedded current algorithm?_" to build Custom ROS Messages)
    - opencv-python (install via *pip*)
    - empy (*pip*)
    - yaml
    - numpy, numba, scipy, FilterPy, sklearn, yacs
    - sip 4.18 (for PyKDL support, version number is important!)
    - motmetrics (*pip*, for MOT Performance Evaluation)
    
   
- Build Detection Module (tentative, for current detector model: **yolo v4** \[[Paper](https://arxiv.org/abs/2004.10934)\])
    - Setting Requirements (_as far as we know_...)
        - CMake >= 3.12: https://cmake.org/download/ ( cf. https://snowdeer.github.io/linux/2018/04/10/upgrade-cmake/ (korean) )
        - CUDA 10
        - OpenCV >= 2.4
        - cudnn >= 7.0 for CUDA 10.0
        - requirement detail : https://github.com/AlexeyAB/darknet
    - Download yolov4.weights file
        - https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT
        - path : src/snu_module/models/detector/YOLOv4/day/
    - Build Steps
        1. Run [build.sh](/src/snu_module/scripts4/detection_lib/darknet/build.sh) \
        `>> name@name:~/src/snu_module/scripts4/detection_lib/darknet$ ./build.sh`
        

- How to use ROS-embedded current algorithm?
    1. Install [ROS-kinetic](http://wiki.ros.org/kinetic/Installation) and set basic things
        - Recommended to Install the "**ros-kinetic-desktop-full**" repository
            - `>> sudo apt install ros-kinetic-desktop-full`
        - For convenience, add source "**/opt/ros/kinetic/setup.bash**" to the "**bashrc**" file
            - `>> echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc`\
              `>> source ~/.bashrc`
        - Install other dependencies
            - `>> sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential`
        - Initialize rosdep
            - `>> sudo apt install python-rosdep`\
              `>> sudo rosdep init`\
              `>> rosdep update`
              
    2. Build Custom ROS Messages
        - [osr_msgs](/src/osr/osr_msgs)
            1. [BoundingBox.msg](/src/osr/osr_msgs/msg/BoundingBox.msg)
            2. [Track.msg](/src/osr/osr_msgs/msg/Track.msg)
            3. [Tracks.msg](/src/osr/osr_msgs/msg/Tracks.msg)
        
        - How to build **osr_msgs**
            1. At the master directory, (_**i.e.**_ /path/to/SNU\_USR\_dev) run the following:\
            `>> catkin_make`
            2. If successful, then additionally import the following path to the python interpreter:\
            **/devel/lib/python2.7/dist-packages**
            3. \[**Important**\] Before running the code, run the following at the terminal:\
            `>> source /path/to/SNU_USR_dev/devel/setup.bash`
              
    3. Run SNU USR Integrated Algorithm\
         (The main execution file is  [**_run_snu_module.py_**](src/snu_module/scripts4/run_snu_module.py))
         1. `>> roscore`
         2. Publish _rostopics_ to the ROS Core
            - [1] Unpack **_\*.bag_** file
                1. Use "_rosbag_" command (CUI-based) \[[options](http://wiki.ros.org/rosbag/Commandline)\]
                    - `>> rosbag play <file_name>.bag`
                2. Use "_rqt_bag_" command (GUI-based)
                    - `>> rqt_bag <file_name>.bag`
                3. Use "_rqt\_image\_view_" command (for image-like rostopics only)
                    - `>> rqt_image_view <file_name>.bag`
                4. Use "_rviz_" command
                    - TBA
            - [2] Publish all the "_rostopics-of-interest_"\
            (_i.e._) "**/osr/image_color**", "**/osr/lidar_pointcloud**", ...
            - [3] Play the bag file
         3. Run the execution file
            - For command-line,\
            `>> rosrun snu_module path/to/scripts4/run_snu_module.py`
            - For PyCharm IDE
                - _run as same as ordinary pycharm projects_\
                (debugging is also possible)

    3. Script Information
        > [ros_snu_module.py](src/snu_module/scripts4/run_snu_module.py)
        >    - the main execution code
        
        > [options.py](src/snu_module/scripts4/options.py)
        >    - option file for SNU USR integrated algorithm

        - TBA

---
#### Code Know-hows and Trouble-shootings
1. rospkg.common.Resourcenotfound (**tf2_ros**) \[on **PyCharm** IDE\]
    - Run PyCharm at the terminal with Anaconda environment activated
    
2. TBA











