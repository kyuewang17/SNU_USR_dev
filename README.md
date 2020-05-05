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
        - [Kyuewang Lee]()
        - [Daeho Um]()
        
- **Machine Intelligence and Pattern Recognition Laboratory (MIPAL)**
    - Professor
        - [Nojun Kwak](http://mipal.snu.ac.kr/index.php/Nojun_Kwak)
    - Ph.D. Candidate
        - [Jae-Young Yoo]()
        - [Jee-soo Kim]()
        - [Hojun Lee]()
        - [Inseop Chung]()

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
    - ROS-kinetic (install at base)
        - need "rospkg" module, install via *pip*
        - for "pycharm" IDE, refer to [**THIS**](https://stackoverflow.com/questions/24197970/pycharm-import-external-library/24206781#24206781)
            - import: **/opt/ros/\<distro\>/lib/python2.7/dist-packages**\
              also refer to [THIS](https://developpaper.com/ros-python-libraries-such-as-import-rospy-are-not-available-in-sublime-text-3-and-pycharm/)
    - opencv-python (install via *pip*)
    - empy (*pip*)
    - yaml
    - numpy, numba, scipy, FilterPy, sklearn, yacs
    
- Build Detection Module (tentative, for current detector model: **RefineDet** \[[Paper](https://arxiv.org/abs/1711.06897)\])
    - TBA

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
              
    2. Run SNU USR Integrated Algorithm\
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
            - [2] Publish all the "_rostopics-of-interest_"
            - [3] Play the bag file
         3. Run the execution file
            - For command-line,\
            `>> rosrun snu_module path/to/scripts4/run_snu_module.py`
            - For PyCharm IDE
                - _run as same as ordinary pycharm projects_\
                (debugging is also possible)


















