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
    - ROS-kinetics (install at base)
        - need "rospkg" module, install via *pip*
        - for "pycharm" IDE, refer to [**THIS**](https://stackoverflow.com/questions/24197970/pycharm-import-external-library/24206781#24206781)
            - import: **/opt/ros/\<distro\>/lib/python2.7/dist-packages**\
              also refer to [THIS](https://developpaper.com/ros-python-libraries-such-as-import-rospy-are-not-available-in-sublime-text-3-and-pycharm/) 
    - opencv-python (install via *pip*)
    - empy (*pip*)
    - yaml
    - numpy, numba, scipy, FilterPy, sklearn, yacs






















