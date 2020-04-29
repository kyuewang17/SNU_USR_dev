cmake_minimum_required(VERSION 2.8.3)
project(osr_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
#add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
    std_msgs
    geometry_msgs
    nav_msgs
    message_generation
    
)

## Generate messages in the 'msg' folder
add_message_files(
   FILES
   Track.msg
   Tracks.msg
   BoundingBox.msg
)


## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES std_msgs nav_msgs geometry_msgs
)

catkin_package(
    #INCLUDE_DIRS include
    #LIBRARIES
    CATKIN_DEPENDS geometry_msgs nav_msgs message_runtime std_msgs
    #DEPENDS
)
