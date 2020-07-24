#!/usr/bin/env bash

TARGET_ROSBAG_BASE_PATH="/home/kyle/bag_files"
t_now=$(date +"%Y-%m-%d")

# Change to Target ROSBAG Base Directory for Saving
if [ ! -d ${TARGET_ROSBAG_BASE_PATH} ]; then
  echo "Path ${TARGET_ROSBAG_BASE_PATH} Does Not Exist!"
  exit 101
fi

# Make Current YYYY-MM-DD Directory if not exist
TARGET_ROSBAG_PATH=${TARGET_ROSBAG_BASE_PATH}/${t_now}
#echo "${TARGET_ROSBAG_PATH}"
if [ ! -e "${TARGET_ROSBAG_PATH}" ]; then
  mkdir "${TARGET_ROSBAG_PATH}"
else
  echo "Path ${TARGET_ROSBAG_PATH} Already Exists...!"
fi

# Save ROSBAG File to the Target ROSBAG directory
rosbag record -o "${TARGET_ROSBAG_PATH}"/ \
"/osr/image_color" \
"/osr/image_color_camerainfo" \
"/osr/image_aligned_depth" \
"/osr/image_depth_camerainfo" \
"/osr/image_thermal" \
"/osr/image_thermal_camerainfo" \
"/osr/image_ir" \
"/osr/image_ir_camerainfo" \
"/osr/image_nv1" \
"/osr/lidar_pointcloud" \
"/robot_odom" \
"/tf" \
"/tf_static" \
"/osr/snu_det_result_image" \
"/osr/snu_trk_acl_result_image"
