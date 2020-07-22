#!/usr/bin/env bash

TARGET_ROSBAG_BASE_PATH="/home/kyle/bag_files/"
t_now=$(date +"%Y-%m-%d")

# Change to Target ROSBAG Base Directory for Saving
if [ -d ${TARGET_ROSBAG_BASE_PATH} ]; then
  cd ${TARGET_ROSBAG_BASE_PATH}
fi || {
    echo "Path ${TARGET_ROSBAG_BASE_PATH} Does Not Exist!"
  }

# Make Current YYYY-MM-DD Directory if not exist
TARGET_ROSBAG_PATH=${TARGET_ROSBAG_BASE_PATH}${t_now}
echo "${TARGET_ROSBAG_PATH}"
if [ ! -e "${TARGET_ROSBAG_PATH}" ]; then
  mkdir "${TARGET_ROSBAG_PATH}"
else
  echo "Path ${TARGET_ROSBAG_PATH} Already Exists...!"
fi

