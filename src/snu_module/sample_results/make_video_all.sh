#!/usr/bin/env bash

VIDEO_SAVE_PATH="./"


BASE_PATH="./NONE/"

# Make DET Video
python2 make_video.py \
--img_path=${BASE_PATH}"/detections/color/" \
--video_path=${VIDEO_SAVE_PATH} \
--video_name="NONE__DET__color" \
--frame_rate="5" \

# Make TRK+ACL Video
python2 make_video.py \
--img_path=${BASE_PATH}"/tracklets/color/" \
--video_path=${VIDEO_SAVE_PATH} \
--video_name="NONE__TRK_ACL__color" \
--frame_rate="5" \


BASE_PATH="./TRK/"

# Make DET Video
python2 make_video.py \
--img_path=${BASE_PATH}"/detections/color/" \
--video_path=${VIDEO_SAVE_PATH} \
--video_name="TRK__DET__color" \
--frame_rate="5" \

# Make TRK+ACL Video
python2 make_video.py \
--img_path=${BASE_PATH}"/tracklets/color/" \
--video_path=${VIDEO_SAVE_PATH} \
--video_name="TRK__TRK_ACL__color" \
--frame_rate="5" \


BASE_PATH="./TRKC/"

# Make DET Video
python2 make_video.py \
--img_path=${BASE_PATH}"/detections/color/" \
--video_path=${VIDEO_SAVE_PATH} \
--video_name="TRKC__DET__color" \
--frame_rate="5" \

# Make TRK+ACL Video
python2 make_video.py \
--img_path=${BASE_PATH}"/tracklets/color/" \
--video_path=${VIDEO_SAVE_PATH} \
--video_name="TRKC__TRK_ACL__color" \
--frame_rate="5" \


BASE_PATH="./TRK_TRKC/"

# Make DET Video
python2 make_video.py \
--img_path=${BASE_PATH}"/detections/color/" \
--video_path=${VIDEO_SAVE_PATH} \
--video_name="TRK_TRKC__DET__color" \
--frame_rate="5" \

# Make TRK+ACL Video
python2 make_video.py \
--img_path=${BASE_PATH}"/tracklets/color/" \
--video_path=${VIDEO_SAVE_PATH} \
--video_name="TRK_TRKC__TRK_ACL__color" \
--frame_rate="5" \
