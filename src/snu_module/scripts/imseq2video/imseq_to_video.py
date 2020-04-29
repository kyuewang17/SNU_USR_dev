"""
Image Sequence to Video
"""
import cv2
import os
import numpy as np
import glob


# Base Path
base_path = "/home/kyle/bag_files_temp/candidate_bags/"


def imseq_to_video(imseq_master_path, image_fmt, result_video_name, result_video_fmt, fourcc="DIVX", frame_rate=15):
    img_array = []
    width, height = -1, -1
    for file_idx, img_file_name in enumerate(sorted(os.listdir(imseq_master_path))):
        # Percentage
        percentage = float(file_idx+1) / len(sorted(os.listdir(imseq_master_path)))*100

        # Store Image Message
        mesg_str = "Appending Image...{%3.3f %s}" % (percentage, chr(37))
        print(mesg_str)

        # Check for File Extension
        _, file_extension = os.path.splitext(os.path.join(imseq_master_path, img_file_name))

        if file_extension.__contains__(image_fmt) is not True:
            assert 0, "Format must be %s...! (current file format is %s)" % (image_fmt, file_extension[1:])

        frame = cv2.imread(os.path.join(imseq_master_path, img_file_name))
        height, width, layers = frame.shape
        img_array.append(frame)
    size = (width, height)

    # Video Save Path
    result_video_name = result_video_name + "." + result_video_fmt
    video_save_path = os.path.join(base_path, result_video_name)

    # Video Writer
    out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*fourcc), frame_rate, size)

    # Write Images
    for img_array_idx in range(len(img_array)):
        # Percentage
        percentage = (float(img_array_idx+1) / len(img_array))*100

        # Work Message
        mesg_str = "Writing Images...{%3.3f %s}" % (percentage, chr(37))
        print(mesg_str)

        out.write(img_array[img_array_idx])
    out.release()


# Main
def main():
    # Folder Name
    # folder_name = "__image_sequence__[BAG_FILE]_[kiro_all]"
    # folder_name = "__image_sequence__[BAG_FILE]_[2019-09-26-17-19-40]"
    # folder_name = "__image_sequence__[BAG_FILE]_[2019-09-27-(1)]"
    # folder_name = "__image_sequence__[BAG_FILE]_[2019-09-27-(2)]"

    # folder_name = "__image_sequence__[BAG_FILE]_[snu_result_image_0926]"
    folder_name = "__image_sequence__[BAG_FILE]_[snu_result_image_0927]"

    # Parse to Get Video Name
    video_name = folder_name.split("]_[")[1].split("]")[0]

    # RGB Folder
    rgb_folder = os.path.join(os.path.join(base_path, folder_name), "rgb_image")

    # Image Sequence to Video
    imseq_to_video(rgb_folder, "png", video_name, "avi")


# Namespace
if __name__ == '__main__':
    main()
