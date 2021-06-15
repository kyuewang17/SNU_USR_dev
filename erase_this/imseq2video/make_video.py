"""
Image Sequence to Video

"""
import cv2
import os
import argparse


# Argument Parser
parser = argparse.ArgumentParser(
    prog="MAKE-VIDEO-FROM-IMAGE-SEQUENCE",
    description="Python Script to Generate Video of Specific Extension, from Image Sequences"
)
parser.add_argument(
    "--img_path", "-I",
    type=str,
    help="Source Image Sequence Path"
)
parser.add_argument(
    "--image_format",
    default="png", type=str,
    help="Image Format"
)
parser.add_argument(
    "--video_path", "-V",
    type=str,
    help="Target Video Path (Save Result Video at this Path)"
)
parser.add_argument(
    "--video_name", "-N",
    type=str,
    help="Result Video Name"
)
parser.add_argument(
    "--video_format", "-F",
    default="avi", type=str,
    help="Result Video Extension (Video Format)"
)
parser.add_argument(
    "--frame_rate", "-R",
    default=15, type=int,
    help="Result Video Frame Rate (Hz)"
)
parser.add_argument(
    "--forcc", "-C",
    default="DIVX", type=str,
    help="(opencv videowriter)"
)
args = parser.parse_args()


def main():
    img_array = []
    width, height = None, None
    for file_idx, img_file_name in enumerate(sorted(os.listdir(args.img_path))):
        # Percentage
        percentage = float(file_idx + 1) / len(sorted(os.listdir(args.img_path))) * 100

        # Store Image Message
        mesg_str = "Appending Image...{%3.3f %s}" % (percentage, chr(37))
        print(mesg_str)

        # Check for File Extension
        _, file_extension = os.path.splitext(os.path.join(args.img_path, img_file_name))
        if file_extension.__contains__(args.image_format) is not True:
            assert 0, "Format must be %s...! (current file format is %s)" % (args.image_format, file_extension[1:])

        frame = cv2.imread(os.path.join(args.img_path, img_file_name))
        height, width, layers = frame.shape
        img_array.append(frame)
    size = (width, height)

    # Video Save Path
    result_video_name = args.video_name + "." + args.video_format
    video_save_path = os.path.join(args.video_path, result_video_name)

    # Video Writer
    out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*args.forcc), args.frame_rate, size)

    # Write Images
    for img_array_idx in range(len(img_array)):
        # Percentage
        percentage = (float(img_array_idx + 1) / len(img_array)) * 100

        # Work Message
        mesg_str = "Writing Images...{%3.3f %s}" % (percentage, chr(37))
        print(mesg_str)

        out.write(img_array[img_array_idx])
    out.release()


if __name__ == "__main__":
    main()
