#!/usr/bin/env python
"""
- Description PLZ

"""
import os
import logging
import argparse
import ros_numpy
import ros_numpy
import numpy as np
from pandas import read_csv
import xmltodict
import rosbag
import rospy
import roslib
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import PointCloud2
from osr_msgs.msg import Annotation, Annotations
from cv_bridge import CvBridge
import cv2

from mmt_params import CAMERA_INFO as _CAMERAINFO
from camera_objects import IMAGE_MODAL_OBJ, LIDAR_MODAL_OBJ, MODAL_DATA_OBJ, MULTIMODAL_DATA_OBJ
from target_objects import BBOX, TARGET


def argument_parser():
    # Define Argument Parser
    parser = argparse.ArgumentParser(
        prog="seq2bag.py",
        description="Python Script to Multi-modal Sensor Data to ROS bag file"
    )
    parser.add_argument(
        "--base-path", "-P",
        help="Base Path"
    )
    parser.add_argument(
        "--start-fidx", "-S", default=2000, type=int,
        help="Starting Frame Index"
    )
    parser.add_argument(
        "--end-fidx", "-E", default=4500, type=int,
        help="Ending Frame Index"
    )
    parser.add_argument(
        "--override-mode", "-O", default=False, type=bool,
        help="Override Mode for Bag File if same exists...! (False: creates new bag file with additional character) (True: overrides previous bag file)"
    )

    # Parse Arguments
    args = parser.parse_args()

    return args


def set_logger(logging_level=logging.INFO):
    # Define Logger
    logger = logging.getLogger()

    # Set Logger Display Level
    logger.setLevel(level=logging_level)

    # Set Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("[%(levelname)s] | %(asctime)s : %(message)s")
    )
    logger.addHandler(stream_handler)

    return logger


def load_multimodal_data(base_path, logger, frame_interval):
    # Check if Base Path Exists
    if os.path.isdir(base_path) is False:
        raise AssertionError()

    # Get Folder Lists
    modal_lists = os.listdir(base_path)
    matchers = [
        "xml", ".bag"
    ]
    matchings = [s for s in modal_lists if any(xs in s for xs in matchers)]
    for matching in matchings:
        modal_lists.remove(matching)

    # Get Frame Interval
    if frame_interval[0] == 0 and frame_interval[1] == -1:
        min_fidx, max_fidx = -1, -1
    else:
        # Assertion
        assert 0 <= frame_interval[0] < frame_interval[1]
        min_fidx, max_fidx = frame_interval[0], frame_interval[1]

    # For XML Files,
    # seq_foldername = base_path.split("/")[-1]
    # if seq_foldername in ["1-01d", "1-02d", "1-06d", "1-07d"]:
    #     _time = "rgb"
    # else:
    #     _time = "thermal"
    #
    # xml_base_path = os.path.join(base_path, "xml", seq_foldername)
    # assert os.path.isdir(xml_base_path)
    xml_base_path = os.path.join(base_path, "xml")
    xml_file_list = sorted(os.listdir(xml_base_path))

    # Traverse through XML Files to Get Object Annotations
    anno_list = []
    anno_timing_list = []
    anno_modal_list = []
    for xml_idx, xml_filename in enumerate(xml_file_list):
        # Get XML File Path
        xml_filepath = os.path.join(xml_base_path, xml_filename)

        # Open XML File
        with open(xml_filepath) as anno_f:
            anno_data = xmltodict.parse(anno_f.read())

        # Get Modality
        anno_modal = anno_data["Annotation"]["subfolder"].encode("utf8")
        anno_modal_list.append(anno_modal)

        # Get XML Timing String
        xml_parsed_timing = xml_filename.split(".xml")[0].split("_")[2:]
        xml_date, xml_time, xml_fidx = xml_parsed_timing[0], xml_parsed_timing[1], xml_parsed_timing[2]
        anno_timing_list.append( {"date": xml_date, "time": xml_time, "fidx": xml_fidx})

        # Get Modal Image Width, Height
        anno_size_dict = anno_data["Annotation"]["size"]
        modal_width = int(anno_size_dict["width"].encode("utf8"))
        modal_height = int(anno_size_dict["height"].encode("utf8"))

        # Get Objects
        xml_objects = anno_data["Annotation"]["object"]
        if isinstance(xml_objects, list) is False:
            xml_objects = [xml_objects]

        # Initialize Object List
        obj_list = []

        for xml_obj in xml_objects:
            # Get Object ID
            if "index" in xml_obj:
                obj_id = int(xml_obj["index"].encode("utf8"))
            else:
                obj_id = None

            # Get Object Class
            obj_class = xml_obj["name"].encode("utf8")

            # Get Object Pose
            obj_pose = xml_obj["pose"].encode("utf8")

            # Get Object BBOX
            bbox_xmin = float(xml_obj["bndbox"]["xmin"].encode("utf8"))
            bbox_ymin = float(xml_obj["bndbox"]["ymin"].encode("utf8"))
            bbox_xmax = float(xml_obj["bndbox"]["xmax"].encode("utf8"))
            bbox_ymax = float(xml_obj["bndbox"]["ymax"].encode("utf8"))
            obj_bbox = BBOX(
                LT_X=bbox_xmin, LT_Y=modal_height-bbox_ymax,
                RB_X=bbox_xmax, RB_Y=modal_height-bbox_ymin
            )

            # Append to Object List
            obj_list.append(
                TARGET(
                    id=obj_id, bbox=obj_bbox, cls=obj_class, pose=obj_pose,
                    modal=anno_modal, date=xml_date, time=xml_time, fidx=xml_fidx
                )
            )

        # Append to anno_list
        anno_list.append(obj_list)

    # for each modalities
    modal_data_obj_dict = {}
    for modal_idx, modal in enumerate(modal_lists):
        # Join Path
        curr_modal_base_path = os.path.join(base_path, modal)

        # Filter Annotation Indices w.r.t. modality
        if modal != "RGB":
            anno_list_modal_indices = [i for i, m in enumerate(anno_modal_list) if m == modal]
        else:
            anno_list_modal_indices = [i for i, m in enumerate(anno_modal_list) if "color" in m]

        # Get Modal File Lists
        modal_obj_list = []
        curr_modal_file_lists = sorted(os.listdir(curr_modal_base_path))
        if len(curr_modal_file_lists) > 0:
            # Traverse through Files
            for file_idx, filename in enumerate(curr_modal_file_lists):
                # Break for Frame Interval
                if min_fidx == -1 and max_fidx == -1:
                    total_proc_frame_numbers = len(curr_modal_file_lists)+1
                else:
                    if min_fidx < file_idx < max_fidx:
                        total_proc_frame_numbers = (max_fidx - min_fidx)
                    else:
                        continue

                # Log
                if (file_idx + 1) % 100 == 0 and file_idx > 0:
                    logging_mesg = "[Modal: {}] ({}/{}) -- Processing ( #{} out of total #{} frames )".format(
                        modal, modal_idx+1, len(modal_lists), file_idx+1, total_proc_frame_numbers
                    )
                    logger.info(logging_mesg)

                # Current File Path
                curr_filepath = os.path.join(curr_modal_base_path, filename)

                # Parse Timestamp
                _date = filename.split(".")[0].split("_")[2]
                _time = filename.split(".")[0].split("_")[3]
                _fidx = filename.split(".")[0].split("_")[4]
                timestamp = {"date": _date, "time": _time, "fidx": _fidx}

                # Filter Annotation Indices w.r.t. filename (date-time-fidx)
                anno_list_timing_indices = [i for i, t in enumerate(anno_timing_list) if t == timestamp]

                # Intersection Indices
                anno_list_idx = list(set(anno_list_modal_indices).intersection(
                    set(anno_list_timing_indices)
                ))
                if len(anno_list_idx) > 1:
                    raise AssertionError()
                else:
                    annos = anno_list[anno_list_idx[0]] if len(anno_list_idx) != 0 else None

                # Initialize Modal Object
                if modal != "lidar":
                    modal_obj = IMAGE_MODAL_OBJ(modal=modal, timestamp=timestamp)

                    # Update Camera Parameters
                    modal_camerainfo = _CAMERAINFO[modal]
                    modal_obj.update_camera_parameters(
                        D=modal_camerainfo["D"], K=modal_camerainfo["K"],
                        R=modal_camerainfo["R"], P=modal_camerainfo["P"]
                    )
                else:
                    modal_obj = LIDAR_MODAL_OBJ()

                # Load Annotations
                modal_obj.set_annos(annos=annos)

                # Load Data
                if modal == "RGB":
                    data = cv2.cvtColor(cv2.imread(curr_filepath), cv2.COLOR_BGR2RGB)
                elif modal != "lidar":
                    data = cv2.imread(curr_filepath, -1)
                else:
                    pc_csv_data = read_csv(curr_filepath)

                    # Split
                    pc_split_data = pc_csv_data.to_dict('split')
                    pc_data_coord_list = pc_split_data["index"]
                    pc_data_intensity_list = pc_split_data["data"]
                    pc_data = []
                    data = np.zeros(len(pc_data_coord_list), dtype=[
                        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('d', np.float32)
                    ])
                    for idx, pc_data_coord in enumerate(pc_data_coord_list):
                        data[idx]['x'] = pc_data_coord[0]
                        data[idx]['y'] = pc_data_coord[1]
                        data[idx]['z'] = pc_data_coord[2]
                        data[idx]['d'] = pc_data_intensity_list[idx][0]

                # Append Data and Timestamp to Modal Object
                modal_obj.set_data(data=data, timestamp=timestamp)
                modal_obj_list.append(modal_obj)

        # Make Modal Data Object
        modal_data_obj = MODAL_DATA_OBJ(modal_obj_list=modal_obj_list, modal=modal)
        modal_data_obj_dict[modal] = modal_data_obj

    # Multimodal Data Object
    multimodal_data_obj = MULTIMODAL_DATA_OBJ(modal_data_obj_dict=modal_data_obj_dict)
    # _ = multimodal_data_obj.get_camera_parameters(0)

    return multimodal_data_obj


def generate_multimodal_bag_file(MMT_OBJ, logger, base_path, override_mode):
    # Get Bag File Name
    bag_name = os.path.split(base_path)[-1] + ".bag"

    # Initialize Bag File
    save_path = os.path.dirname(base_path)
    if os.path.isfile(os.path.join(save_path, bag_name)) is False:
        bag = rosbag.Bag(os.path.join(save_path, bag_name), "w")
    else:
        if override_mode is True:
            os.remove(os.path.join(save_path, bag_name))
            bag = rosbag.Bag(os.path.join(save_path, bag_name), "w")
        else:
            while True:
                bag_name = bag_name.split(".")[0] + "c.bag"
                if os.path.isfile(os.path.join(save_path, bag_name)) is False:
                    break
            bag = rosbag.Bag(os.path.join(save_path, bag_name), "w")

    # Iterate for Multimodal Sensors
    try:
        for fidx in range(len(MMT_OBJ)):
            # Logger
            iter_logger_msg = "Get Multi-modal Objects ({}/{})".format(
                fidx+1, len(MMT_OBJ)
            )
            logger.info(iter_logger_msg)

            mmt_data_dict, mmt_timestamp_dict, mmt_annos_dict = MMT_OBJ.get_data(fidx)
            mmt_camera_info_dict = MMT_OBJ.get_camera_info(fidx)
            bridge = CvBridge()
            for modal, modal_data in mmt_data_dict.items():
                # Define Each Modals' Encoding and Frame ID
                if modal == "RGB":
                    modal_encoding = "rgb8"
                    modal_frame_id = "osr/image_color"
                elif modal == "aligned_depth":
                    modal_encoding = "mono16"
                    modal_frame_id = "osr/image_" + modal.lower()
                elif modal == "thermal":
                    # modal_encoding = "mono16"
                    modal_encoding = "mono16"
                    modal_frame_id = "osr/image_" + modal.lower()
                elif modal == "infrared":
                    modal_encoding = "mono8"
                    modal_frame_id = "osr/image_ir"
                elif modal == "nightvision":
                    modal_encoding = "rgb8"
                    modal_frame_id = "osr/image_nv1"
                elif modal == "lidar":
                    modal_encoding = None
                    modal_frame_id = "osr/lidar_pointcloud"
                else:
                    raise NotImplementedError()

                # Get Each Modals' Camera Parameters
                modal_camera_info = mmt_camera_info_dict[modal]

                if modal_data is None:
                    pass
                else:
                    # Get Timestamp of the Current Modal
                    modal_timestamp = mmt_timestamp_dict[modal]
                    t_hour = int(modal_timestamp["time"][0:2])
                    t_min = int(modal_timestamp["time"][2:4])
                    t_sec = int(modal_timestamp["time"][4:])
                    t_seq = float(modal_timestamp["fidx"])

                    modal_stamp = rospy.rostime.Time.from_sec(
                        float(t_hour*3600+t_min*60+t_sec) + 0.1*t_seq
                    )

                    # Initialize ROS Message Type
                    if modal == "lidar":
                        ROS_LIDAR_PC2 = ros_numpy.point_cloud2.array_to_pointcloud2(
                            modal_data, modal_stamp, frame_id="velodyne_link"
                        )
                        bag.write(modal_frame_id, ROS_LIDAR_PC2, modal_stamp)
                    else:
                        # Get Annotations of the Current Modal
                        modal_annos = mmt_annos_dict[modal]

                        # Set ROS Topic Name
                        modal_annos_frame_id = "osr/annos_{}".format(modal)

                        # Initialize Annotations Msg
                        ROS_MODAL_ANNOS = Annotations()
                        ROS_MODAL_ANNOS.header.seq = fidx
                        ROS_MODAL_ANNOS.header.stamp = modal_stamp
                        ROS_MODAL_ANNOS.header.frame_id = modal_frame_id

                        if modal_annos is not None:
                            # Iterate for all 'modal_annos'
                            for modal_anno in modal_annos:

                                # Initialize Annotation Msg
                                ROS_MODAL_ANNO = Annotation()
                                ROS_MODAL_ANNO.header.seq = fidx
                                ROS_MODAL_ANNO.header.stamp = modal_stamp
                                ROS_MODAL_ANNO.header.frame_id = modal_frame_id

                                # Set 'bbox' Argument
                                xywh_bbox_arr = modal_anno.bbox.numpify()
                                ROS_MODAL_ANNO.lt_x = int(xywh_bbox_arr[0])
                                ROS_MODAL_ANNO.lt_y = int(xywh_bbox_arr[1])
                                ROS_MODAL_ANNO.rb_x = int(xywh_bbox_arr[2])
                                ROS_MODAL_ANNO.rb_y = int(xywh_bbox_arr[3])

                                # Set 'id' Argument
                                if modal_anno.id is None:
                                    ROS_MODAL_ANNO.id = -1
                                else:
                                    ROS_MODAL_ANNO.id = int(modal_anno.id)

                                # Set 'pose' Argument
                                if modal_anno.pose is None:
                                    ROS_MODAL_ANNO.pose.data = "other"
                                else:
                                    ROS_MODAL_ANNO.pose.data = modal_anno.pose

                                # Set 'class' Argument
                                ROS_MODAL_ANNO.cls.data = modal_anno.cls

                                # Set 'modal' Argument
                                ROS_MODAL_ANNO.modal.data = modal

                                # Append to Annotations
                                ROS_MODAL_ANNOS.annotations.append(ROS_MODAL_ANNO)

                        # Write to Bag File
                        bag.write(modal_annos_frame_id, ROS_MODAL_ANNOS, modal_stamp)

                        # ROS_MODAL_IMG = Image()
                        ROS_MODAL_IMG = bridge.cv2_to_imgmsg(modal_data, modal_encoding)
                        ROS_MODAL_IMG.header.seq = fidx
                        ROS_MODAL_IMG.header.stamp = modal_stamp
                        ROS_MODAL_IMG.header.frame_id = modal_frame_id

                        # CameraInfo
                        MODAL_CAMERA_INFO = modal_camera_info.to_CameraInfo(
                            width=modal_data.shape[1], height=modal_data.shape[0]
                        )

                        if bag is not None:
                            bag.write(modal_frame_id, ROS_MODAL_IMG, modal_stamp)
                            if modal != "nightvision":
                                # Write CameraInfo
                                if modal == "aligned_depth":
                                    bag.write("osr/image_depth_camerainfo", MODAL_CAMERA_INFO, modal_stamp)
                                else:
                                    bag.write(modal_frame_id + "_camerainfo", MODAL_CAMERA_INFO, modal_stamp)

    finally:
        if bag is not None:
            bag.close()


if __name__ == "__main__":
    # # base_path = "/home/snu/DATA/4th-year-dynamic/005"
    # base_path = "/mnt/wwn-0x50014ee212217ddb-part1/Unmanned Surveillance/2020/Detection/ICCAS2021"
    # folder_name = "1-10d"
    # base_path = os.path.join(base_path, folder_name)

    # Argparser
    args = argument_parser()
    # args.base_path = "/mnt/wwn-0x50014ee212217ddb-part1/Unmanned Surveillance/2020/Detection/ICCAS2021/1-01d"
    # args.base_path = "/mnt/wwn-0x50014ee212217ddb-part1/Unmanned Surveillance/2021/1-05d-iitp21"
    # args.base_path = "/mnt/wwn-0x50014ee212217ddb-part1/Unmanned Surveillance/rosbag/iitp21_final/day_night/1-07d/imseqs"
    # args.base_path = "/media/kyle/DATA003/mmosr_RAL_db/2020_retagged/1-10d"
    args.base_path = "/media/kyle/DATA003/mmosr_RAL_db/2021/MV01"

    # Set Logger
    logger = set_logger()

    # Load Multimodal Data Objects
    MMT_DATA_OBJ = load_multimodal_data(base_path=args.base_path, logger=logger, frame_interval=[args.start_fidx, args.end_fidx])

    # # Visualize Modal Data
    # vis_frames, vis_timestamps = MMT_DATA_OBJ.draw_modal_annos(modal="thermal")

    # # Make Named Window
    # winname = "thermal annos"
    # cv2.namedWindow("thermal annos")
    #
    # # Imshow Frames
    # fidx = 0
    # while True:
    #     vis_frame = vis_frames[fidx]
    #     print(vis_timestamps[fidx])
    #     fidx += 1
    #     if fidx == len(vis_frames):
    #         fidx = 0
    #     cv2.imshow(winname, vis_frame)
    #     cv2.waitKey(100)

    generate_multimodal_bag_file(MMT_OBJ=MMT_DATA_OBJ, logger=logger, base_path=args.base_path, override_mode=args.override_mode)
    pass
