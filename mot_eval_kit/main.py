#!/usr/bin/env python
"""
MOT Evaluation Kit
    - Code made by Kyuewang Lee
    - Using motmetrics
        - [URL] https://github.com/cheind/py-motmetrics



"""
import cv2
import argparse
import os
import random
import copy
import numpy as np
import logging
import motmetrics
import rosbag
from cv_bridge import CvBridge
import matplotlib
import matplotlib.pyplot as plt


class COLORMAP(object):
    def __init__(self):
        # Referred to https://sashamaps.net/docs/resources/20-colors/

        # List of Color Maps
        self.colormaps = [
            # Maroon
            (128, 0, 0),
            # Red
            (230, 25, 75),
            # Pink
            (250, 190, 212),
            # Brown
            (170, 110, 40),
            # Orange
            (245, 130, 48),
            # Yellow
            (255, 225, 25),
            # Teal
            (0, 128, 128),
            # Cyan
            (70, 240, 240),
            # Navy
            (0, 0, 128),
            # Blue
            (0, 130, 200),
            # Lavender
            (220, 190, 255),
            # Magenta
            (240, 50, 230),
            # Purple
            (145, 30, 180)
        ]

    def __call__(self, idx, **kwargs):
        # Check for OpenCV Mode
        is_opencv = kwargs.get("is_opencv", False)
        assert isinstance(is_opencv, bool)

        # Check for Random Index Mode
        is_random_idx = kwargs.get("is_rand_idx", False)
        assert isinstance(is_random_idx, bool)

        # Get Colormap
        colormap = self.colormaps[(idx % len(self.colormaps))]

        if is_opencv is True:
            colormap = (colormap[2], colormap[1], colormap[0])

        return colormap


class ANNOTATION(object):
    def __init__(self, annotation_msg):
        msg_name = annotation_msg.__class__.__name__
        assert msg_name == "_osr_msgs__Annotation"

        # Header Message
        self.header = annotation_msg.header

        # ID Data
        self.id = annotation_msg.id

        # Class Data
        self.cls = annotation_msg.cls.data

        # Modal Info
        self.modal = annotation_msg.modal.data

        # Pose Data
        self.pose = annotation_msg.pose.data

        # Bounding Box Coordinates
        self.lt_x = annotation_msg.lt_x
        self.lt_y = annotation_msg.lt_y
        self.width = annotation_msg.rb_x - self.lt_x
        self.height = annotation_msg.rb_y - self.lt_y

        # Iteration Counter
        self.__iter_counter = 0

    def __len__(self):
        return 4

    def __repr__(self):
        return "anno__Cls[{}]_ID[{}]".format(self.cls, self.id)

    def __getitem__(self, idx):
        item_list = [self.lt_x, self.lt_y, self.width, self.height]
        return item_list[idx]

    def __iter__(self):
        return self

    def next(self):
        try:
            iter_item = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return iter_item

    def numpify(self):
        ret_arr = np.empty(4)
        for idx in range(len(self)):
            ret_arr[idx] = self[idx]
        return ret_arr

    def check_bbox_sanity(self, frame_width, frame_height):
        check_bool = True

        # X-coord check
        if self.lt_x < 0 or (self.lt_x + self.width) > frame_width:
            check_bool = check_bool and False

        # Y-coord check
        if self.lt_y < 0 or (self.lt_y + self.width) > frame_height:
            check_bool = check_bool and False

        # Width and Height check
        if self.width <= 0 or self.height <= 0:
            check_bool = check_bool and False

        return check_bool


class TRAJECTORY(object):
    def __init__(self, trajectory_msg):
        msg_name = trajectory_msg.__class__.__name__
        assert msg_name == "_osr_msgs__Track"

        # ID Data
        self.id = trajectory_msg.id

        # Class Data
        self.cls = trajectory_msg.type

        # Pose Data
        self.pose = trajectory_msg.posture

        # Bounding Box Coordinates
        cx, cy = trajectory_msg.bbox_pose.x, trajectory_msg.bbox_pose.y
        w, h = trajectory_msg.bbox_pose.width, trajectory_msg.bbox_pose.height
        self.lt_x, self.lt_y = cx - w / 2.0, cy - h / 2.0
        self.width, self.height = w, h

        # Initialize Unique Color for Visualization
        self.id_color = None

        # Iteration Counter
        self.__iter_counter = 0

    def __len__(self):
        return 4

    def __repr__(self):
        return "trk__Cls[{}]_ID[{}]".format(self.cls, self.id)

    def __getitem__(self, idx):
        item_list = [self.lt_x, self.lt_y, self.width, self.height]
        return item_list[idx]

    def __iter__(self):
        return self

    def next(self):
        try:
            iter_item = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return iter_item

    def numpify(self):
        ret_arr = np.empty(4)
        for idx in range(len(self)):
            ret_arr[idx] = self[idx]
        return ret_arr

    def check_bbox_sanity(self, frame_width, frame_height):
        check_bool = True

        # X-coord check
        if self.lt_x < 0 or (self.lt_x + self.width) > frame_width:
            check_bool = check_bool and False

        # Y-coord check
        if self.lt_y < 0 or (self.lt_y + self.width) > frame_height:
            check_bool = check_bool and False

        # Width and Height check
        if self.width <= 0 or self.height <= 0:
            check_bool = check_bool and False

        return check_bool


# Evaluator Container Object
class EVALUATOR(object):
    def __init__(self, msg, logger):
        # Set Logger
        self.logger = logger

        self.header = msg.header
        self.annotations = msg.annos
        self.trajectories = msg.tracks

        # Frame Image
        self.frame = None

        # Visualization Frame Image
        self.vis_frame = None

        # Check Header Sanity
        self.__check_header_sanity()

        # Convert Annotation and Track Into familiar formats
        self.__cvt_annos_format()
        self.__cvt_tracks_format()

        # Evaluation Flag
        self.is_evaluated = False

    def __check_header_sanity(self):
        # Check for Annotations (Header-Check, cross-check integrated)
        header_seq, header_stamp = None, None
        for annos in self.annotations:
            if np.logical_xor(header_seq is None, header_stamp is None) is True:
                raise AssertionError()
            else:
                if header_seq is not None:
                    assert (annos.header.seq == header_seq) and annos.header.stamp == header_stamp, "Error btw Annotation Headers"
                    assert annos.header.stamp == self.header.stamp, "Error with Evaluator Header"
                header_seq, header_stamp = annos.header.seq, annos.header.stamp

        # Sanity Check Logging Message
        self.logger.info("[PASSED] Checking Sanity for Annotation...")

    def __cvt_annos_format(self):
        new_annos = []
        for anno in self.annotations:
            new_annos.append(ANNOTATION(anno))
        self.annotations = new_annos
        self.logger.info("Annotations Successfully Converted...!")

    def __cvt_tracks_format(self):
        new_tracks = []
        for track in self.trajectories:
            new_tracks.append(TRAJECTORY(track))
        self.trajectories = new_tracks
        self.logger.info("Trajectories Successfully Converted...!")

    def get_max_trk_id(self):
        max_id = None
        for trk in self.trajectories:
            if max_id is None:
                max_id = trk.id
            else:
                if max_id < trk.id:
                    max_id = trk.id
        return max_id

    def check_bbox_sanity_and_destroy_invalid(self):
        new_annos = []
        for anno in self.annotations:
            anno_bbox_sanity = anno.check_bbox_sanity(
                frame_width=self.frame.shape[1], frame_height=self.frame.shape[0]
            )
            if anno_bbox_sanity is True:
                new_annos.append(anno)
        self.annotations = new_annos

        new_trks = []
        for trk in self.trajectories:
            trk_bbox_sanity = trk.check_bbox_sanity(
                frame_width=self.frame.shape[1], frame_height=self.frame.shape[0]
            )
            if trk_bbox_sanity is True:
                new_trks.append(trk)
        self.trajectories = new_trks

    def gather_frame(self, frame, frame_msg_header):
        assert self.frame is None

        # Get Timestamps
        abs_d_timestamp = abs(self.header.stamp - frame_msg_header.stamp)
        if abs_d_timestamp.secs > 0:
            raise AssertionError("Timestamp Error...!")
        else:
            if (abs_d_timestamp.nsecs * 1e-9) > 0.11:
                raise AssertionError("Timestamp Error...!")

        # Gather Frame
        self.frame = frame

    # TODO: ID visualization, Color-coding for Different IDs, etc...
    def draw(self, **kwargs):
        assert self.frame is not None

        # Get KWARGS Options
        is_anno_draw = kwargs.get("draw_anno", True)
        is_trk_draw = kwargs.get("draw_trk", True)
        txt_font = kwargs.get("txt_font", cv2.FONT_HERSHEY_COMPLEX_SMALL)

        # Copy Visualization Frame and Adjust Dims and Types
        vis_frame = copy.deepcopy(self.frame)
        if vis_frame.dtype != np.uint8:
            vis_frame = vis_frame.astype(np.uint8)
        if len(vis_frame.shape) == 2:
            ch_vis_frame = vis_frame[:, :, None]
            vis_frame = np.dstack((ch_vis_frame, ch_vis_frame, ch_vis_frame))

        # Draw Annotation BBOX
        if is_anno_draw is True:
            for anno in self.annotations:
                cv2.rectangle(
                    vis_frame,
                    (int(np.floor(anno.lt_x)), int(np.floor(anno.lt_y))),
                    (int(np.floor(anno.lt_x + anno.width)), int(np.floor(anno.lt_y + anno.height))),
                    (0, 255, 0), 2
                )

        # Draw Trajectory BBOX
        if is_trk_draw is True:
            for trk in self.trajectories:
                # Get Trajectory Color
                trk_color = trk.id_color if trk.id_color is not None else (255, 0, 0)

                # Draw Trajectory BBOX
                cv2.rectangle(
                    vis_frame,
                    (int(np.floor(trk.lt_x)), int(np.floor(trk.lt_y))),
                    (int(np.floor(trk.lt_x + trk.width)), int(np.floor(trk.lt_y + trk.height))),
                    trk_color, 2
                )

                # Center Coordinates
                cx, cy = (trk.lt_x + trk.width / 2.0), (trk.lt_y + trk.height / 2.0)

                # Put Trajectory ID Text
                fontScale = 1
                thickness = 2

                trk_id_str = "{}".format(trk.id)
                (tx, ty) = \
                    cv2.getTextSize(trk_id_str, txt_font, fontScale=fontScale, thickness=thickness)[0]
                txt_C = [cx, cy - trk.height / 2.0]
                txt_LT = [txt_C[0] - tx / 2.0, txt_C[1] - ty / 2.0]
                txt_RB = [txt_C[0] + tx / 2.0, txt_C[1] + ty / 2.0]

                pad_x, pad_y = 1, 2
                txtbox_LT = [txt_LT[0] - pad_x, txt_LT[1] - pad_y]
                txtbox_RB = [txt_RB[0] + pad_x, txt_RB[1] + pad_y]
                cv2.rectangle(
                    vis_frame,
                    (int(txtbox_LT[0]), int(txtbox_LT[1])), (int(txtbox_RB[0]), int(txtbox_RB[1])),
                    (255, 255, 255), -1
                )
                cv2.putText(
                    vis_frame,
                    trk_id_str, (int(txt_LT[0]), int(txt_RB[1])), txt_font,
                    color=trk_color,
                    fontScale=fontScale, thickness=thickness,
                    bottomLeftOrigin=False
                )

        # Set Visualization Frame
        self.vis_frame = vis_frame

    def show(self, winname, **kwargs):
        # Get Draw Target
        draw_target_frame = kwargs.get("draw_target_frame")

        if draw_target_frame == "frame":
            show_frame = copy.deepcopy(self.frame)
            if show_frame.dtype != np.uint8:
                show_frame = show_frame.astype(np.uint8)
            if len(show_frame.shape) == 2:
                show_frame = show_frame[:, :, None]
                show_frame = np.dstack((show_frame, show_frame, show_frame))
        else:
            assert self.vis_frame is not None
            show_frame = self.vis_frame

        # Imshow
        cv2.imshow(winname=winname, mat=show_frame)

    def eval(self, acc):
        """
        Code re-worked and referred from py-motmetrics github readme page
            - < URL > : https://github.com/cheind/py-motmetrics
                      : https://github.com/julianoks/670_final_proj/blob/d56acd3157a37e067e10a7fdf10f0ad888740c10/eval/evaluate.py
        """

        # Get Annotations and Trajectories
        annos, trks = self.annotations, self.trajectories

        # Compute IOU Matrix using motmetrics module
        anno_bboxes, anno_ids = [], []
        for anno in annos:
            anno_bboxes.append(anno.numpify())
            anno_ids.append(anno.id)
        anno_bboxes = np.asarray(anno_bboxes)

        trk_bboxes, trk_ids = [], []
        for trk in trks:
            trk_bboxes.append(trk.numpify())
            trk_ids.append(trk.id)
        trk_bboxes = np.asarray(trk_bboxes)

        # IOU Matrix
        iou_matrix = motmetrics.distances.iou_matrix(anno_bboxes, trk_bboxes, max_iou=1)

        # Set Distance Matrix (in this case, IOU Matrix is the Distance Matrix)
        dist_matrix = iou_matrix

        # Update Accumulator
        if len(anno_bboxes) > 0:
            try:
                acc.update(anno_ids, trk_ids, dist_matrix)
            except:
                print(anno_ids)
                print(trk_ids)
                print(dist_matrix)

        # Mark Evaluated Flag as 'True'
        self.is_evaluated = True

        # Return
        return acc


class MOT_EVAL_OBJECT(object):
    def __init__(self, bag_filepath, logger, **kwargs):
        # Set Logger
        self.logger = logger

        # Set Sequence Name
        bag_filename = bag_filepath.split("/")[-1].split(".")[0]
        self.seq_name = bag_filename

        # Load Bag File
        bag = rosbag.Bag(bag_filepath, "r")

        # NOTE: Set Target Topic Names
        eval_topicname = kwargs.get("eval_topicname")
        frame_topicname = kwargs.get("frame_topicname")
        frame_cvt_encoding = kwargs.get("frame_cvt_encoding")

        # Get Result Save Path
        result_save_base_path = \
            os.path.join(kwargs.get("save_base_path"), bag_filename)
        if os.path.isdir(result_save_base_path) is True:
            self.result_save_base_path = result_save_base_path
        else:
            raise AssertionError()

        # Traverse through Bag Messages, Gather existing Topics
        bag_topics = []
        iter_eval, iter_frame = 0, 0

        eval_stamps, frame_stamps = [], []
        for topic, msg, _ in bag.read_messages(topics=[]):
            if topic not in bag_topics:
                bag_topics.append(topic)
            if topic == eval_topicname:
                iter_eval += 1
                eval_stamps.append(msg.header.stamp)
            if topic == frame_topicname:
                iter_frame += 1
                frame_stamps.append(msg.header.stamp)

        matched_eval_indices, matched_frame_indices = [], []
        for eval_idx, eval_stamp in enumerate(eval_stamps):
            for frame_idx, frame_stamp in enumerate(frame_stamps):
                if eval_idx in matched_eval_indices or frame_idx in matched_frame_indices:
                    continue
                dstamp = abs(eval_stamp - frame_stamp)
                if dstamp.secs == 0:
                    if dstamp.nsecs*1e-9 < 0.11:
                        matched_eval_indices.append(eval_idx)
                        matched_frame_indices.append(frame_idx)
                        break

        # Gather Evaluation Message ( "/osr/eval" )
        self.evaluators = []
        for idx, (_, msg, _) in enumerate(bag.read_messages(topics=eval_topicname)):
            gathering_eval_msg = "Gathering Evaluation ({} / {})".format(idx+1, iter_eval)
            self.logger.info(gathering_eval_msg)
            if idx in matched_eval_indices:
                self.evaluators.append(EVALUATOR(msg, self.logger))

        # Gather Frames
        bridge = CvBridge()
        eval_idx_cnt = 0
        for idx, (_, msg, _) in enumerate(bag.read_messages(topics=frame_topicname)):
            gathering_frame_msg = "Gathering Frame ({} / {})".format(idx+1, iter_frame)
            self.logger.info(gathering_frame_msg)
            if idx in matched_frame_indices:
                self.evaluators[eval_idx_cnt].gather_frame(
                    frame=bridge.imgmsg_to_cv2(img_msg=msg, desired_encoding=frame_cvt_encoding),
                    frame_msg_header=msg.header
                )
                eval_idx_cnt += 1

        # Sort-out Invalid Annotations or Trajectories
        for eval_idx in range(len(self)):
            self.evaluators[eval_idx].check_bbox_sanity_and_destroy_invalid()

        # Iteration Counter
        self.__iter_counter = 0

        # Traverse through annotations in evaluators, to correct id considering class labels
        self.__correct_annos_id()

        # Traverse through trajectories in evaluators, size-down id
        self.__correct_tracks_id()

        # Traverse for All Trajectories, get Maximum ID and give ID Color for the ID's proportion
        max_trk_id = None
        for eval in self:
            # Get Max Trk ID
            curr_max_trk_id = eval.get_max_trk_id()

            if max_trk_id is None:
                max_trk_id = curr_max_trk_id
            else:
                if max_trk_id < curr_max_trk_id:
                    max_trk_id = curr_max_trk_id

        cmap = COLORMAP()
        rsi = list(range(max_trk_id+1))
        random.shuffle(rsi)
        for eval_idx, eval in enumerate(self):
            for trk_idx, trk in enumerate(eval.trajectories):
                # colormap = list(cmap(rsi[trk.id]))
                colormap = list(cmap(trk.id, is_opencv=True))
                colormap[0] = int(np.floor(colormap[0]))
                colormap[1] = min(int(np.floor(colormap[1])), 10)
                colormap[2] = int(np.floor(colormap[2]))
                colormap = tuple(colormap)
                self.evaluators[eval_idx].trajectories[trk_idx].id_color = colormap

    def __len__(self):
        return len(self.evaluators)

    def __repr__(self):
        return self.seq_name

    def __getitem__(self, idx):
        return self.evaluators[idx]

    def __iter__(self):
        return self

    def next(self):
        try:
            iter_item = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return iter_item

    def __correct_annos_id(self):
        # Get Annotation Data
        anno_data = self.get_wrapped_data(pool_data="annotations")

        # Flatten All Annotation Data into Single List object
        annos = []
        for _annos in anno_data:
            for _anno in _annos:
                annos.append(_anno)

        # Get Class and ID List
        cls_list, id_list = [], []
        for anno in annos:
            cls_list.append(anno.cls)
            id_list.append(anno.id)

        net_cls = sorted(list(set(cls_list)))
        clswise_max_id = []
        cls_wise_id_dict = {k: {"idx": None, "id": None} for k in net_cls}
        for curr_cls in net_cls:
            # Get Current Class Indices from Class List
            curr_cls_indices = [
                i for i in range(len(cls_list)) if cls_list[i] == curr_cls
            ]

            # Get ID of Current Class Indices
            matching_id_indices, matching_id_values = [], []
            for idx, id_val in enumerate(id_list):
                if idx in curr_cls_indices:
                    matching_id_indices.append(idx)
                    matching_id_values.append(id_val)

            # Max ID of Current Class
            clswise_max_id.append(max(matching_id_values))

            # To Dict
            if len(matching_id_indices) > 0:
                cls_wise_id_dict[curr_cls]["idx"] = matching_id_indices
                cls_wise_id_dict[curr_cls]["id"] = matching_id_values

        # Traverse Annotation Data, apply ID Correction
        for eval_idx, evaluator in enumerate(self.evaluators):
            for anno_idx, annotation in enumerate(evaluator.annotations):
                # Get Class Label
                cls = annotation.cls

                # Get ID Adder
                id_adder = clswise_max_id[net_cls.index(cls)] if cls != net_cls[0] else 0

                # Get original ID
                id = annotation.id

                # Add ID
                self.evaluators[eval_idx].annotations[anno_idx].id = id + id_adder

            # Logger
            self.logger.info("Correcting ID of Evaluators ({} / {})".format(eval_idx+1, len(self.evaluators)))

    def __correct_tracks_id(self):
        # Get Trajectory Data
        trk_data = self.get_wrapped_data(pool_data="trajectories")

        # Traverse through all Trajectory Objects, get minimum ID as possible
        trk_min_id = None
        for trks in trk_data:
            for trk in trks:
                if trk_min_id is None:
                    trk_min_id = trk.id
                else:
                    if trk_min_id > trk.id:
                        trk_min_id = trk.id

        # Traverse trajectory Data, apply correction
        for eval_idx, evaluator in enumerate(self.evaluators):
            for trk_idx, trajectory in enumerate(evaluator.trajectories):
                # Correct
                self.evaluators[eval_idx].trajectories[trk_idx].id -= trk_min_id

    def get_wrapped_data(self, **kwargs):
        # Detect for Indices
        indices = kwargs.get("indices", list(range(len(self))))
        if indices is not None:
            assert isinstance(indices, (list, tuple, np.ndarray))
            if isinstance(indices, tuple):
                indices = list(tuple)
            elif isinstance(indices, np.ndarray):
                assert indices.size == 2
                indices = indices.reshape(2).tolist()

        # Detect for Data Types
        _data_type_pool_ = ["annotations", "trajectories", "header", "frame"]
        pools = kwargs.get("pool_data", _data_type_pool_)
        assert isinstance(pools, (str, list, tuple))
        if isinstance(pools, str):
            assert pools in _data_type_pool_
            pools = [pools]
        else:
            assert set(pools).issubset(set(_data_type_pool_))
            if isinstance(pools, tuple):
                pool = list(pools)

        # Get Attributes that matches Pools and Indices
        out_data = []
        for idx, evaluator in enumerate(self):
            # Continue if index is not regarded
            if idx not in indices:
                continue

            # If Pools length is 1, just append to out_data
            if len(pools) == 1:
                out_data.append(getattr(evaluator, pools[0]))

            else:
                # Initialize Pool Data Dict
                pool_data = {k: None for k in pools}

                # Iterate for Pool
                for pool in pools:
                    pool_data[pool] = getattr(evaluator, pool)

                # Append
                out_data.append(pool_data)

        return out_data

    def get_frames(self):
        out_frames = []
        for evaluator in self:
            frame_dict = {
                "header": evaluator.header,
                "frame": evaluator.frame
            }
            out_frames.append(frame_dict)
        return out_frames

    def get_annotations(self):
        out_annos = []
        for evaluator in self:
            out_annos.append(evaluator.annotations)
        return out_annos

    def draw(self, **kwargs):
        # Get KWARGS Options
        is_anno_draw = kwargs.get("draw_anno", True)
        is_trk_draw = kwargs.get("draw_trk", True)

        # for evaluators,
        for eval_idx in range(len(self)):
            self.evaluators[eval_idx].draw(draw_anno=is_anno_draw, draw_trk=is_trk_draw)

    def show(self, **kwargs):
        # Get Draw Target
        draw_target_frame = kwargs.get("draw_target_frame")

        # Show Delay
        show_delay = kwargs.get("show_delay", 1)

        # Set Window Name
        winname = self.seq_name

        # Make Named Window
        cv2.namedWindow(winname=winname)

        # Iterate for Evaluators,
        for evaluator in self:
            evaluator.show(winname=winname, draw_target_frame=draw_target_frame)
            cv2.waitKey(show_delay)

        # Destroy Window
        cv2.destroyWindow(winname=winname)

    def save_vis_frame(self, is_override):
        for eval_idx, evaluator in enumerate(self):
            # Set Frame File Name and Path
            frame_filename = "{:08d}.png".format(eval_idx)
            frame_filepath = \
                os.path.join(self.result_save_base_path, frame_filename)
            if is_override is False:
                assert os.path.isfile(frame_filepath) is False
            else:
                if os.path.isfile(frame_filepath) is True:
                    save_logger_msg = "[Override] Saving Results for [{}]...! ({} / {})".format(
                        self, eval_idx + 1, len(self)
                    )
                else:
                    save_logger_msg = "Saving Results for [{}]...! ({} / {})".format(
                        self, eval_idx + 1, len(self)
                    )
                self.logger.info(save_logger_msg)

            # Lookup Original Frame, predict if frame is RGB or not
            if len(evaluator.frame.shape) == 3:
                cv2.imwrite(frame_filepath, cv2.cvtColor(evaluator.vis_frame, cv2.COLOR_RGB2BGR))
            elif len(evaluator.frame.shape) == 2:
                cv2.imwrite(frame_filepath, evaluator.vis_frame)
            else:
                raise NotImplementedError()

    def eval(self, metrics, motchallenge_format=True, return_acc=False):
        """
        Code re-worked and referred from py-motmetrics github readme page
            - < URL > : https://github.com/cheind/py-motmetrics
                      : https://github.com/julianoks/670_final_proj/blob/d56acd3157a37e067e10a7fdf10f0ad888740c10/eval/evaluate.py
        """
        # Create an Accumulator that will be updated during each frame
        acc = motmetrics.MOTAccumulator(auto_id=True)

        for eval_idx, evaluator in enumerate(self.evaluators):
            # Evaluation
            acc = evaluator.eval(acc=acc)

            # Logging
            self.logger.info("Evaluating...! ({} / {})".format(eval_idx+1, len(self.evaluators)))

        # Return Accuracy
        if return_acc is True:
            return acc

        # Create Evaluation Metrics and Compute Evaluation
        # TODO: Change MOTP according to MOTChallenge GitHub
        mh = motmetrics.metrics.create()
        # summary = mh.compute(acc, metrics=["num_frames", "mota", "motp"], name="acc")
        summary = mh.compute(acc, metrics=metrics, name="acc")

        if motchallenge_format is True:
            strsummary = motmetrics.io.render_summary(
                summary,
                formatters=mh.formatters,
                namemap=motmetrics.io.motchallenge_metric_names
            )
            print(strsummary)

        else:
            print(summary)


class MOT_EVAL_OBJECTS(object):
    def __init__(self, bag_filepaths, logger, **kwargs):
        # Set Logger
        self.logger = logger

        # Get Save Base Path
        save_base_path = kwargs.get("save_base_path")

        # Unpack KWARGS
        eval_topicname = kwargs.get("eval_topicname")
        frame_topicname = kwargs.get("frame_topicname")
        frame_cvt_encoding = kwargs.get("frame_cvt_encoding")

        # Initialize List of MOT Evaluation Object
        self.eval_objects = []
        assert isinstance(bag_filepaths, (str, list))
        if isinstance(bag_filepaths, str):
            bag_filepaths = [bag_filepaths]
        for bag_filepath in bag_filepaths:
            self.eval_objects.append(
                MOT_EVAL_OBJECT(
                    bag_filepath=bag_filepath, logger=logger, save_base_path=save_base_path,
                    eval_topicname=eval_topicname, frame_topicname=frame_topicname,
                    frame_cvt_encoding=frame_cvt_encoding
                )
            )

        # Set Iteration Counter
        self.__iter_counter = 0

    def __len__(self):
        return len(self.eval_objects)

    def __getitem__(self, idx):
        return self.eval_objects[idx]

    def __iter__(self):
        return self

    def next(self):
        try:
            iter_item = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return iter_item

    def draw(self, **kwargs):
        # Get KWARGS Options
        is_anno_draw = kwargs.get("draw_anno", True)
        is_trk_draw = kwargs.get("draw_trk", True)

        # for all evaluation objects
        for eval_obj_idx in range(len(self)):
            self.eval_objects[eval_obj_idx].draw(draw_anno=is_anno_draw, draw_trk=is_trk_draw)

    def show(self, **kwargs):
        # Get Draw Target
        draw_target_frame = kwargs.get("draw_target_frame")
        assert draw_target_frame in ["frame", "vis_frame"]

        # Show Delay
        show_delay = kwargs.get("show_delay", 1)

        # Save Bool Options
        is_save = kwargs.get("is_save", False)
        assert isinstance(is_save, bool)

        # Iterate for Evaluator Objects
        for eval_obj in self:
            eval_obj.show(draw_target_frame=draw_target_frame, show_delay=show_delay, is_save=is_save)

    def draw_and_show(self, **kwargs):
        # Get KWARGS Options
        is_anno_draw, is_trk_draw = kwargs.get("draw_anno", True), kwargs.get("draw_trk", True)
        is_show = kwargs.get("is_show", False)
        show_delay = kwargs.get("show_delay", 1)
        draw_target_frame = "vis_frame"

        # Draw for all evaluation objects
        for eval_obj_idx in range(len(self)):
            self.eval_objects[eval_obj_idx].draw(draw_anno=is_anno_draw, draw_trk=is_trk_draw)

        # Iterate for Evaluator Objects and Show (visualize via OpenCV)
        if is_show is True:
            for eval_obj in self:
                eval_obj.show(draw_target_frame=draw_target_frame, show_delay=show_delay)

    def save(self, **kwargs):
        # Get Duplicate Options
        is_override = kwargs.get("is_override", True)
        assert isinstance(is_override, bool)

        for eval_obj in self:
            eval_obj.save_vis_frame(is_override=is_override)

    def evaluate(self, metrics, save_path, motchallenge_format=True):
        # Evaluate individual Evaluator
        acc_list, bag_names = [], []
        for eval_obj in self:
            acc_list.append(
                eval_obj.eval(
                    metrics=metrics, motchallenge_format=motchallenge_format, return_acc=True
                )
            )
            bag_names.append("{}".format(eval_obj))

        # Compute Many
        mh = motmetrics.metrics.create()
        summary = mh.compute_many(
            acc_list, metrics=metrics, names=bag_names
        )
        strsummary = motmetrics.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=motmetrics.io.motchallenge_metric_names
        )
        print(strsummary)

        f = open(os.path.join(save_path, "mot_results.txt"), "w")
        f.write(strsummary.encode("utf8"))
        f.close()


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


def argument_parser():
    # Define Argument Parser
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Python Script for MOT Evaluation"
    )
    parser.add_argument(
        "--base-path", "-P",
        help="Base Path"
    )

    # Parse Arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Argument Parser
    args = argument_parser()

    # Evaluation Bag Base Path
    # __EVAL_BAG_FILE_BASEPATH__ = "/media/kyle/DATA003/mmosr_RAL_db/iitp_2020_mot_results/1-10d/01"
    # __EVAL_BAG_FILE_BASEPATH__ = "/media/kyle/DATA003/mmosr_RAL_db/MMOSDX_DB_RESULTS/iitp_2021_mot_results/01"
    __EVAL_BAG_FILE_BASEPATH__ = args.base_path
    assert os.path.isdir(__EVAL_BAG_FILE_BASEPATH__)

    # Search for All bag files, sort ( search for *.bag files )
    eval_bag_files = os.listdir(__EVAL_BAG_FILE_BASEPATH__)
    eval_bag_filenames = sorted([fn for fn in eval_bag_files if fn.endswith("bag")])
    eval_bag_filepaths = [
        os.path.join(__EVAL_BAG_FILE_BASEPATH__, f) for f in eval_bag_filenames
    ]

    # Generate Save Path for Bag Files
    save_base_path = \
        os.path.join(__EVAL_BAG_FILE_BASEPATH__, "results")
    for eval_bag_filename in eval_bag_filenames:
        bag_save_path = os.path.join(save_base_path, eval_bag_filename.split(".")[0])
        if os.path.isdir(bag_save_path) is False:
            os.makedirs(bag_save_path)

    # Set Logger
    logger = set_logger(logging_level=logging.INFO)

    # Set Target Topic Names and Encoding
    eval_topicname = "/osr/eval"
    frame_topicname = "/osr/image_color"
    frame_cvt_encoding = "8UC3"

    # Set Evaluation Metrics
    # metrics = ["num_frames", "mota", "motp", "idf1", "recall", "precision"]
    metrics = motmetrics.metrics.motchallenge_metrics

    # # Test for Single Eval Object
    # test_single_eval_object(
    #     logger=logger, eval_topicname=eval_topicname, frame_topicname=frame_topicname,
    #     frame_cvt_encoding=frame_cvt_encoding, metrics=metrics
    # )
    #

    # ------------------------------ #
    # Test for Multiple Eval Objects #
    # ------------------------------ #
    # Initialize Evaluation Objects
    eval_objs = MOT_EVAL_OBJECTS(
        bag_filepaths=eval_bag_filepaths, save_base_path=save_base_path,
        logger=logger, eval_topicname=eval_topicname, frame_topicname=frame_topicname,
        frame_cvt_encoding=frame_cvt_encoding, metrics=metrics
    )

    # Evaluate
    eval_objs.evaluate(metrics=metrics, save_path=__EVAL_BAG_FILE_BASEPATH__, motchallenge_format=True)

    # Draw and Show
    eval_objs.draw_and_show(
        draw_anno=True, draw_trk=True, is_show=False, draw_target_frame="vis_frame", show_delay=50
    )

    # Save
    eval_objs.save()
