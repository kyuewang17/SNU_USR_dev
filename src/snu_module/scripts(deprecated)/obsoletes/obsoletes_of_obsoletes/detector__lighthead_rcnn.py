"""
Outdoor Unmanned Surveillance Robot SNU Detector Module v1.0

    - Code based on [Lighthead-RCNN]

    - Code written/modified by : (SNU Students)

"""

from __future__ import division, print_function

import os
import numpy as np
import tensorflow as tf
from functools import partial

# Import from Detection Module
from obsoletes.lighthead_rcnn.lib import dataset, network_desp
from obsoletes.lighthead_rcnn.lib.config import config
from obsoletes.lighthead_rcnn import nms
from obsoletes.lighthead_rcnn.lib.detection_opr import DetBox
from obsoletes.lighthead_rcnn.lib.detection_opr import clip_boxes, bbox_transform_inv


# Load Lighthead-rcnn Neural Network Model
def load_model(model_path, dev='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = dev

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    sess = tf.Session(config=tfconfig)

    net = network_desp.Network()  # create network
    inputs = net.get_inputs()  # create input node (placeholder)

    net.inference('TEST', inputs)  # define forward path
    test_collect_dict = net.get_test_collection()  # create result node
    test_collect = [it for it in test_collect_dict.values()]
    saver = tf.train.Saver()

    saver.restore(sess, model_path)
    return partial(sess.run, test_collect), inputs


# Detector
def detector(image, infer_func, inputs):
    # image:
    #   - input data, image

    ori_shape = image.shape

    if config.eval_resize == False:
        resized_img, scale = image, 1
    else:
        resized_img, scale = dataset.resize_img_by_short_and_max_size(
            image, config.eval_image_short_size, config.eval_image_max_size)
    height, width = resized_img.shape[0:2]

    resized_img = resized_img.astype(np.float32) - config.image_mean
    resized_img = np.ascontiguousarray(resized_img[:, :, [2, 1, 0]])

    im_info = np.array(
        [[height, width, scale, ori_shape[0], ori_shape[1], 0]],
        dtype=np.float32)

    feed_dict = {inputs[0]: resized_img[None, :, :, :], inputs[1]: im_info}

    _, scores, pred_boxes, rois = infer_func(feed_dict=feed_dict)

    boxes = rois[:, 1:5] / scale

    # if cfg.TEST.BBOX_REG:
    pred_boxes = bbox_transform_inv(boxes, pred_boxes)
    pred_boxes = clip_boxes(pred_boxes, ori_shape)

    pred_boxes = pred_boxes.reshape(-1, config.num_classes, 4)
    result_boxes = []
    for j in range(1, config.num_classes):
        inds = np.where(scores[:, j] > config.test_cls_threshold)[0]
        cls_scores = scores[inds, j]
        cls_bboxes = pred_boxes[inds, j, :]
        cls_dets = np.hstack((cls_bboxes, cls_scores[:, np.newaxis])).astype(
            np.float32, copy=False)

        keep = nms(cls_dets, config.test_nms)
        cls_dets = np.array(cls_dets[keep, :], dtype=np.float, copy=False)
        for i in range(cls_dets.shape[0]):
            db = cls_dets[i, :]
            dbox = DetBox(
                db[0], db[1], db[2] - db[0], db[3] - db[1],
                tag=config.class_names[j], score=db[-1])

            result_boxes.append(dbox)

    if len(result_boxes) > config.test_max_boxes_per_image:
        result_boxes = sorted(
            result_boxes, reverse=True, key=lambda t_res: t_res.score) \
            [:config.test_max_boxes_per_image]

    temp_result_boxes = []
    for box in result_boxes:
        if box.score > config.test_vis_threshold and box.tag == 'person':
            x, y, w, h = box.x, box.y, box.w, box.h
            tag = box.tag
            score = box.score
            dbox = dict()
            dbox['x'] = x
            dbox['y'] = y
            dbox['w'] = w
            dbox['h'] = h
            dbox['d'] = -1
            dbox['class'] = tag
            dbox['score'] = score
            temp_result_boxes.append(dbox)

    result_dict = dict()
    result_dict['result_boxes'] = temp_result_boxes
    return result_dict


# Get Detection BBOXES
def get_detection_bboxes(result_dict):
    result_boxes = result_dict['result_boxes']
    curr_dets = np.empty((len(result_boxes), 4))
    for idx, box in enumerate(result_boxes):
        curr_dets[idx, 0] = box['x']
        curr_dets[idx, 1] = box['y']
        curr_dets[idx, 2] = box['x'] + box['w']
        curr_dets[idx, 3] = box['y'] + box['h']
    return curr_dets




























