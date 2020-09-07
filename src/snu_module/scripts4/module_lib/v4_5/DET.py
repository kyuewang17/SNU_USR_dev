"""
SNU Integrated Module v2.05
  - Detection (RefineDet)

"""
import os
import copy
import cv2
import numpy as np
import scipy.misc
import torch
from detection_lib.detector import RefineDet, YOLOv4
from detection_lib import util

thermal_camera_params = util.ThermalHeuristicCameraParams()

detector_dict = {
    "refinedet": RefineDet,
    "yolov4": YOLOv4,
}

# import rescue.force_thermal_align_iitp_final_night as rgb_t_align
cnt = 0
def load_model(opts, is_default_device=True):
    device = 0 if is_default_device else opts.detector.device 

    detection_args = opts.detector.detection_args
    detector_args = opts.detector.detector_args

    detector = detector_dict[detector_args["name"]](
        detection_args, detector_args)
    detector.build()
    detector.cuda(device)
    detector.load(opts.detector.model_dir)
    return detector


def detect(detector, sync_data_dict, opts, is_default_device=True):
    """
    [MEMO]
    -> Develop this code more~

    """
    # Select GPU Device for Detector Model
    device = (0 if is_default_device is True else opts.detector.device)
    torch.autograd.set_grad_enabled(False)

    # Get Color and Thermal Image Frame
    """
    Color (numpy array)
    - shape: (h, w, 3)
    - range: [0, 255]
    - if there is no color image frame, "color_frame" is None
    
    Thermal (numpy array)
    - shape: (h, w)
    - range: [0, 255]
    - if there is no thermal image frame, "thermal_frame" is None
    """
    color_frame = (sync_data_dict["color"].get_data() if "color" in sync_data_dict.keys() else None)
    thermal_frame = (sync_data_dict["thermal"].get_data() if "thermal" in sync_data_dict.keys() else None)

    ######## by JIS (maybe...?) #########
    # global cnt
    # scipy.misc.imsave(os.path.join('/home/mipal/Project/MUIN/png_img/',str(cnt)+'.png'), color_frame)
    # cnt += 1
    #####################################
    
    # Get Color Frame Size
    color_size = (color_frame.shape[0], color_frame.shape[1])
    input_size = (opts.detector.detection_args['input_h'], opts.detector.detection_args['input_w'])

    if (opts.detector.sensor_dict["thermal"] is True) and (thermal_frame is not None):
        img_size = thermal_frame.shape[:2]
        # img = torch.from_numpy(scipy.misc.imresize(thermal_frame, size=input_size)).unsqueeze(dim=2)
        img = torch.from_numpy(
            cv2.resize(thermal_frame, dsize=input_size)
        ).unsqueeze(dim=2)
        img = torch.cat([img, img, img], dim=2)
    elif (opts.detector.sensor_dict["color"] is True) and (color_frame is not None):
        img_size = color_size
        # img = torch.from_numpy(scipy.misc.imresize(color_frame, size=input_size))
        img = torch.from_numpy(
            cv2.resize(color_frame, dsize=input_size)
        )
    else:
        raise NotImplementedError

    # Feed-forward / Get BBOX, Confidence, Labels
    # result_dict = darknet.inference_(framework, img)
    boxes, confs, labels = detector.forward(img)
    boxes[:, [0, 2]] *= (float(img_size[1]) / float(input_size[1]))
    boxes[:, [1, 3]] *= (float(img_size[0]) / float(input_size[0]))

    # Copy Before Conversion
    thermal_boxes = boxes.copy() if opts.detector.sensor_dict["thermal"] else None

    # Detection Results
    det_results = np.concatenate([boxes, confs, labels], axis=1)

    if opts.detector.sensor_dict["thermal"] is True:
        return det_results, thermal_boxes
    elif opts.detector.sensor_dict["color"] is True:
        return det_results
    else:
        raise NotImplementedError


def detect_old(framework, imgStruct_dict, opts, is_default_device=True):
    if is_default_device is True:
        device = 0
    else:
        device = opts.detector.device

    # rgb.shape: (h, w, 3), rgb.range: [0, 255], img.type: np array
    # thermal.shape: (h, w), thermal.range: [0, 255], thermal.type: np array
    rgb = imgStruct_dict['rgb'].frame.raw
    thermal = imgStruct_dict['thermal'].frame.raw

    # rgb_size = rgb.shape[:2]
    rgb_size = (480, 640)
    input_size = (opts.detector.detection_args['input_h'], opts.detector.detection_args['input_w'])
    if opts.detector.thermal and (thermal is not None):
        img_size = thermal.shape[:2]
        img = torch.from_numpy(scipy.misc.imresize(thermal, size=input_size)).unsqueeze(dim=2)
        img = torch.cat([img, img, img], dim=2)
    else:
        img_size = rgb_size
        img = torch.from_numpy(scipy.misc.imresize(rgb, size=input_size))
    img = img.permute(2, 0, 1).unsqueeze(dim=0).float().cuda(device) / 255.0

    _, result_dict = framework.forward({'img': img}, train=False)
    boxes = result_dict['boxes_l'][0]
    confs = result_dict['confs_l'][0]
    labels = result_dict['labels_l'][0]

    boxes[:, [0, 2]] *= (float(img_size[1]) / float(input_size[1]))
    boxes[:, [1, 3]] *= (float(img_size[0]) / float(input_size[0]))
    # boxes= fbbox.zxs_to_bboxes(boxes, is_torch=True)

    # Copy before Conversion
    thermal_boxes = boxes.cpu().numpy()

    # if opts.detector.thermal and (thermal is not None):
    #     for i in range(len(boxes)):
    #         boxes[i, 0], boxes[i, 1] = util.thermal_coord_to_rgb_coord(
    #             thermal_camera_params, rgb_size, boxes[i, 0], boxes[i, 1])
    #         boxes[i, 2], boxes[i, 3] = util.thermal_coord_to_rgb_coord(
    #             thermal_camera_params, rgb_size, boxes[i, 2], boxes[i, 3])

    # if opts.detector.thermal and (thermal is not None):
    #     for i in range(len(boxes)):
    #         boxes[i, 0], boxes[i, 1] = rgb_t_align.thermal_to_rgb_coord(boxes[i, 0], boxes[i, 1])
    #         boxes[i, 2], boxes[i, 3] = rgb_t_align.thermal_to_rgb_coord(boxes[i, 2], boxes[i, 3])

    boxes = boxes.detach().cpu().numpy()
    confs = confs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    # boxes.shape: (#boxes, 4), confs.shape: (#boxes, 1), labels.shape: (#boxes, 1)
    det_results = np.concatenate([boxes, confs, labels], axis=1)
    # det_results.shape: (#boxes, 6), [n, 0~3]: xywh, [n, 4]: confidence, [n, 5]: label

    if opts.detector.thermal and (thermal is not None):
        return det_results, thermal_boxes
    else:
        return det_results


def standalone_detector(detection_model):
    pass


if __name__ == "__main__":
    # Load Model (framework)
    detection_model = []

    standalone_detector(detection_model)