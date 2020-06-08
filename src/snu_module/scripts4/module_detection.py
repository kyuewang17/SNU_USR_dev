"""
SNU Integrated Module v2.05
  - Detection (RefineDet)

"""
import os
import copy
import numpy as np
import scipy.misc
import torch
from detection_lib.backbone import RefineDetResNet34
from detection_lib.detector import RefineDet
from detection_lib.postproc import RefineDetPostProc
from detection_lib.framework import OneStageFramework
from detection_lib import util

thermal_camera_params = util.ThermalHeuristicCameraParams()

# import rescue.force_thermal_align_iitp_final_night as rgb_t_align


def load_model(opts, is_default_device=True):
    model_dir = opts.detector.model_dir
    if is_default_device is True:
        device = 0
    else:
        device = opts.detector.device

    backbone_path = os.path.join(model_dir, 'backbone.pth')
    detector_path = os.path.join(model_dir, 'detector.pth')

    backbone = RefineDetResNet34(opts.detector.detection_args, opts.detector.backbone_args)
    detector = RefineDet(opts.detector.detection_args, opts.detector.detector_args)
    backbone.build()
    detector.build()

    backbone.cuda(device)
    detector.cuda(device)
    backbone.load(backbone_path)
    detector.load(detector_path)

    postproc = RefineDetPostProc(opts.detector.detection_args, opts.detector.postproc_args, detector.anchors)
    framework = OneStageFramework(
        opts.detector.detection_args,
        network_dict={'backbone': backbone, 'detector': detector},
        postproc_dict={'detector': postproc})
    return framework


def detect(framework, sync_data_dict, opts, is_default_device=True):
    """
    [MEMO]
    -> Develop this code more~

    """
    # Select GPU Device for Detector Model
    device = (0 if is_default_device is True else opts.detector.device)

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
    color_frame = (sync_data_dict["color"].frame if "color" in sync_data_dict.keys() else None)
    thermal_frame = (sync_data_dict["thermal"].frame if "thermal" in sync_data_dict.keys() else None)

    # Get Color Frame Size
    color_size = (color_frame.shape[0], color_frame.shape[1])
    input_size = (opts.detector.detection_args['input_h'], opts.detector.detection_args['input_w'])

    if (opts.detector.sensor_dict["thermal"] is True) and (thermal_frame is not None):
        img_size = thermal_frame.shape[:2]
        img = torch.from_numpy(scipy.misc.imresize(thermal_frame, size=input_size)).unsqueeze(dim=2)
        img = torch.cat([img, img, img], dim=2)
    elif (opts.detector.sensor_dict["color"] is True) and (color_frame is not None):
        img_size = color_size
        img = torch.from_numpy(scipy.misc.imresize(color_frame, size=input_size))
    else:
        raise NotImplementedError

    img = img.permute(2, 0, 1).unsqueeze(dim=0).float().cuda(device) / 255.0

    # Feed-forward
    _, result_dict = framework.forward({"img": img}, train=False)

    # Get Result BBOX, Confidence, and Labels
    boxes, confs, labels = result_dict["boxes_l"][0], result_dict["confs_l"][0], result_dict["labels_l"][0]

    boxes[:, [0, 2]] *= (float(img_size[1]) / float(input_size[1]))
    boxes[:, [1, 3]] *= (float(img_size[0]) / float(input_size[0]))

    # Copy Before Conversion
    if opts.detector.sensor_dict["thermal"] is True:
        thermal_boxes = boxes.cpu().numpy()
    else:
        thermal_boxes = None

    # Get BBOX, Confidence, Labels
    boxes, confs, labels = \
        boxes.detach().cpu().numpy(), confs.detach().cpu().numpy(), labels.detach().cpu().numpy()

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


def standalone_detector():
    import time
    import cv2
    import argparse
    from config import cfg
    from options import snu_option_class

    # Save Detection Results Options
    is_save_det_results = True

    # Set Image Sequence Base Path
    imseq_base_path = "/mnt/usb-USB_3.0_Device_0_000000004858-0:0-part1"
    color_imseq_dir = os.path.join(
        imseq_base_path, "__image_sequence__[BAG_FILE]_[190823_kiro_lidar_camera_calib]", "color"
    )
    if os.path.isdir(color_imseq_dir) is False:
        assert 0, "[%s] is not a directory!" % imseq_base_path

    parser = argparse.ArgumentParser(description="StandAlone Detection Algorithm")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config", "190823_kiro_lidar_camera_calib.yaml"),
        type=str, help="configuration file"
    )
    args = parser.parse_args()

    # Merge Parsed cfg
    cfg.merge_from_file(args.config)

    # Load Option with Configuration
    opts = snu_option_class(cfg=cfg)

    # Load Detection Model
    detector_framework = load_model(opts=opts)

    # Select GPU Device for Detection Model
    device = opts.detector.device

    # Iterate through Color Frame Sequence Path
    color_frame_list = sorted(os.listdir(color_imseq_dir))

    # Read Sample Image (first frame)
    sample_color_frame = cv2.imread(os.path.join(color_imseq_dir, color_frame_list[0]))
    color_size = (sample_color_frame.shape[0], sample_color_frame.shape[1])
    input_size = (opts.detector.detection_args['input_h'], opts.detector.detection_args['input_w'])

    img_size = color_size

    # Initialize List of Numpy Array for Storing Frame-wise
    detection_result_list = []

    for frame_idx, frame_name in enumerate(color_frame_list):
        # Message
        if frame_idx % 10 == 0:
            det_msg = "Run Detector at Frame: [%06d / %06d]" % (frame_idx, len(color_frame_list))
            print(det_msg)

        # Get Full Path String of Current Frame Image
        frame_path = os.path.join(color_imseq_dir, frame_name)

        # Load Frame with cv2
        color_frame = cv2.imread(frame_path)

        # To PyTorch
        img = torch.from_numpy(scipy.misc.imresize(color_frame, size=input_size))
        img = img.permute(2, 0, 1).unsqueeze(dim=0).float().cuda(device) / 255.0

        # Feed-forward
        _, result_dict = detector_framework.forward({"img": img}, train=False)

        # Get Result BBOX, Confidence, and Labels
        boxes, confs, labels = result_dict["boxes_l"][0], result_dict["confs_l"][0], result_dict["labels_l"][0]

        boxes[:, [0, 2]] *= (float(img_size[1]) / float(input_size[1]))
        boxes[:, [1, 3]] *= (float(img_size[0]) / float(input_size[0]))

        # Get BBOX, Confidence, Labels
        dets, confs, labels = \
            boxes.detach().cpu().numpy(), confs.detach().cpu().numpy(), labels.detach().cpu().numpy()

        # Remove Too Small Detections
        keep_indices = []
        for det_idx, det in enumerate(dets):
            if det[2] * det[3] >= opts.detector.tiny_area_threshold:
                keep_indices.append(det_idx)
        dets = dets[keep_indices, :]
        confs = confs[keep_indices, :]
        labels = labels[keep_indices, :]

        curr_frame_detection_result_array = np.zeros((dets.shape[0], 7))
        for det_idx, det in enumerate(dets):
            curr_frame_detection_result_array[det_idx, :] = np.array(
                [
                 frame_idx,
                 det[0], det[1], det[2], det[3],
                 confs[det_idx, 0],
                 labels[det_idx, 0]
                ]
            )
        detection_result_list.append(curr_frame_detection_result_array)

    # Make Save File
    if is_save_det_results is True:
        det_result_filename = "det_result.txt"
        det_result_file_path = os.path.join(os.path.dirname(color_imseq_dir), det_result_filename)

        if os.path.isfile(det_result_file_path) is True:
            print("[WARNING] Overwriting Detection Result File...!")
            time.sleep(3)

        # Open(make) Save File
        with open(det_result_file_path, "w") as f:
            for fidx, detection_result_array in enumerate(detection_result_list):
                if detection_result_array.shape[0] != 0:
                    print("Saving Detection at Frame: [%06d]" % fidx)
                    for d in detection_result_array:
                        f.write("%s %s %s %s %s %s %s\n" % (d[0], d[1], d[2], d[3], d[4], d[5], d[6]))



if __name__ == "__main__":
    standalone_detector()
