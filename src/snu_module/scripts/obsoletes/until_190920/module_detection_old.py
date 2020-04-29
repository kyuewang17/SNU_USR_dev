import os
import datetime
import numpy as np
import scipy.misc
import torch
from detection_lib.backbone import RefineDetResNet34
from detection_lib.detector import RefineDet
from detection_lib.postproc import RefineDetPostProc
from detection_lib.framework import OneStageFramework
import detection_option


def load_model(model_dir, device=0):
    backbone_path = os.path.join(model_dir, 'backbone.pth')
    detector_path = os.path.join(model_dir, 'detector.pth')

    backbone = RefineDetResNet34(detection_option.detection_args, detection_option.backbone_args)
    detector = RefineDet(detection_option.detection_args, detection_option.detector_args)
    backbone.build()
    detector.build()

    backbone.cuda(device)
    detector.cuda(device)
    backbone.load(backbone_path)
    detector.load(detector_path)

    postproc = RefineDetPostProc(detection_option.detection_args, detection_option.postproc_args, detector.anchors)
    framework = OneStageFramework(
        detection_option.detection_args,
        network_dict={'backbone': backbone, 'detector': detector},
        postproc_dict={'detector': postproc})
    return framework


def detect(framework, img, device=0):
    # img.shape: (h, w, 3), img.range: [0, 255], img.type: np array
    img_size = img.shape[:2]
    input_size = (detection_option.detection_args['input_h'], detection_option.detection_args['input_w'])
    img = torch.from_numpy(scipy.misc.imresize(img, size=input_size))
    img = img.permute(2, 0, 1).unsqueeze(dim=0).float().cuda(device) / 255

    _, result_dict = framework.forward({'img': img}, train=False)
    boxes = result_dict['boxes_l'][0]
    confs = result_dict['confs_l'][0]
    labels = result_dict['labels_l'][0]

    boxes[:, [0, 2]] *= (float(img_size[1]) / float(input_size[1]))
    boxes[:, [1, 3]] *= (float(img_size[0]) / float(input_size[0]))
    # boxes= fbbox.zxs_to_bboxes(boxes, is_torch=True)

    boxes = boxes.detach().cpu().numpy()
    confs = confs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    # boxes.shape: (#boxes, 4), confs.shape: (#boxes, 1), labels.shape: (#boxes, 1)
    det_results = np.concatenate([boxes, confs, labels], axis=1)
    # det_results.shape: (#boxes, 6), [n, 0~3]: xywh, [n, 4]: confidence, [n, 5]: label

    return det_results