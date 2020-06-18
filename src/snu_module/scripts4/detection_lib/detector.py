import os
import time
import math
import itertools
import numpy as np
import cv2
import torch
import torch.nn as nn
from detection_lib.darknet import darknet
from .backbone import RefineDetResNet34
from .postproc import RefineDetPostProc
from .__base__ import DetectorBase
from .__module__ import FirstTCB, TCB


class YOLOv4(DetectorBase):
    def __init__(self, global_args, network_args):
        super(YOLOv4, self).__init__(global_args, network_args)
        self.net_width = network_args['net_width']
        self.net_height = network_args['net_height']
        self.meta_path = network_args['meta_path']
        self.hier_thresh = network_args['hier_thresh']
        self.nms_thresh = network_args['nms_thresh']
        self.thresh = network_args['thresh']
        self.net = None
        self.meta = None

        self.name2number_map = {
            # 'background': 0, 'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4,
            'background': 0, 'person': 1, 'bicycle': 2, 'car': 3, 'motorbike': 4,
            'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,
            'fire hydrant': 11, 'stop sign': 12, 'parking meter': 13, 'bench': 14, 'bird': 15,
            'cat': 16, 'dog': 17, 'horse': 18, 'sheep': 19, 'cow': 20, 'elephant': 21, 'bear': 22,
            'zebra': 23, 'giraffe': 24, 'backpack': 25, 'umbrella': 26, 'handbag': 27,
            'tie': 28, 'suitcase': 29, 'frisbee': 30, 'skis': 31, 'snowboard': 32,
            'sports ball': 33, 'kite': 34, 'baseball bat': 35, 'baseball glove': 36,
            'skateboard': 37, 'surfboard': 38, 'tennis racket': 39, 'bottle': 40,
            'wine glass': 41, 'cup': 42, 'fork': 43, 'knife': 44, 'spoon': 45, 'bowl': 46,
            'banana': 47, 'apple': 48, 'sandwich': 49, 'orange': 50, 'broccoli': 51,
            'carrot': 52, 'hot dog': 53, 'pizza': 54, 'donut': 55, 'cake': 56,
            'chair': 57, 'couch': 58, 'potted plant': 59, 'bed': 60, 'dining table': 61,
            'toilet': 62, 'tv': 63, 'laptop': 64, 'mouse': 65, 'remote': 66, 'keyboard': 67,
            'cell phone': 68, 'microwave': 69, 'oven': 70, 'toaster': 71, 'sink': 72,
            'refrigerator': 73, 'book': 74, 'clock': 75, 'vase': 76, 'scissors': 77,
            'teddy bear': 78, 'hair drier': 79, 'toothbrush': 80
        }
        self.name2number_map_reduced = {
            'background':0, 'person':1, 'car':2, 'bike':3
        }
        
    def build(self):
        self.meta = darknet.lib.get_metadata(self.meta_path.encode('utf-8'))

    def forward(self, img):
        # s_t = time.time()
        # img_path = '/home/mipal/Desktop/test.jpeg'
        # img = darknet.load_image(img_path, 0, 0)
        # print(img.data)
        # print(type(img.data), img.data.shape)
        img = img.numpy()
        img = cv2.resize(img, (self.net_width, self.net_height), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)
        img_arr = np.ascontiguousarray(img.flat, dtype=np.float32) / 255.0
        img_data = img_arr.ctypes.data_as(darknet.POINTER(darknet.c_float))
        img_strt = darknet.IMAGE(img.shape[2], img.shape[1], img.shape[0], img_data)
        # print(darknet.network_width(self.net), darknet.network_height(self.net))
        # print(img.shape)
        # exit()
        box_infos = darknet.detect_image(
            self.net, self.meta, img_strt, self.thresh, 
            self.hier_thresh, self.nms_thresh, False)
        # darknet.free_image(img)
        
        boxes, confs, labels = list(), list(), list()
        if len(box_infos) == 0 :
            boxes.append((0.0, 0.0, 0.0, 0.0))
            confs.append(float(1.0))
            labels.append(self.name2number_map_reduced['background'])
        else:    
            for box_info in box_infos:
                boxes.append(box_info[2])
                confs.append(box_info[1])
                labels.append(self.name2number_map_reduced[box_info[0]])

        boxes = np.array(boxes)
        boxes[:, 0:2] = boxes[:, 0:2] - boxes[:, 2:4] / 2.0
        boxes[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4] 
        confs = np.expand_dims(np.array(confs), axis=1)
        labels = np.expand_dims(np.array(labels), axis=1)
        # cv2.waitKey(0)
        # print("inf_time:", time.time() - s_t)
        return boxes, confs, labels

    def load(self, load_dir):
        config_path = os.path.join(load_dir, 'yolov4.cfg')
        weight_path = os.path.join(load_dir, 'yolov4.weights')
        self.net = darknet.load_net_custom(
            config_path.encode('utf-8'), weight_path.encode('utf-8'), 0, 1)


class RefineDet(DetectorBase):
    def __init__(self, detection_args, network_args):
        super(RefineDet, self).__init__(detection_args, network_args)
        self.n_classes = detection_args['n_classes']
        self.input_h = float(detection_args['input_h'])
        self.input_w = float(detection_args['input_w'])

        self.is_bnorm = network_args['is_bnorm']
        self.tch_ch = network_args['tcb_ch']
        self.fmap_chs = network_args['fmap_chs']
        self.fmap_sizes = network_args['fmap_sizes']
        self.fmap_steps = network_args['fmap_steps']

        self.anch_scale = network_args['anch_scale']
        self.anch_min_sizes = network_args['anch_min_sizes']
        self.anch_max_sizes = network_args['anch_max_sizes']
        self.anch_aspect_ratios = network_args['anch_aspect_ratios']
        self.n_boxes_list = network_args['n_boxes_list']
        self.is_anch_clip = network_args['is_anch_clip']

        # self.pos_anch_thresh = network_args['pos_anch_thresh']
        self.backbone = RefineDetResNet34(detection_args, network_args['backbone_args'])
        self.postproc = RefineDetPostProc(detection_args, network_args['postproc_args'])
        self.anchors = None

    def build(self):
        self.anchors = self.__create_anchor_boxes__()
        self.backbone.build()

        net = dict()
        net['arm_loc_nets'], net['arm_conf_nets'] = self.__create_arm__()
        net['odm_loc_nets'], net['odm_conf_nets'] = self.__create_odm__()
        net['tcb_nets'] = self.__create_tdb__()
        self.net = nn.ModuleDict(net)
        # TODO: initialization

    def forward(self, img):
        self.train(False)
        img = img.float().cuda(self.device).permute(2, 0, 1).unsqueeze(dim=0) / 255.0

        fmap_list = self.backbone.forward(img)
        p4 = self.net['tcb_nets'][3](fmap_list[3])
        p3 = self.net['tcb_nets'][2](fmap_list[2], p4)
        p2 = self.net['tcb_nets'][1](fmap_list[1], p3)
        p1 = self.net['tcb_nets'][0](fmap_list[0], p2)

        pyramid_fmap_list = [p1, p2, p3, p4]
        arm_loc, arm_conf = self.__forward_arm_head__(fmap_list)
        odm_loc, odm_conf = self.__forward_odm_head__(pyramid_fmap_list)
        # refined_anchors, anch_ignore_flags = self.__refine_arm_anchors__(arm_predictions)
        
        boxes_l, confs_l, labels_l = self.postproc.process(
            arm_loc, arm_conf, odm_loc, odm_conf, self.anchors)
        boxes = boxes_l[0].detach().cpu().numpy() 
        confs = confs_l[0].detach().cpu().numpy()
        labels = labels_l[0].detach().cpu().numpy()
        return boxes, confs, labels

    def cuda(self, device=0):
        self.net.cuda(device)
        self.backbone.cuda(device)
        self.anchors = self.anchors.cuda(device)
        self.device = device

    def load(self, load_dir):
        super(RefineDet, self).load(os.path.join(load_dir, "detector.pth"))
        self.backbone.load(os.path.join(load_dir, "backbone.pth"))

    def train(self, mode):
        self.backbone.train(mode)
        self.net.train(mode)

    def __forward_arm_head__(self, fmap_list):
        """
        Apply ARM heads to forward features and concatenate results of all heads
        into one variable.
        :param fmap_dict: a list, features of four layers.
        :return arm_loc_pred: location predictions for layers,
            shape is (batch, num_anchors, 4)
        :return arm_conf_pred: confidence predictions for layers,
            shape is (batch, num_anchors, 2)
        """
        arm_loc_pred = []
        arm_conf_pred = []
        num_classes = 2
        # Apply ARM heads to forward_features and concatenate results
        for (x, l, c) in zip(fmap_list, self.net['arm_loc_nets'], self.net['arm_conf_nets']):
            arm_loc_pred.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf_pred.append(c(x).permute(0, 2, 3, 1).contiguous())
        # (batch, num_anchors*pred)
        arm_loc_pred = torch.cat([o.view(o.size(0), -1)
                                  for o in arm_loc_pred], 1)
        arm_conf_pred = torch.cat([o.view(o.size(0), -1)
                                   for o in arm_conf_pred], 1)
        arm_loc_pred = arm_loc_pred.view(arm_loc_pred.size(0), -1, 4)
        arm_conf_pred = arm_conf_pred.view(arm_conf_pred.size(0), -1, num_classes)
        return arm_loc_pred, arm_conf_pred

    def __forward_odm_head__(self, pyramid_features):
        """
        Apply ODM heads to pyramid features and concatenate results of all heads
        into one variable.
        :param pyramid_features: a list, features of four layers.
        :return odm_loc_pred: location predictions for layers,
            shape is (batch, num_anchors, 4)
        :return odm_conf_pred: confidence predictions for layers,
            shape is (batch, num_anchors, num_classes)
        """
        odm_loc_pred = []
        odm_conf_pred = []
        # Apply ODM heads to pyramid features and concatenate results.
        for (x, l, c) in zip(pyramid_features, self.net['odm_loc_nets'], self.net['odm_conf_nets']):
            odm_loc_pred.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf_pred.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc_pred = torch.cat([o.view(o.size(0), -1)
                                  for o in odm_loc_pred], 1)
        odm_conf_pred = torch.cat([o.view(o.size(0), -1)
                                   for o in odm_conf_pred], 1)
        # Shape is (batch, num_anchors, 4)
        odm_loc_pred = odm_loc_pred.view(odm_loc_pred.size(0), -1, 4)
        # Shape is (batch, num_anchors, num_classes)
        odm_conf_pred = odm_conf_pred.view(odm_conf_pred.size(0), -1,
                                           self.n_classes)
        return odm_loc_pred, odm_conf_pred

    # ARM(Object Detection Module)
    def __create_arm__(self):
        loc_layers = []
        conf_layers = []
        num_classes = 2
        # Relu module in self.layer# does not have 'out_channels' attribution,
        # so we must supply layers_out_channles as inputs for 'Conv2d'
        assert len(self.fmap_chs) == len(self.n_boxes_list)
        for k, v in enumerate(self.fmap_chs):
            loc_layers += [nn.Conv2d(v, self.n_boxes_list[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v, self.n_boxes_list[k] * num_classes, kernel_size=3, padding=1)]

        arm_loc = nn.ModuleList(loc_layers)
        arm_conf = nn.ModuleList(conf_layers)
        return arm_loc, arm_conf

    # ODM(Object Detection Module)
    def __create_odm__(self):
        loc_layers = []
        conf_layers = []
        # internal_channels
        for k in range(len(self.fmap_chs)):
            loc_layers += [nn.Conv2d(self.tch_ch, self.n_boxes_list[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(self.tch_ch, self.n_boxes_list[k] * self.n_classes, kernel_size=3, padding=1)]

        odm_loc = nn.ModuleList(loc_layers)
        odm_conf = nn.ModuleList(conf_layers)
        return odm_loc, odm_conf

    def __create_tdb__(self):
        pyramid_layer0 = TCB(self.fmap_chs[0], self.tch_ch, self.tch_ch, self.is_bnorm)
        pyramid_layer1 = TCB(self.fmap_chs[1], self.tch_ch, self.tch_ch, self.is_bnorm)
        pyramid_layer2 = TCB(self.fmap_chs[2], self.tch_ch, self.tch_ch, self.is_bnorm)
        pyramid_layer3 = FirstTCB(self.fmap_chs[3], self.tch_ch, self.is_bnorm)
        return nn.ModuleList([pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3])

    def __create_anchor_boxes__(self):
        mean = []
        if self.input_h > self.input_w:
            ratio_for_h = float(self.input_h) / float(self.input_w)
            ratio_for_w = 1
        else:
            ratio_for_h = 1
            ratio_for_w = float(self.input_w) / float(self.input_h)

        for k, f in enumerate(self.fmap_sizes):
            for i, j in itertools.product(range(int(f * ratio_for_h)), range(int(f * ratio_for_w))):

                # for i, j in product(range(f), repeat=2):
                f_k_w = self.input_w / self.fmap_steps[k]
                f_k_h = self.input_h / self.fmap_steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k_w
                cy = (i + 0.5) / f_k_h

                # aspect_ratio: 1
                # rel size: min_size
                s_k_w = self.anch_min_sizes[k] / self.input_w
                s_k_h = self.anch_min_sizes[k] / self.input_h
                mean += [cx, cy, s_k_w, s_k_h]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                if len(self.anch_max_sizes) > 0:
                    s_k_prime_w = math.sqrt(s_k_w * (self.max_sizes[k] / self.input_w))
                    s_k_prime_h = math.sqrt(s_k_w * (self.max_sizes[k] / self.input_h))
                    mean += [cx, cy, s_k_prime_w, s_k_prime_h]

                # rest of aspect ratios
                for ar in self.anch_aspect_ratios[k]:
                    mean += [cx, cy, s_k_w * math.sqrt(ar), s_k_h / math.sqrt(ar)]
                    mean += [cx, cy, s_k_w / math.sqrt(ar), s_k_h * math.sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)

        if self.is_anch_clip:
            output.clamp_(max=1, min=0)
        return output
