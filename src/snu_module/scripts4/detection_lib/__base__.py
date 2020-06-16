import abc
import numpy as np
import torch
import torch.nn as nn


class NetworkBase(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, detection_args, network_args):
        super(NetworkBase, self).__init__()
        self.detection_args = detection_args
        self.network_args = network_args
        self.device = None
        self.net = None

    @ abc.abstractmethod
    def build(self):
        pass

    @ abc.abstractmethod
    def forward(self, *x):
        pass

    def cuda(self, device=0):
        if self.net is not None:
            self.net.cuda(device)
            self.device = device

    def save(self, save_path):
        if self.net is not None:
            torch.save(self.net.state_dict(), save_path)
            print('[SAVE] %s' % save_path)

    def load(self, load_path):
        if self.net is not None:
            self.net.load_state_dict(torch.load(load_path, map_location='cuda:%d' % self.device))
            print('[LOAD] %s' % load_path)


class DetectorBase(NetworkBase):
    def __init__(self, detection_args, network_args):
        super(DetectorBase, self).__init__(detection_args, network_args)


class BackboneBase(NetworkBase):
    def __init__(self, detection_args, network_args):
        super(BackboneBase, self).__init__(detection_args, network_args)
        self.pretrained = network_args['pretrained']

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        mean = mean.reshape((1, 3, 1, 1))
        std = std.reshape((1, 3, 1, 1))
        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()

    def cuda(self, device=0):
        if self.net is not None:
            self.net.cuda(device)
            self.mean = self.mean.cuda(device)
            self.std = self.std.cuda(device)
            self.device = device


class PostProcBase:
    __metaclass__ = abc.ABCMeta

    def __init__(self, detection_args, postproc_args):
        self.detection_args = detection_args
        self.postproc_args = postproc_args
        self.device = postproc_args['device']
        self.only_infer = postproc_args['only_infer']

    @ abc.abstractmethod
    def process(self, output_dict, data_dict):
        pass


class RefineDetPostProcBase(PostProcBase):
    def __init__(self, detection_args, postproc_args):
        super(RefineDetPostProcBase, self).__init__(detection_args, postproc_args)
        self.n_classes = detection_args['n_classes']
        self.input_h = float(detection_args['input_h'])
        self.input_w = float(detection_args['input_w'])

        self.n_infer_rois = postproc_args['n_infer_rois']
        self.pos_anchor_threshold = postproc_args['pos_anchor_threshold']
        self.ignore_flags_refined_anchor = None

        # assert global_args['input_w'] == global_args['input_h']
        self.n_infer_rois = postproc_args['n_infer_rois']

        self.conf_thresh = postproc_args['conf_thresh']
        self.nms_thresh = postproc_args['nms_thresh']
        self.max_boxes = postproc_args['max_boxes']
        # add
        self.max_w = postproc_args['max_w']
        self.anchor_scale = postproc_args['anch_scale']
