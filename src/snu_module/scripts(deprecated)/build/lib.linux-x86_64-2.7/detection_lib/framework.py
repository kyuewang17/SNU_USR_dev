import torch
from .__base__ import FrameworkBase


class OneStageFramework(FrameworkBase):
    def __init__(self, global_args, network_dict, postproc_dict):
        super(OneStageFramework, self).__init__(
            global_args, network_dict, postproc_dict)

        self.backbone = network_dict['backbone']
        self.detector = network_dict['detector']
        self.det_postproc = postproc_dict['detector']

    def forward(self, data_dict, train=False):
        self.backbone.train(train)
        self.detector.train(train)
        torch.autograd.set_grad_enabled(False)

        with torch.cuda.device(self.backbone.device):
            img = data_dict['img'].requires_grad_(False).float().cuda()
            fmap_dict = self.backbone.forward(img)

        with torch.cuda.device(self.detector.device):
            for key, fmap in fmap_dict.items():
                fmap_dict[key] = fmap.cuda(self.detector.device)
            output_dict = self.detector.forward(fmap_dict)

            result_dict = self.det_postproc.process(output_dict, data_dict)
            return output_dict, result_dict
