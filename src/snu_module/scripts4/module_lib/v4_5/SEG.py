import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(opts, is_default_device=True):
    device = 0 if is_default_device else opts.segnet.device
    # segnet = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    segnet = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    segnet.eval()
    segnet.cuda(device)
    return segnet


def run(segnet, sync_data_dict, opts):
    torch.autograd.set_grad_enabled(False)
    color_frame = (sync_data_dict["color"].get_data() if "color" in sync_data_dict.keys() else None)
    input_size = (opts.segnet.segmentation_args["input_w"], opts.segnet.segmentation_args["input_h"])

    img_size = color_frame.shape[:2]
    img = cv2.resize(color_frame, dsize=input_size)
    img = preprocess(img).unsqueeze(0).cuda(opts.segnet.device)

    output = segnet(img)["out"][0]
    heatmap = output[opts.segnet.segnet_args["classes"]].argmax(0)
    heatmap = (heatmap != 0).int()
    # heatmap = heatmap.unsqueeze(2).repeat(1,1,3)
    heatmap = cv2.resize((heatmap * 255).detach().cpu().numpy().astype(np.uint8), dsize=(img_size[1], img_size[0]))
    # heatmap = (torch.sum(output[opts.segnet.segnet_args["classes"]], dim=0)).long()
    # heatmap = cv2.resize((heatmap * 10).detach().cpu().numpy(), dsize=img_size)
    return heatmap
