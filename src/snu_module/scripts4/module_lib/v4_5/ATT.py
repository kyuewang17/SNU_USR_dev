import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2

class Input_Att(nn.Module):
    def __init__(self):
        super(Input_Att, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, rgb, nv1):
        x = torch.cat([rgb, nv1], dim=1)
        x = self.attention(x)
        return x

    def load(self, load_path):
        net_dict = torch.load(load_path, map_location='cpu')
        net_dict = {k.partition('attention.')[2]: net_dict[k] for k in net_dict.keys()}
        self.attention.load_state_dict(net_dict)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
preprocess_to_tensor = transforms.Compose([
    transforms.ToTensor(),
])

def load_model(opts, is_default_device=True):
    torch.autograd.set_grad_enabled(False)
    device = 0 if is_default_device else opts.attnet.device
    attnet = Input_Att()
    attnet.load(opts.attnet.model_dir+'/input_att.pth')

    attnet.eval()
    attnet.cuda(device)
    return attnet


def run(attnet, sync_data_dict, opts):
    color_frame = (sync_data_dict["color"].get_data() if "color" in sync_data_dict.keys() else None)
    nv_frame = (sync_data_dict["nightvision"].get_data() if "nightvision" in sync_data_dict.keys() else None)

    nv_frame = cv2.resize(nv_frame[87:-98, 155:-165, :], dsize=(color_frame.shape[1], color_frame.shape[0]))
    nv_frame = (preprocess_to_tensor(nv_frame)/255.0).unsqueeze(0).cuda(opts.attnet.device)
    img = preprocess(color_frame).unsqueeze(0).cuda(opts.attnet.device)
    
    output = attnet(img, nv_frame)
    img *= output
    return img

