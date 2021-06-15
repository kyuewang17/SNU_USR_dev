"""
SNU Integrated Module v5.0
    - Action Classification

"""
import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn as nn
from model_thermal_4th_year_tobedeleted import resnet18_th_4th
from model_rgb_4th_year_tobedeleted import resnet18_rb_4th


def load_model(opts):
    # Set CUDA Device
    cuda_device_str = "cuda:" + str(opts.aclassifier.device)
    device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")

    # Model Directory
    model_dir = os.path.join(opts.aclassifier.model_dir, opts.time)

    # Find Model
    pth_file_list = []
    for f in os.listdir(model_dir):
        if f.endswith(".pth"):
            pth_file_list.append(f)
    if len(pth_file_list) > 1:
        raise AssertionError("Too many pth files...!")
    elif len(pth_file_list) == 0:
        raise AssertionError("No pth files...!")
    model_path = os.path.join(model_dir, pth_file_list[0])

    # Get Model
    if opts.time == "day":
        model = resnet18_rb_4th(num_classes=3)
    elif opts.time == "night":
        model = resnet18_th_4th(num_classes=3)
    else:
        raise AssertionError()
    model.load_state_dict(torch.load(model_path))

    # Load Model to GPU device
    model = model.to(device)
    return model


def voting(pose_list):
    counter = 0
    num = pose_list[0]
    for i in pose_list:
        curr_freq = pose_list.count(i)
        if curr_freq > counter:
            counter = curr_freq
            num = i

    return num + 1


def aclassify(model, sync_data_dict, trajectories, opts):
    # Get Time
    acl_time = opts.time

    # Get Frame
    if acl_time == "day":
        frame = sync_data_dict["color"].get_data()
    else:
        frame = sync_data_dict["thermal"].get_data() / 65535.0
    H, W = frame.shape[0], frame.shape[1]

    # Set CUDA Device
    cuda_device_str = "cuda:" + str(opts.aclassifier.device)
    device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")

    # Classify Action for Human Label
    for trk_idx, trk in enumerate(trajectories):
        if trk.label != 1:
            trk.pose = None
        else:
            # Get Trajectory State
            x3 = trk.x3.reshape(len(trk.x3))

            # Get BBOX Points
            a = max(0, int(x3[1] - (x3[6] / 2)))
            b = min(int(x3[1] + (x3[6] / 2)), H - 1)
            c = max(0, int(x3[0] - (x3[5] / 2)))
            d = min(int(x3[0] + (x3[5] / 2)), W - 1)

            if (a >= b) or (c >= d):
                trk.pose_list.insert(0, 0)
            else:
                ratio = torch.tensor([[(d - c) / float(b - a)]]).to(device)
                crop_image = frame[a:b, c:d]
                cr_rsz_img = cv2.resize(crop_image, dsize=(60, 60), interpolation=cv2.INTER_AREA)

                if acl_time == "day":
                    b, g, r = np.split(cr_rsz_img, 3, axis=2)
                    d = np.array([r, g, b]).squeeze() / 255.0
                    e = d.tolist()
                    x = torch.tensor(e, dtype=torch.float)
                    x = transforms.Normalize([0.35, 0.30, 0.27], [0.19, 0.18, 0.17])(x)
                    x = (x.view(1, 3, 60, 60)).to(device)
                    tmp = model(x, ratio)
                else:
                    e = cr_rsz_img.tolist()
                    z = torch.tensor(e, dtype=torch.float).view(1, 60, 60)
                    x = transforms.Normalize([0.677], [0.172])(z)
                    x = (x.view(1, 1, 60, 60)).to(device)
                    tmp = model(x, ratio)
                    tmp = tmp + torch.tensor([0.7, 0.1, 0.0]).to(device)

                trk.pose_list.insert(0, torch.max(tmp, 1)[1])

            # Pose List has a maximum window size
            if len(trk.pose_list) > 5:
                trk.pose_list.pop()
            trk.pose = voting(trk.pose_list)
            trajectories[trk_idx] = trk

    return trajectories


# Test Action Classification (for snu-osr-pil computer)
def test_aclassifier_snu_osr_pil():
    pass


if __name__ == "__main__":
    pass
