import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# Resnet module from [torchvision-0.3.0]
import utils.resnet as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("/home/kyle/USR_SNU_MODULE/SNU_Integrated_v2/src/snu_module/model/model_img60_res18_1_full_copy_6.pth")
model = model.to(device)


def voting(pose_list):
    counter = 0
    num = pose_list[0]
    for i in pose_list :
        curr_freq = pose_list.count(i)
        if (curr_freq > counter) :
            counter = curr_freq
            num = i
    return num


def res_clf(color_img, trackers):
    H = color_img.shape[0]
    W = color_img.shape[1]
    for tracker_idx, tracker in enumerate(trackers):

        a = max(0,int(tracker.x[1]-(tracker.x[5]/2)))
        b = min(int(tracker.x[1]+(tracker.x[5]/2)), H-1)
        c = max(0,int(tracker.x[0]-(tracker.x[4]/2)))
        d = min(int(tracker.x[0]+(tracker.x[4]/2)), W-1)

        if (a>=b) or (c>=d):
            tracker.poselist.insert(0, 0)
        else :
            crop_image = color_img[a:b, c:d]
            cr_rsz_img = cv2.resize(crop_image, dsize=(60, 60), interpolation=cv2.INTER_AREA)
            b, g, r = np.split((cr_rsz_img), 3, axis=2)
            d = np.array([r, g, b]).squeeze() / 255.0
            e = d.tolist()
            x = torch.tensor(e, dtype=torch.float)
            x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
            x = (x.view(1, 3, 60, 60)).to(device)
            print(model(x))
            print(torch.max(model(x),1))
            tracker.poselist.insert(0, torch.max(model(x), 1).indices)
        trackers[tracker_idx] = tracker

        if len(tracker.poselist) > 5:
            tracker.poselist.pop()
        tracker.pose = voting(tracker.poselist)

    return trackers