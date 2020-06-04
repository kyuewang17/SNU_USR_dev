import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

import snu_utils.bounding_box as fbbox

# Rescue Option for Night Version in IITP Final demo
# import rescue.force_thermal_align_iitp_final_night as rgb_t_align


def load_model(opts):
    cuda_device_str = "cuda:" + str(opts.aclassifier.device)
    device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")

    model = torch.load(opts.aclassifier.model_dir)
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

    return num+1


def aclassify(model, imgStruct_dict, trackers, opts):
    # Get RGB Image
    color_img = imgStruct_dict['rgb'].frame.raw

    cuda_device_str = "cuda:" + str(opts.aclassifier.device)
    device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")

    H = color_img.shape[0]
    W = color_img.shape[1]
    for tracker_idx, tracker in enumerate(trackers):

        # Do only for Person
        if tracker.label == 1:

            a = max(0, int(tracker.x[1]-(tracker.x[5]/2)))
            b = min(int(tracker.x[1]+(tracker.x[5]/2)), H-1)
            c = max(0, int(tracker.x[0]-(tracker.x[4]/2)))
            d = min(int(tracker.x[0]+(tracker.x[4]/2)), W-1)

            if (a >= b) or (c >= d):
                tracker.pose_list.insert(0, 0)
            else:
                crop_image = color_img[a:b, c:d]
                cr_rsz_img = cv2.resize(crop_image, dsize=(60, 60), interpolation=cv2.INTER_AREA)
                b, g, r = np.split(cr_rsz_img, 3, axis=2)
                d = np.array([r, g, b]).squeeze() / 255.0
                e = d.tolist()
                x = torch.tensor(e, dtype=torch.float)
                x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
                x = (x.view(1, 3, 60, 60)).to(device)
                tmp = model(x)
                tracker.pose_list.insert(0, torch.max(tmp, 1)[1])

            trackers[tracker_idx] = tracker

            if len(tracker.pose_list) > 5:
                tracker.pose_list.pop()

            tracker.pose = voting(tracker.pose_list)
        else:
            tracker.pose = []

    return trackers


def aclassify_night(model, imgStruct_dict, trackers, opts):
    # Get RGB and Thermal Image
    rgb_img = imgStruct_dict['rgb'].frame.raw
    thermal_img = imgStruct_dict['thermal'].frame.raw

    cuda_device_str = "cuda:" + str(opts.aclassifier.device)
    device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")

    H = thermal_img.shape[0]
    W = thermal_img.shape[1]
    for tracker_idx, tracker in enumerate(trackers):

        # Do only for Person
        if tracker.label == 1:

            a = max(0, int(tracker.x_bbox_thermal_aclassify[1]))
            b = min(H-1, int(tracker.x_bbox_thermal_aclassify[3]))
            c = max(0, int(tracker.x_bbox_thermal_aclassify[0]))
            d = min(W-1, int(tracker.x_bbox_thermal_aclassify[2]))

            # a = max(0, int(tracker.x[1]-(tracker.x[5]/2)))
            # b = min(int(tracker.x[1]+(tracker.x[5]/2)), H-1)
            # c = max(0, int(tracker.x[0]-(tracker.x[4]/2)))
            # d = min(int(tracker.x[0]+(tracker.x[4]/2)), W-1)

            ######################################
            # RGB->Thermal Coordinate Conversion #
            ######################################

            if (a >= b) or (c >= d):
                tracker.pose_list.insert(0, 0)
            else:
                crop_image = thermal_img[a:b, c:d]
                img_thermal_rs = cv2.resize(crop_image, dsize=(60, 60), interpolation=cv2.INTER_AREA)

                d = img_thermal_rs / 255.0
                d = d.reshape(60, 60, 1)
                x = transforms.ToTensor()(d)
                x = transforms.Normalize([0.677], [0.172])(x)
                x = (x.view(1, 1, 60, 60)).to(device)  #
                tmp = model(x)
                tracker.pose_list.insert(0, torch.max(tmp, 1)[1])

            trackers[tracker_idx] = tracker

            if len(tracker.pose_list) > 5:
                tracker.pose_list.pop()

            tracker.pose = voting(tracker.pose_list)
        else:
            tracker.pose = []

    return trackers
