"""
SNU Integrated Module v3.0
    - Action Classification

"""
import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn as nn
from model_thermal_4th_year_tobedeleted import resnet18_th_4th
from model_thermal_3rd_year_tobedeleted import resnet18_th_3rd
from model_rgb_4th_year_tobedeleted import resnet18_rb_4th

def load_model(opts):
    # Set CUDA Device
    cuda_device_str = "cuda:" + str(opts.aclassifier.device)
    device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")

    # Find Model
    pth_file_list = []
    for f in os.listdir(opts.aclassifier.model_dir):
        if f.endswith(".pth") or f.endswith(".pt"):
            pth_file_list.append(f)
    pth_file_list.sort()
    if opts.aclassifier.time == 'day':
        if len(pth_file_list) == 1:
            model_path = os.path.join(opts.aclassifier.model_dir, pth_file_list[0])
        else:
            assert 0, "Error"

    else :
        if opts.aclassifier.test_which_year == '3':
            model_path = os.path.join(opts.aclassifier.model_dir, pth_file_list[0])
        else :
            model_path = os.path.join(opts.aclassifier.model_dir, pth_file_list[1])

    # Get Model
    if opts.aclassifier.time == 'day' :
        model = resnet18_rb_4th(num_classes=3)
        model.load_state_dict(torch.load(model_path))
    else:
        if opts.aclassifier.test_which_year == '4' :
            model = resnet18_th_4th(num_classes=3)
            model.load_state_dict(torch.load(model_path))
        else :
            model = resnet18_th_3rd(num_classes=3)
            model.load_state_dict(torch.load(model_path))
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


# Action Classification
def aclassify(model, sync_data_dict, trackers, opts):
    if opts.aclassifier.time == "day":
        # Get Color Frame
        color_frame = sync_data_dict["color"].get_data()
        H, W = color_frame.shape[0], color_frame.shape[1]

        # Set CUDA Device
        cuda_device_str = "cuda:" + str(opts.aclassifier.device)
        device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")

        # Classify Action for Human Label
        for tracker_idx, tracker in enumerate(trackers):
            if tracker.label == 1:
                # Get Tracklet State
                x3 = tracker.x3.reshape(len(tracker.x3))

                # Get BBOX Points
                a = max(0, int(x3[1] - (x3[6] / 2)))
                b = min(int(x3[1] + (x3[6] / 2)), H - 1)
                c = max(0, int(x3[0] - (x3[5] / 2)))
                d = min(int(x3[0] + (x3[5] / 2)), W - 1)

                if (a >= b) or (c >= d):
                    tracker.pose_list.insert(0, 0)
                else:
                    ratio = torch.tensor([[(d-c) / float(b-a)]]).to(device)
                    crop_image = color_frame[a:b, c:d]
                    cr_rsz_img = cv2.resize(crop_image, dsize=(60, 60), interpolation=cv2.INTER_AREA)
                    b, g, r = np.split(cr_rsz_img, 3, axis=2)
                    d = np.array([r, g, b]).squeeze() / 255.0
                    e = d.tolist()
                    x = torch.tensor(e, dtype=torch.float)
                    x = transforms.Normalize([0.35, 0.30, 0.27], [0.19, 0.18, 0.17])(x)
                    x = (x.view(1, 3, 60, 60)).to(device)
                    tmp = model(x, ratio)
                    tracker.pose_list.insert(0, torch.max(tmp, 1)[1])

                # Pose List has a maximum window size
                if len(tracker.pose_list) > 5:
                    tracker.pose_list.pop()
                tracker.pose = voting(tracker.pose_list)

                trackers[tracker_idx] = tracker

            # For non-human,
            else:
                tracker.pose = None
    else :
        if opts.aclassifier.test_which_year == '4':
            th_frame = sync_data_dict["thermal"].get_data() /65535.0
            H, W = th_frame.shape[0], th_frame.shape[1]

            # Set CUDA Device
            cuda_device_str = "cuda:" + str(opts.aclassifier.device)
            device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")

            # Classify Action for Human Label
            for tracker_idx, tracker in enumerate(trackers):
                if tracker.label == 1:
                    # Get Tracklet State
                    x3 = tracker.x3.reshape(len(tracker.x3))

                    # Get BBOX Points
                    a = max(0, int(x3[1] - (x3[6] / 2)))
                    b = min(int(x3[1] + (x3[6] / 2)), H - 1)
                    c = max(0, int(x3[0] - (x3[5] / 2)))
                    d = min(int(x3[0] + (x3[5] / 2)), W - 1)

                    if (a >= b) or (c >= d):
                        tracker.pose_list.insert(0, 0)
                    else:
                        ratio = torch.tensor([[(d - c) / float(b - a)]]).to(device)
                        crop_image = th_frame[a:b, c:d]
                        cr_rsz_img = cv2.resize(crop_image, dsize=(60, 60), interpolation=cv2.INTER_AREA)
                        e = cr_rsz_img.tolist()
                        z = torch.tensor(e, dtype=torch.float).view(1, 60, 60)

                        x = transforms.Normalize([0.677], [0.172])(z)
                        x = (x.view(1, 1, 60, 60)).to(device)
                        tmp = model(x, ratio)
                        tmp = tmp + torch.tensor([0.7, 0.1, 0.0]).to(device)
                        tracker.pose_list.insert(0, torch.max(tmp, 1)[1])

                    # Pose List has a maximum window size
                    if len(tracker.pose_list) > 5:
                        tracker.pose_list.pop()
                    tracker.pose = voting(tracker.pose_list)

                    trackers[tracker_idx] = tracker

                # For non-human,
                else:
                    tracker.pose = None
        else :
            if opts.aclassifier.test_which_year == '4':
                th_frame = sync_data_dict["thermal"].get_data() / 65535.0
                H, W = th_frame.shape[0], th_frame.shape[1]

                # Set CUDA Device
                cuda_device_str = "cuda:" + str(opts.aclassifier.device)
                device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")

                # Classify Action for Human Label
                for tracker_idx, tracker in enumerate(trackers):
                    if tracker.label == 1:
                        # Get Tracklet State
                        x3 = tracker.x3.reshape(len(tracker.x3))

                        # Get BBOX Points
                        a = max(0, int(x3[1] - (x3[6] / 2)))
                        b = min(int(x3[1] + (x3[6] / 2)), H - 1)
                        c = max(0, int(x3[0] - (x3[5] / 2)))
                        d = min(int(x3[0] + (x3[5] / 2)), W - 1)

                        if (a >= b) or (c >= d):
                            tracker.pose_list.insert(0, 0)
                        else:
                            ratio = torch.tensor([[(d - c) / float(b - a)]]).to(device)
                            crop_image = th_frame[a:b, c:d]
                            cr_rsz_img = cv2.resize(crop_image, dsize=(60, 60), interpolation=cv2.INTER_AREA)
                            e = cr_rsz_img.tolist()
                            z = torch.tensor(e, dtype=torch.float).view(1, 60, 60)

                            x = transforms.Normalize([0.677], [0.172])(z)
                            x = (x.view(1, 1, 60, 60)).to(device)
                            tmp = model(x, ratio)
                            tmp = tmp + torch.tensor([0.5, 0.1, 0.0]).to(device)
                            tracker.pose_list.insert(0, torch.max(tmp, 1)[1])

                        # Pose List has a maximum window size
                        if len(tracker.pose_list) > 5:
                            tracker.pose_list.pop()
                        tracker.pose = voting(tracker.pose_list)

                        trackers[tracker_idx] = tracker

                    # For non-human,
                    else:
                        tracker.pose = None
            else :
                th_frame = sync_data_dict["thermal"].get_data() / 65535.0
                H, W = th_frame.shape[0], th_frame.shape[1]

                # Set CUDA Device
                cuda_device_str = "cuda:" + str(opts.aclassifier.device)
                device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")

                # Classify Action for Human Label
                for tracker_idx, tracker in enumerate(trackers):
                    if tracker.label == 1:
                        # Get Tracklet State
                        x3 = tracker.x3.reshape(len(tracker.x3))

                        # Get BBOX Points
                        a = max(0, int(x3[1] - (x3[6] / 2)))
                        b = min(int(x3[1] + (x3[6] / 2)), H - 1)
                        c = max(0, int(x3[0] - (x3[5] / 2)))
                        d = min(int(x3[0] + (x3[5] / 2)), W - 1)

                        if (a >= b) or (c >= d):
                            tracker.pose_list.insert(0, 0)
                        else:
                            crop_image = th_frame[a:b, c:d]
                            cr_rsz_img = cv2.resize(crop_image, dsize=(60, 60), interpolation=cv2.INTER_AREA)
                            e = cr_rsz_img.tolist()
                            z = torch.tensor(e, dtype=torch.float).view(1, 60, 60)

                            x = transforms.Normalize([0.677], [0.172])(z)
                            x = (x.view(1, 1, 60, 60)).to(device)
                            tmp = model(x)
                            tmp = tmp + torch.tensor([0.0, 1.0, 0.0]).to(device)
                            tracker.pose_list.insert(0, torch.max(tmp, 1)[1])

                        # Pose List has a maximum window size
                        if len(tracker.pose_list) > 5:
                            tracker.pose_list.pop()
                        tracker.pose = voting(tracker.pose_list)

                        trackers[tracker_idx] = tracker

                    # For non-human,
                    else:
                        tracker.pose = None


    # Get Color Frame
    # color_frame = sync_data_dict["color"].get_data()
    # H, W = color_frame.shape[0], color_frame.shape[1]
    #
    # # Set CUDA Device
    # cuda_device_str = "cuda:" + str(opts.aclassifier.device)
    # device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")
    #
    # # Classify Action for Human Label
    # for tracker_idx, tracker in enumerate(trackers):
    #     if tracker.label == 1:
    #         # Get Tracklet State
    #         x3 = tracker.x3.reshape(len(tracker.x3))
    #
    #         # Get BBOX Points
    #         a = max(0, int(x3[1] - (x3[6] / 2)))
    #         b = min(int(x3[1] + (x3[6] / 2)), H - 1)
    #         c = max(0, int(x3[0] - (x3[5] / 2)))
    #         d = min(int(x3[0] + (x3[5] / 2)), W - 1)
    #
    #         if (a >= b) or (c >= d):
    #             tracker.pose_list.insert(0, 0)
    #         else:
    #             ratio = torch.tensor([[(d-c) / float(b-a)]]).to(device)
    #             crop_image = color_frame[a:b, c:d]
    #             cr_rsz_img = cv2.resize(crop_image, dsize=(60, 60), interpolation=cv2.INTER_AREA)
    #             b, g, r = np.split(cr_rsz_img, 3, axis=2)
    #             d = np.array([r, g, b]).squeeze() / 255.0
    #             e = d.tolist()
    #             x = torch.tensor(e, dtype=torch.float)
    #             x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
    #             x = (x.view(1, 3, 60, 60)).to(device)
    #             tmp = model(x, ratio)
    #             tracker.pose_list.insert(0, torch.max(tmp, 1)[1])
    #
    #         # Pose List has a maximum window size
    #         if len(tracker.pose_list) > 5:
    #             tracker.pose_list.pop()
    #         tracker.pose = voting(tracker.pose_list)
    #
    #         trackers[tracker_idx] = tracker
    #
    #     # For non-human,
    #     else:
    #         tracker.pose = None
    # # Get Color Frame
    # color_frame = sync_data_dict["color"].get_data()
    # H, W = color_frame.shape[0], color_frame.shape[1]
    #
    # # Set CUDA Device
    # cuda_device_str = "cuda:" + str(opts.aclassifier.device)
    # device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")
    #
    # # Classify Action for Human Label
    # for tracker_idx, tracker in enumerate(trackers):
    #     if tracker.label == 1:
    #         # Get Tracklet State
    #         x3 = tracker.x3.reshape(len(tracker.x3))
    #
    #         # Get BBOX Points
    #         a = max(0, int(x3[1] - (x3[6] / 2)))
    #         b = min(int(x3[1] + (x3[6] / 2)), H - 1)
    #         c = max(0, int(x3[0] - (x3[5] / 2)))
    #         d = min(int(x3[0] + (x3[5] / 2)), W - 1)
    #
    #         if (a >= b) or (c >= d):
    #             tracker.pose_list.insert(0, 0)
    #         else:
    #             ratio = torch.tensor([[(d-c) / float(b-a)]]).to(device)
    #             crop_image = color_frame[a:b, c:d]
    #             cr_rsz_img = cv2.resize(crop_image, dsize=(60, 60), interpolation=cv2.INTER_AREA)
    #             b, g, r = np.split(cr_rsz_img, 3, axis=2)
    #             d = np.array([r, g, b]).squeeze() / 255.0
    #             e = d.tolist()
    #             x = torch.tensor(e, dtype=torch.float)
    #             x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
    #             x = (x.view(1, 3, 60, 60)).to(device)
    #             tmp = model(x, ratio)
    #             tracker.pose_list.insert(0, torch.max(tmp, 1)[1])
    #
    #         # Pose List has a maximum window size
    #         if len(tracker.pose_list) > 5:
    #             tracker.pose_list.pop()
    #         tracker.pose = voting(tracker.pose_list)
    #
    #         trackers[tracker_idx] = tracker
    #
    #     # For non-human,
    #     else:
    #         tracker.pose = None
    return trackers


# Test Action Classification (for snu-osr-pil computer)
def test_aclassifier_snu_osr_pil():
    pass


if __name__ == "__main__":
    pass
