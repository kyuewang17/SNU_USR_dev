"""
SNU Integrated Module v3.0
    - Action Classification

"""
import cv2
import numpy as np
import torch
from torchvision import models, transforms


def load_model(opts):
    # Set CUDA Device
    cuda_device_str = "cuda:" + str(opts.aclassifier.device)
    device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")

    # Get Model
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

    return num + 1


# Action Classification
def aclassify(model, sync_data_dict, trackers, opts):
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
                crop_image = color_frame[a:b, c:d]
                cr_rsz_img = cv2.resize(crop_image, dsize=(60, 60), interpolation=cv2.INTER_AREA)
                b, g, r = np.split(cr_rsz_img, 3, axis=2)
                d = np.array([r, g, b]).squeeze() / 255.0
                e = d.tolist()
                x = torch.tensor(e, dtype=torch.float)
                x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
                x = (x.view(1, 3, 60, 60)).to(device)
                tmp = model(x)
                tracker.pose_list.insert(0, torch.max(tmp, 1)[1])

            # Pose List has a maximum window size
            if len(tracker.pose_list) > 5:
                tracker.pose_list.pop()
            tracker.pose = voting(tracker.pose_list)

            trackers[tracker_idx] = tracker

        # For non-human,
        else:
            tracker.pose = None

    return trackers


if __name__ == "__main__":
    pass
