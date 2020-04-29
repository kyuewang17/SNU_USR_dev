import torch
from . import _C

nms = _C.nms


def sort_boxes_s(boxes_s, confs_s, labels_s=None):
    sorted_confs_s, sorted_idxs = torch.sort(confs_s, dim=0, descending=True)
    # print('\t\t', sorted_idxs.shape)
    if len(sorted_idxs.shape) == 2:
        sorted_idxs = torch.squeeze(sorted_idxs, dim=1)
    # print('\t\t', sorted_idxs.shape)
    sorted_boxes_s = boxes_s[sorted_idxs]
    if labels_s is None:
        return sorted_boxes_s, sorted_confs_s
    else:
        sorted_labels_s = labels_s[sorted_idxs]
        return sorted_boxes_s, sorted_confs_s, sorted_labels_s


def cvt_torch2numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = tensor.detach().numpy()
    elif isinstance(tensor, list) or isinstance(tensor, tuple):
        for i in range(len(tensor)):
            tensor[i] = cvt_torch2numpy(tensor[i])
    # else:
    #     print(type(tensor))
    #     print(tensor)
    return tensor
