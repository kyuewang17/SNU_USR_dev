"""
SNU Integrated Module v3.0
  - Code that converts Bounding Box Format
  * bbox ==>> {Left-top Right-bottom}
  ** zx ==>> {Center(u,v) du, dv, width, height}
  *** zx3 ==>> {Center(u,v) Depth(D), du, dv, w, h}

"""
# Import Module
import numpy as np
import torch


# Convert bbox to z(or x)
def _bbox_to_zx(bbox, velocity=None):
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    u = bbox[0]+w/2
    v = bbox[1]+h/2
    if velocity is None:
        return np.array([u, v, w, h]).reshape(4, 1)
    else:
        return np.array([u, v, velocity[0], velocity[1], w, h]).reshape(6, 1)


# Convert bbox to z(or x)
def bbox_to_zx(bbox, velocity=None, depth=None):
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    u = bbox[0]+w/2
    v = bbox[1]+h/2
    if velocity is None:
        if depth is None:
            return np.array([u, v, w, h]).reshape(4, 1)
        else:
            return np.array([u, v, depth, w, h]).reshape(5, 1)
    else:
        if depth is None:
            return np.array([u, v, velocity[0], velocity[1], w, h]).reshape(6, 1)
        else:
            return np.array([u, v, depth, velocity[0], velocity[1], w, h]).reshape(7, 1)


# Re-size BBOX
def resize_bbox(bbox, x_ratio, y_ratio):
    # Get Points
    a, b, c, d = bbox[0], bbox[1], bbox[2], bbox[3]

    # Set Coefficients
    K_alpha_plus, K_alpha_minus = 0.5*(1+x_ratio), 0.5*(1-x_ratio)
    K_beta_plus, K_beta_minus = 0.5*(1+y_ratio), 0.5*(1-y_ratio)

    # Convert
    _a = K_alpha_plus*a + K_alpha_minus*c
    _b = K_beta_plus*b + K_beta_minus*d
    _c = K_alpha_minus*a + K_alpha_plus*c
    _d = K_beta_minus*b + K_beta_plus*d

    return np.array([_a, _b, _c, _d])


# Convert x3(z3) to x2(z2)
def zx3_to_zx2(zx):
    return np.delete(zx, 2)


# Get Diagonal Length of bbox
def get_diagonal_of_bbox(bbox):
    lt, rb = bbox[0:2], bbox[2:4]
    return np.linalg.norm(rb-lt)


def zxs_to_bboxes(bboxes, is_torch=False):
    u = bboxes[:, 0:1]
    v = bboxes[:, 1:2]
    half_w = bboxes[:, 2:3] / 2
    half_h = bboxes[:, 3:4] / 2
    l = u - half_w
    t = v - half_h
    r = u + half_w
    b = v + half_h
    if is_torch is True:
        return torch.cat([l, t, r, b], dim=1)
    else:
        return np.concatenate([l, t, r, b], axis=1)


# Convert z(or x) to bbox
def zx_to_bbox(zx):
    zx = zx.reshape(len(zx))
    if len(zx) == 6:
        u1 = zx[0] - zx[4] / 2
        v1 = zx[1] - zx[5] / 2
        u2 = zx[0] + zx[4] / 2
        v2 = zx[1] + zx[5] / 2
    elif len(zx) == 7:
        u1 = zx[0] - zx[5] / 2
        v1 = zx[1] - zx[6] / 2
        u2 = zx[0] + zx[5] / 2
        v2 = zx[1] + zx[6] / 2
    else:
        assert 0, "Check for input numpy array dimension!"

    bbox = np.array([u1, v1, u2, v2]).reshape(4)
    velocity = np.array([zx[2], zx[3]]).reshape(2)
    return bbox, velocity


# Get Intersection BBOX Coordinates
def intersection_bbox(bbox1, bbox2):
    uu1 = np.maximum(bbox1[0], bbox2[0])
    vv1 = np.maximum(bbox1[1], bbox2[1])
    uu2 = np.minimum(bbox1[2], bbox2[2])
    vv2 = np.minimum(bbox1[3], bbox2[3])
    return np.array([uu1, vv1, (uu2 - uu1), (vv2 - vv1)]).reshape(4)


# Get Intersection-over-Union between two BBOXES
def iou(bbox1, bbox2):
    common_bbox = intersection_bbox(bbox1, bbox2)
    w = np.maximum(0., common_bbox[2])
    h = np.maximum(0., common_bbox[3])
    intersection_area = w * h
    union_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - intersection_area
    if union_area == 0:
        ret_iou = 0
    else:
        ret_iou = float(intersection_area) / float(union_area)

    return ret_iou


if __name__ == '__main__':
    b1 = [0, 0, 100, 100]
    b2 = [101, 101, 200, 200]
    c1 = intersection_bbox(b1, b2)

    x = [10, 10, 100, 100]
    kk = bbox_to_zx(bbox=x, velocity=np.zeros(2), depth=1)


    pass



















