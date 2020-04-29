import torch
import torch.nn.functional as func
from .__base__ import RefineDetPostProcBase
from . import __util__ as util


class RefineDetPostProc(RefineDetPostProcBase):
    def process(self, output_dict, data_dict):
        refine_anchors, ignore_flags_refined_anchor = self.__refine_arm_anchors__(output_dict, self.anchors)

        # r1, r2, r3 : [#batch * tensor(shape=(self.max_boxes,4))],
        # [#batch * tensor(shape=(self.max_boxes,1))], [#batch * tensor(shape=(self.max_boxes,1))]
        r1, r2, r3 = self.detect([output_dict['odm_loc'], output_dict['odm_conf']],
                                 refine_anchors, ignore_flags_refined_anchor)

        result_dict = dict()
        result_dict['boxes_l'] = r1
        result_dict['confs_l'] = r2
        result_dict['labels_l'] = r3
        return result_dict

    def __refine_arm_anchors__(self, output_dict, anchors):
        """
        Refine anchores and get ignore flag for refined anchores using outputs of ARM.
        """
        arm_loc = output_dict['arm_loc']
        arm_conf = output_dict['arm_conf']
        # Detach softmax of confidece predictions to block backpropation.
        arm_score = func.softmax(arm_conf.detach(), -1)
        # Adjust anchors with arm_loc.
        # The refined_pirors is better to be considered as predicted ROIs,
        # like Faster RCNN in a sence.
        refined_anchors = refine_anchors(arm_loc.data, anchors.data, self.anchor_scale)
        ignore_flags_refined_anchor = arm_score[:, :, 1] < self.pos_anchor_threshold
        return refined_anchors, ignore_flags_refined_anchor

    # using ignore flag, ruturn -> box, conf, label
    def detect(self, odm_predictions, refined_anchors, ignore_flags_refined_anchor):
        """
        :param odm_predictions:
            0).odm_loc_data: (tensor) location predictions from loc layers of ODM
            Shape: (batch_size, num_anchors, 4)
            1).odm_conf_data: (tensor) confidence predictions from conf layers of ODM
            Shape: (batch_size, num_anchors, num_classes)
        :param refined_anchors: (batch_size, num_anchors, 4)
        :param ignore_flags_refined_anchor: (batch_size, num_anchors),
            1 means an igored negative anchor, otherwise reserved.

        """
        _boxes_l = list()
        _confs_l = list()
        _labels_l = list()
        loc_data = odm_predictions[0].data
        score_data = func.softmax(odm_predictions[1].detach(), dim=-1).data
        # Output
        num = refined_anchors.size(0)
        # select
        # For each image, keep keep_top_k,
        # retain top_k per class for nms.
        for idx in range(num):
            # Decoded odm bbox prediction to get boxes
            all_boxes = decode(loc_data[idx], refined_anchors[idx], self.anchor_scale)
            # Ignore predictions whose positive scores are small.
            # pdb.set_trace()
            flag = ignore_flags_refined_anchor[idx].data < 1
            box_flag = flag.unsqueeze(flag.dim()).expand_as(all_boxes)
            conf_flag = flag.unsqueeze(flag.dim()).expand_as(score_data[idx])

            boxes_s = all_boxes[box_flag].view(-1, 4)
            cls_boxes_s = boxes_s.unsqueeze(0).repeat(self.n_classes - 1, 1, 1)
            cls_confs_s = score_data[idx][conf_flag].view(-1, self.n_classes).transpose(1, 0)[1:]

            boxes_s, confs_s, labels_s = nms_cls_boxes_s(
                cls_boxes_s, cls_confs_s, self.n_classes - 1,
                self.conf_thresh, self.nms_thresh)
            boxes_s = boxes_s * self.img_size
            labels_s += 1

            # if len(boxes_s[0].size()) == 1:
            #     boxes_s[0] = torch.zeros((1, 4)).cuda()
            #     confs_s[1] = torch.zeros((1, 1)).cuda()
            #     labels_s[2] = torch.zeros((1, 1)).cuda()

            _boxes_l.append(boxes_s[:self.max_boxes])
            _confs_l.append(confs_s[:self.max_boxes])
            _labels_l.append(labels_s[:self.max_boxes])
        return _boxes_l, _confs_l, _labels_l


def refine_anchors(loc_pred, anchors, variance):
    """
    Refine location of anchors with location predicts.
    :param loc_pred: (batch_size, num_anchors, 4),
        (norm_cx, norm_cy, norm_w, norm_h)
    :param anchors: (num_anchors, 4), (cx, cy, w, h)
    :param variance: (var_xy, var_wh)
    :return: refined_anchors (batch_size, num_anchors, 4),
        (cx, cy, w, h)
    """
    num = loc_pred.size(0)
    num_anchors = anchors.size(0)

    assert loc_pred.size(1) == num_anchors, 'anchors'

    refined_anchors = torch.Tensor(num, num_anchors, 4).type_as(anchors)
    for ind in range(num):
        cur_loc = loc_pred[ind]
        # cur_loc (norm_dx, norm_dy, norm_w, norm_h)
        # anchors(cx, cy, w, h)
        # boxes (x1, y1, x2, y2)
        boxes = decode(cur_loc, anchors, variance)
        ori_boxes = boxes.clone()
        # (cx, cy, x2, y2)
        # pdb.set_trace()
        boxes[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        # (cx, cy, w, h)
        boxes[:, 2:] = (boxes[:, 2:] - boxes[:, :2]) * 2.0
        refined_anchors[ind] = boxes
        # nan_flags = (boxes != boxes)
        # if len(nan_flags.nonzero()) > 0:
        #    pdb.set_trace()
        # a = 0
    return refined_anchors


def decode(loc, anchors, variances):
    """Decode locations from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_anchors,4]
        anchors (tensor): Prior boxes in center-offset form.
            Shape: [num_anchors,4].
        variances: (list[float]) Variances of anchorboxes
    Return:
        decoded bounding box predictions
        (xmin, ymin, xmax, ymax)
    """

    boxes = torch.cat((
        anchors[:, :2] + loc[:, :2] * variances[0] * anchors[:, 2:],
        anchors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms_cls_boxes_s(boxes_s, confs_s, n_classes, conf_threshold, nms_threshold):
    # boxes_cs.shape: (n_classes, n_boxes, 4)
    # confs_cs.shape: (n_classes, n_boxes)
    cls_boxes_sl = list()
    cls_confs_sl = list()
    cls_labels_sl = list()

    for c in range(n_classes):
        # boxes_css.shape: (n_boxes, 4)
        # confs_css.shape: (n_boxes)
        cls_boxes_sc = boxes_s[c]
        cls_confs_sc = confs_s[c]
        # print(cls_boxes_sc.shape, cls_confs_sc.shape)
        if len(cls_boxes_sc) == 0:
            continue

        keep_idxes = torch.nonzero(cls_confs_sc > conf_threshold).view(-1)
        cls_boxes_sc = cls_boxes_sc[keep_idxes]
        cls_confs_sc = cls_confs_sc[keep_idxes]
        # print(cls_boxes_sc.shape, cls_confs_sc.shape)
        if keep_idxes.shape[0] == 0:
            continue

        if nms_threshold <= 0.0:
            cls_boxes_sc, cls_confs_sc = util.sort_boxes_s(cls_boxes_sc, cls_confs_sc)
            cls_boxes_sc, cls_confs_sc = cls_boxes_sc[:1], cls_confs_sc[:1]

        elif nms_threshold > 1.0:
            pass

        else:
            # print(cls_boxes_sc.shape, cls_confs_sc.shape)
            keep_idxes = util.nms(cls_boxes_sc, cls_confs_sc, nms_threshold)
            keep_idxes = keep_idxes.long().view(-1)
            cls_boxes_sc = cls_boxes_sc[keep_idxes]
            cls_confs_sc = cls_confs_sc[keep_idxes].unsqueeze(dim=1)

        labels_css = torch.zeros(cls_confs_sc.shape).float().cuda()
        labels_css += c

        cls_boxes_sl.append(cls_boxes_sc)
        cls_confs_sl.append(cls_confs_sc)
        cls_labels_sl.append(labels_css)

    if len(cls_boxes_sl) > 0:
        boxes_s = torch.cat(cls_boxes_sl, dim=0)
        confs_s = torch.cat(cls_confs_sl, dim=0)
        labels_s = torch.cat(cls_labels_sl, dim=0)
    else:
        boxes_s = torch.zeros((1, 4)).float().cuda()
        confs_s = torch.zeros((1, 1)).float().cuda()
        labels_s = torch.zeros((1, 1)).float().cuda()

    # cls_boxes_s, cls_confs_s, cls_labels_s = cls_boxes_s[:30], cls_confs_s[:30], cls_labels_s[:30]
    # print(cls_boxes_s.shape, cls_confs_s.shape)
    # print(torch.cat([cls_boxes_s, cls_confs_s, cls_labels_s], dim=1))
    boxes_s, confs_s, labels_s = \
        util.sort_boxes_s(boxes_s, confs_s, labels_s)
    # print(torch.cat([cls_boxes_s, cls_confs_s, cls_labels_s], dim=1), '\n')
    return boxes_s, confs_s, labels_s
