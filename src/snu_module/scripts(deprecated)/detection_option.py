# input_img_path = './voc07-0000005.png'
# detection_model_dir = './result/refinedet320x320-wdecay0.0005/snapshot/iter-00160000'

# 2019.12 day (threaml = False), night (thermal = True) ----------------------------------------------------------------
# thermal = False
thermal = True

detection_args = {
    'n_classes': 3,
    'input_h': 320, 'input_w': 448,
}

backbone_args = {
    'name': 'res34level4',
    'pretrained': False
}

detector_args = {
    'is_bnorm': False, 'tcb_ch': 256,
    'fmap_chs': [128, 256, 512, 128],
    'fmap_sizes': [40, 20, 10, 5], 'fmap_steps': [8, 16, 32, 64],
    'anch_scale': [0.1, 0.2], 'anch_min_sizes': [32, 64, 128, 256],
    'anch_max_sizes': [], 'anch_aspect_ratios': [[2], [2], [2], [2]],
    'n_boxes_list': [3, 3, 3, 3], 'is_anch_clip': True
}

postproc_args = {
    'n_infer_rois': 300, 'device': 0, 'only_infer': True,
    'nms_thresh': 0.45, 'conf_thresh': 0.4,
    'max_boxes': 200, 'pos_anchor_threshold' : 0.01,
    'anch_scale': [0.1, 0.2]
}

# default setting ------------------------------------------------------------------------------------------------------
# detection_model_dir = './snu_module/model/detection_model'
# device = 0
#
# detection_args = {
#     'n_classes': 3,
#     'input_h': 320, 'input_w': 320,
# }
#
# backbone_args = {
#     'pretrained': False
# }
#
# detector_args = {
#     'is_bnorm': False, 'tcb_ch': 256,
#     'fmap_chs': [128, 256, 512, 128],
#     'fmap_sizes': [40, 20, 10, 5], 'fmap_steps': [8, 16, 32, 64],
#     'anch_scale': [0.1, 0.2], 'anch_min_sizes': [32, 64, 128, 256],
#     'anch_max_sizes': [], 'anch_aspect_ratios': [[2], [2], [2], [2]],
#     'n_boxes_list': [3, 3, 3, 3], 'is_anch_clip': True
# }
#
# postproc_args = {
#     'n_infer_rois': 300, 'device': 1, 'only_infer': True,
#     # 'conf_thresh' ==>> Classification(2nd threshold)
#     'nms_thresh': 0.45, 'conf_thresh': 0.3,
#     # 'pos_anchor_threshold ==>> Objectness(1st threshold)
#     'max_boxes': 200, 'pos_anchor_threshold': 0.01,
#     'anch_scale': [0.1, 0.2]
# }
