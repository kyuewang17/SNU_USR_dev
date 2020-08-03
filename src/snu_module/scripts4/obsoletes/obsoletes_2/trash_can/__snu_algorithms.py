"""
SNU Integrated Module v3.0
    - Actual Code that contains SNU key Implementations

    - SNU Algorithms
        - Object Detection
        - Multiple Target Tracking
        - Action Classification

    - Object Detection
        - INFO INFO INFO

    - Multiple Target Tracking
        - INFO INFO INFO

    - Action Classification
        - INFO INFO INFO
"""

import datetime
import cv2
import logging

# Import SNU Algorithm Modules
import module_detection as snu_det
# import module_tracking as snu_trk
import module_action as snu_acl

# Import SNU Visualizer Module
import snu_visualizer as snu_vis


# Unmanned Surveillance Robot Object Detector
def usr_object_detection(sync_data_dict, framework, opts):
    # # Get Detector Framework
    # framework = snu_det.load_model(opts)

    # Start Time for Detector
    DET_START_TIME = datetime.datetime.now()

    # Parse-out Required Sensor Modalities
    detection_sensor_data = {}
    for modal, modal_switch in opts.detector.sensor_dict.items():
        if modal_switch is True:
            detection_sensor_data[modal] = sync_data_dict[modal]

    # Activate Detector Module
    # TODO: by implementing above TODO, this also needs to be altered
    # TODO: Condition it w.r.t. "agent_type" in opts
    dets = snu_det.detect(
        framework=framework, sync_data_dict=detection_sensor_data,
        opts=opts
    )
    confs, labels = dets[:, 4:5], dets[:, 5:6]
    dets = dets[:, 0:4]

    # Remove Too Small Detections
    keep_indices = []
    for det_idx, det in enumerate(dets):
        if det[2] * det[3] >= opts.detector.tiny_area_threshold:
            keep_indices.append(det_idx)
    dets = dets[keep_indices, :]
    confs = confs[keep_indices, :]
    labels = labels[keep_indices, :]

    # Stop Time
    DET_STOP_TIME = datetime.datetime.now()

    # Detection Result Dictionary
    detections = {"dets": dets, "confs": confs, "labels": labels}

    return detections, (DET_STOP_TIME-DET_START_TIME).total_seconds()


# Unmanned Surveillance Robot Multiple Target Tracker
def usr_multiple_target_tracking(sync_data_dict, fidx, multiple_objs, detections, opts):
    """
    NOTE: Keep thinking about how to accommodate maximum id of tracklets
          (essential for giving new tracklet id)
    """
    # Start Time for Multiple Target Tracker
    TRK_START_TIME = datetime.datetime.now()

    # Parse-out Required Sensor Modalities
    tracking_sensor_data = {}
    for modal, modal_switch in opts.tracker.sensor_dict.items():
        if modal_switch is True:
            tracking_sensor_data[modal] = sync_data_dict[modal]

    # Activate Multiple Target Tracker Module
    # TODO: by implementing above TODO, this also needs to be altered
    multiple_objs(
        sync_data_dict=tracking_sensor_data, fidx=fidx,
        detections=detections, is_default_device=True
    )

    # Stop Time
    TRK_STOP_TIME = datetime.datetime.now()

    return multiple_objs, (TRK_STOP_TIME-TRK_START_TIME).total_seconds()

    # return [], [], []


# Unmanned Surveillance Robot Action Classifier
def usr_action_classification(sync_data_dict, framework, tracklets, opts):
    # # Get Action Classifier Framework
    # framework = snu_acl.load_model(opts)
    #
    # # Start Time for Action Classifier
    # ACL_START_TIME = datetime.datetime.now()
    #
    # # Parse-out Required Sensor Modalities
    # aclassify_sensor_data = {}
    # for modal, modal_switch in opts.aclassifier.sensor_dict.items():
    #     if modal_switch is True:
    #         aclassify_sensor_data[modal] = sync_data_dict[modal]
    #
    # # Activate Action Classifier Module
    # # TODO: by implementing above TODO, this also needs to be altered
    # # TODO: Condition it w.r.t. "agent_type" in opts
    # tracklets = snu_acl.aclassify()
    #
    # # Stop Time
    # ACL_STOP_TIME = datetime.datetime.now()
    #
    # return tracklets, (ACL_STOP_TIME-ACL_START_TIME).total_seconds()

    return [], []


# Integrated SNU Algorithm for Unmanned Surveillance Robot
def usr_integrated_snu(fidx, sync_data_dict, frameworks, multiple_objs, opts):
    # SNU Object Detector Module
    detections, DET_TIME = usr_object_detection(
        sync_data_dict=sync_data_dict, framework=frameworks["det"], opts=opts
    )

    # SNU Multiple Target Tracker Module
    multiple_objs, TRK_TIME = usr_multiple_target_tracking(
        sync_data_dict=sync_data_dict, fidx=fidx,
        multiple_objs=multiple_objs, detections=detections, opts=opts
    )

    # # SNU Action Classification Module
    # tracklets, ACL_TIME = usr_action_classification(
    #     sync_data_dict=sync_data_dict, framework=frameworks["acl"],
    #     tracklets=tracklets, opts=opts
    # )
    ACL_TIME = None

    # Pack Algorithm Times
    algorithm_time_dict = {"det": DET_TIME, "trk": TRK_TIME, "acl": ACL_TIME}

    return multiple_objs, detections, algorithm_time_dict


if __name__ == "__main__":
    pass
