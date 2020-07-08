"""
SNU Integrated Module v4.0



"""
import numpy as np
import datetime

import module_detection as snu_det
import module_tracking_v4 as snu_trk
import module_action as snu_acl


class snu_algorithms(object):
    def __init__(self, frameworks, opts):
        # Load Options
        self.opts = opts

        # Load Detection Model
        self.det_framework = frameworks["det"]

        # Load Action Classification Model
        self.acl_framework = frameworks["acl"]

        # Initialize Tracklet and Tracklet Candidates
        self.snu_mot = snu_trk.SNU_MOT(opts=opts)

        # Initialize Detections
        self.detections = {}

        # Initialize Frame Index
        self.fidx = None

        # Initialize Module Time Dictionary
        self.module_time_dict = {
            "det": 0.0,
            "trk": 0.0,
            "acl": 0.0,
        }

    # Detection Module
    def usr_object_detection(self, sync_data_dict, logger):
        # Start Time
        START_TIME = datetime.datetime.now()

        # Parse-out Required Sensor Modalities
        # TODO: Integrate this for all 3 modules
        detection_sensor_data = {}
        for modal, modal_switch in self.opts.detector.sensor_dict.items():
            if modal_switch is True:
                detection_sensor_data[modal] = sync_data_dict[modal]

        # Activate Module
        dets = snu_det.detect(
            detector=self.det_framework, sync_data_dict=detection_sensor_data,
            opts=self.opts
        )
        confs, labels = dets[:, 4:5], dets[:, 5:6]
        dets = dets[:, 0:4]

        # Remove Too Small Detections
        keep_indices = []
        for det_idx, det in enumerate(dets):
            if det[2] * det[3] >= self.opts.detector.tiny_area_threshold:
                keep_indices.append(det_idx)
        dets = dets[keep_indices, :]
        confs = confs[keep_indices, :]
        labels = labels[keep_indices, :]

        # Stop Time
        END_TIME = datetime.datetime.now()

        self.detections = {"dets": dets, "confs": confs, "labels": labels}
        self.module_time_dict["det"] = (END_TIME - START_TIME).total_seconds()

    # Multiple Target Tracking Module
    def usr_multiple_target_tracking(self, sync_data_dict, logger):
        # Start Time
        START_TIME = datetime.datetime.now()

        # Parse-out Required Sensor Modalities
        tracking_sensor_data = {}
        for modal, modal_switch in self.opts.tracker.sensor_dict.items():
            if modal_switch is True:
                tracking_sensor_data[modal] = sync_data_dict[modal]

        # Activate Module
        self.snu_mot(
            sync_data_dict=sync_data_dict, fidx=self.fidx,
            detections=self.detections
        )

        # Stop Time
        END_TIME = datetime.datetime.now()

        self.module_time_dict["trk"] = (END_TIME - START_TIME).total_seconds()

    # Action Classification Module
    def usr_action_classification(self, sync_data_dict, logger):
        START_TIME = datetime.datetime.now()

        # Parse-out Required Sensor Modalities
        aclassify_sensor_data = {}
        for modal, modal_switch in self.opts.aclassifier.sensor_dict.items():
            if modal_switch is True:
                aclassify_sensor_data[modal] = sync_data_dict[modal]

        # Activate Module
        trks = snu_acl.aclassify(
            model=self.acl_framework,
            sync_data_dict=aclassify_sensor_data,
            trackers=self.snu_mot.trks, opts=self.opts
        )
        self.snu_mot.trks = trks

        END_TIME = datetime.datetime.now()

        self.module_time_dict["acl"] = (END_TIME - START_TIME).total_seconds()

    # Call as Function
    def __call__(self, sync_data_dict, logger, fidx):
        # Update Frame Index
        self.fidx = fidx

        # SNU Object Detector Module
        self.usr_object_detection(
            sync_data_dict=sync_data_dict, logger=logger
        )

        # SNU Multiple Target Tracker Module
        self.usr_multiple_target_tracking(
            sync_data_dict=sync_data_dict, logger=logger
        )

        # SNU Action Classification Module
        self.usr_action_classification(
            sync_data_dict=sync_data_dict, logger=logger
        )

        # NOTE: DO NOT USE PYTHON PRINT FUNCTION, USE "LOGGING" INSTEAD
        # NOTE: (2) Due to ROS, LOGGING Module Does not work properly!
        # trk_time = "Frame # (%08d) || DET fps:[%3.3f] | TRK fps:[%3.3f]" \
        #            % (self.fidx, 1/self.module_time_dict["det"], 1/self.module_time_dict["trk"])
        # print(trk_time)

        return self.snu_mot.trks, self.detections, self.module_time_dict
