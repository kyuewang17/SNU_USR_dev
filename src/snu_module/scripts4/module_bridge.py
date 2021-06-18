"""
SNU Integrated Module

    - Module Bridge for SNU Integrated Algorithms

"""
import numpy as np
import importlib
from utils.profiling import Timer

class algorithms(object):
    def __init__(self):
        pass

    def __repr__(self):
        return "Algorithm"

    def __len__(self):
        raise NotImplementedError()


class snu_algorithms(algorithms):
    def __init__(self, opts):
        super(snu_algorithms, self).__init__()

        # Load Options
        self.opts = opts

        # Module Import According to Development Version
        dev_version = str(opts.dev_version)
        dev_main_version, dev_sub_version = dev_version.split(".")[0], dev_version.split(".")[-1]
        self.snu_det = importlib.import_module(
            "module_lib.v{}_{}.DET".format(dev_main_version, dev_sub_version)
        )
        self.snu_trk = importlib.import_module(
            "module_lib.v{}_{}.TRK".format(dev_main_version, dev_sub_version)
        )
        self.snu_acl = importlib.import_module(
            "module_lib.v{}_{}.ACL".format(dev_main_version, dev_sub_version)
        )
        self.snu_seg = importlib.import_module(
            "module_lib.v{}_{}.SEG".format(dev_main_version, dev_sub_version)
        )
        self.snu_ATT = importlib.import_module(
            "module_lib.v{}_{}.ATT".format(dev_main_version, dev_sub_version)
        )

        # Segmentation
        self.seg_framework = self.snu_seg.load_model(opts=opts) if opts.segnet.run else None

        # Load Detection Model
        self.att_net = self.snu_ATT.load_model(opts=opts) if opts.attnet.run else None

        self.det_framework = self.snu_det.load_model(opts=opts)

        # Load Action Classification Model
        # self.acl_framework = self.snu_acl.load_model(opts=opts)
        self.acl_framework = self.snu_acl.load_models(opts=opts)

        # Initialize MOT Framework
        self.snu_mot = self.snu_trk.SNU_MOT(opts=opts)

        # Initialize Detections
        self.detections = {}

        # Initialize Heatmap
        self.heatmap = None

        # Initialize Frame Index
        self.fidx = None

        # Initialize Timer Objects
        self.seg_fpn_obj = Timer(convert="FPS")
        self.det_fps_obj = Timer(convert="FPS")
        self.trk_fps_obj = Timer(convert="FPS")
        self.acl_fps_obj = Timer(convert="FPS")

        # Initialize FPS Dictionary
        self.fps_dict = {
            "seg": None,
            "det": None,
            "trk": None,
            "acl": None
        }

    def __repr__(self):
        return "SNU-Integrated-Algorithm"

    def __len__(self):
        return len(self.get_trajectories())

    # Segmentation Module
    def osr_segmentation(self, sync_data_dict):
        # Start Time
        self.seg_fpn_obj.reset()

        if self.seg_framework is None:
            # End Time
            self.fps_dict["seg"] = -1
        else:
            # Activate Module
            heatmap = self.snu_seg.run(
                segnet=self.seg_framework,
                sync_data_dict=sync_data_dict,
                opts=self.opts,
            )
            self.heatmap = heatmap

            # End Time
            self.fps_dict["seg"] = self.seg_fpn_obj.elapsed

    def osr_object_detection(self, sync_data_dict):
        # Start Time
        self.det_fps_obj.reset()

        # Parse-out Required Sensor Modalities
        # TODO: Integrate this for all 3 modules
        detection_sensor_data = {}
        for modal, modal_switch in self.opts.detector.sensor_dict.items():
            if modal_switch is True:
                detection_sensor_data[modal] = sync_data_dict[modal]

        if self.att_net is not None:
            output = self.snu_ATT.run(attnet=self.att_net, sync_data_dict=detection_sensor_data, opts=self.opts)
            detection_sensor_data['att_tensor'] = output

        # Activate Module
        rgb_dets, thermal_dets = self.snu_det.detect(detector=self.det_framework, sync_data_dict=detection_sensor_data, opts=self.opts)

        # Color Detection Parsing
        rgb_confs, rgb_labels = rgb_dets[:, 4:5], rgb_dets[:, 5:6]
        rgb_dets = rgb_dets[:, 0:4]

        # Remove Too Small Detections
        keep_indices = []
        for det_idx, det in enumerate(rgb_dets):
            if det[2] * det[3] >= self.opts.detector.tiny_area_threshold:
                keep_indices.append(det_idx)
        rgb_dets = rgb_dets[keep_indices, :]
        rgb_confs = rgb_confs[keep_indices, :]
        rgb_labels = rgb_labels[keep_indices, :]

        if thermal_dets is not None:
            # Thermal Detection Parsing
            thermal_confs, thermal_labels = thermal_dets[:, 4:5], thermal_dets[:, 5:6]
            thermal_dets = thermal_dets[:, 0:4]

            # Remove Too Small Thermal Detections
            keep_indices = []
            for det_idx, det in enumerate(thermal_dets):
                if det[2] * det[3] >= self.opts.detector.tiny_area_threshold:
                    keep_indices.append(det_idx)
            thermal_dets = thermal_dets[keep_indices, :]
            thermal_confs = thermal_confs[keep_indices, :]
            thermal_labels = thermal_labels[keep_indices, :]
        else:
            # thermal_dets = np.array([], dtype=np.float32)
            # thermal_confs = np.array([], dtype=np.float32)
            # thermal_labels = np.array([], dtype=np.float32)
            thermal_dets = None
            thermal_confs = None
            thermal_labels = None

        self.detections = {"color": {"dets": rgb_dets, "confs": rgb_confs, "labels": rgb_labels},
                           "thermal": {"dets": thermal_dets, "confs": thermal_confs, "labels": thermal_labels}}

        # self.detections = {
        #     "color": {"dets": rgb_dets, "confs": rgb_confs, "labels": rgb_labels},
        # }

        # End Time
        self.fps_dict["det"] = self.det_fps_obj.elapsed

    # Multiple Target Tracking Module
    def osr_multiple_target_tracking(self, sync_data_dict):
        # Start Time
        self.trk_fps_obj.reset()

        # Parse-out Required Sensor Modalities
        tracking_sensor_data = {}
        for modal, modal_switch in self.opts.tracker.sensor_dict.items():
            if modal_switch is True:
                tracking_sensor_data[modal] = sync_data_dict[modal]

        # Activate Module
        self.snu_mot(
            sync_data_dict=sync_data_dict, fidx=self.fidx, detections=self.detections
        )

        # End Time
        self.fps_dict["trk"] = self.trk_fps_obj.elapsed

    # Action Classification Module
    def osr_action_classification(self, sync_data_dict):
        # Start Time
        self.acl_fps_obj.reset()

        # Parse-out Required Sensor Modalities
        aclassify_sensor_data = {}
        for modal, modal_switch in self.opts.aclassifier.sensor_dict.items():
            if modal_switch is True:
                aclassify_sensor_data[modal] = sync_data_dict[modal]

        # Activate Module
        trks = self.snu_acl.aclassify(
            models=self.acl_framework,
            sync_data_dict=aclassify_sensor_data,
            trajectories=self.get_trajectories(), opts=self.opts
        )
        self.snu_mot.trks = trks

        # End Time
        self.fps_dict["acl"] = self.acl_fps_obj.elapsed

    def get_heatmap(self):
        return self.heatmap

    def get_trajectories(self):
        return self.snu_mot.trks

    def get_detections(self):
        return self.detections

    def get_algorithm_fps(self):
        return self.fps_dict

    # Call as Function
    def __call__(self, sync_data_dict, fidx):
        # Update Frame Index
        self.fidx = fidx

        # SNU Segmentation Module
        self.osr_segmentation(sync_data_dict=sync_data_dict)

        # SNU Object Detector Module
        self.osr_object_detection(sync_data_dict=sync_data_dict)

        # SNU Multiple Target Tracker Module
        self.osr_multiple_target_tracking(sync_data_dict=sync_data_dict)

        # SNU Action Classification Module
        self.osr_action_classification(sync_data_dict=sync_data_dict)

        # NOTE: DO NOT USE PYTHON PRINT FUNCTION, USE "LOGGING" INSTEAD
        # NOTE: (2) Due to ROS, LOGGING Module Does not work properly!
        # trk_time = "Frame # (%08d) || DET fps:[%3.3f] | TRK fps:[%3.3f]" \
        #            % (self.fidx, 1/self.module_time_dict["det"], 1/self.module_time_dict["trk"])
        # print(trk_time)

        return self.get_trajectories(), self.get_detections(), self.get_heatmap(), self.get_algorithm_fps()


if __name__ == "__main__":
    pass
