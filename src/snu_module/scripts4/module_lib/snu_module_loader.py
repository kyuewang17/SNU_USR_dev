"""
SNU Integrated Module v5.0



"""
import module_detection as snu_det
import module_tracking_v4_5 as snu_trk
import module_action as snu_acl

from utils.profiling import Timer


class algorithms(object):
    def __init__(self):
        pass

    def __repr__(self):
        return "Algorithm"

    def __len__(self):
        raise NotImplementedError()


class snu_algorithms(algorithms):
    def __init__(self, frameworks, opts):
        super(snu_algorithms, self).__init__()

        # Load Options
        self.opts = opts

        # Load Detection Model
        self.det_framework = frameworks["det"]

        # Load Action Classification Model
        self.acl_framework = frameworks["acl"]

        # Initialize MOT Framework
        self.snu_mot = snu_trk.SNU_MOT(opts=opts)

        # Initialize Detections
        self.detections = {}

        # Initialize Frame Index
        self.fidx = None

        # Initialize Module FPS Dictionary
        self.fps = {
            "det": None,
            "trk": None,
            "acl": None,
        }

    def __repr__(self):
        return "SNU-Integrated-Algorithm"

    def __len__(self):
        return len(self.get_trajectories())

    # Detection Module
    def usr_object_detection(self, sync_data_dict):
        # Start Time
        det_fps_timer = Timer(convert="FPS")
        det_fps_timer.reset()

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

        self.detections = {"dets": dets, "confs": confs, "labels": labels}

        # End Time
        self.fps["det"] = det_fps_timer.elapsed

    # Multiple Target Tracking Module
    def usr_multiple_target_tracking(self, sync_data_dict):
        # Start Time
        trk_fps_timer = Timer(convert="FPS")
        trk_fps_timer.reset()

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
        self.fps["trk"] = trk_fps_timer.elapsed

    # Action Classification Module
    def usr_action_classification(self, sync_data_dict):
        # Start Time
        acl_fps_timer = Timer(convert="FPS")
        acl_fps_timer.reset()

        # Parse-out Required Sensor Modalities
        aclassify_sensor_data = {}
        for modal, modal_switch in self.opts.aclassifier.sensor_dict.items():
            if modal_switch is True:
                aclassify_sensor_data[modal] = sync_data_dict[modal]

        # Activate Module
        trks = snu_acl.aclassify(
            model=self.acl_framework,
            sync_data_dict=aclassify_sensor_data,
            trackers=self.get_trajectories(), opts=self.opts
        )
        self.snu_mot.trks = trks

        # End Time
        self.fps["acl"] = acl_fps_timer.elapsed

    def get_trajectories(self):
        return self.snu_mot.trks

    def get_detections(self):
        return self.detections

    def get_algorithm_fps(self):
        return self.fps

    # Call as Function
    def __call__(self, sync_data_dict, fidx):
        # Update Frame Index
        self.fidx = fidx

        # SNU Object Detector Module
        self.usr_object_detection(sync_data_dict=sync_data_dict)

        # SNU Multiple Target Tracker Module
        self.usr_multiple_target_tracking(sync_data_dict=sync_data_dict)

        # SNU Action Classification Module
        self.usr_action_classification(sync_data_dict=sync_data_dict)

        # NOTE: DO NOT USE PYTHON PRINT FUNCTION, USE "LOGGING" INSTEAD
        # NOTE: (2) Due to ROS, LOGGING Module Does not work properly!
        # trk_time = "Frame # (%08d) || DET fps:[%3.3f] | TRK fps:[%3.3f]" \
        #            % (self.fidx, 1/self.module_time_dict["det"], 1/self.module_time_dict["trk"])
        # print(trk_time)

        return self.get_trajectories(), self.get_detections(), self.get_algorithm_fps()


# Function for Testing on Synchronized Multimodal Image Sequence
def run_on_img_seq():
    import os
    from config import cfg

    # Set and Load Configuration File Path
    config_file_path = os.path.join(os.path.dirname(__file__), "config", "agents", "dynamic", "base.yaml")
    cfg.merge_from_file(config_file_path)

    # Set 




    pass






































if __name__ == "__main__":
    run_on_img_seq()
