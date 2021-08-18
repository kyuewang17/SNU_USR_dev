"""
SNU Integrated Module v5.0
    - Multimodal Multiple Target Tracking
    - Final Version

"""
import cv2
import numpy as np

from module_lib.v4_5._TRK.objects.trajectory import LATENT_TRAJECTORY, TRAJECTORY, TRAJECTORIES
from module_lib.v4_5._TRK.objects import bbox


class SNU_MOT(object):
    def __init__(self, opts):
        # Load Options
        self.opts = opts

        # Initialize Trajectories
        self.trks = TRAJECTORIES()

        # Initialize Latent Trajectories
        self.latent_trks = []

        # Initialize Maximum Historical Trajectory ID
        self.max_trk_id = None

        # Initialize Frame Index
        self.fidx = None

    def __len__(self):
        return len(self.trks)

    def __repr__(self):
        return "SNU_MOT"

    def destroy_latent_trajectories(self):
        raise NotImplementedError()

    def destroy_trajectories(self):
        raise NotImplementedError()

    def generate_new_trajectories(self):
        raise NotImplementedError()

    def __call__(self, sync_data_dict, fidx, detections):
        # Get Detection Results
        if self.opts.agent_type == "static":
            modal = "color"
            trk_detections = detections[modal]
        else:
            if self.opts.time == "day":
                modal = "color"
                trk_detections = detections[modal]
            else:
                modal = "thermal"
                trk_detections = detections[modal]

        # Load Point-Cloud XYZ Data
        if sync_data_dict["lidar"] is not None:
            sync_data_dict["lidar"].load_pc_xyz_data()

        # Initialize New Trajectory Variable
        new_trks = []

        # Destroy Latent Trajectories and Trajectories
        self.destroy_latent_trajectories()
        self.destroy_trajectories()

        # Associate Detections with Trajectories
        residual_detections = self.trks.update(
            frame=sync_data_dict[modal], lidar_obj=sync_data_dict["lidar"], detections=trk_detections,
            fidx=fidx, cost_thresh=self.opts.tracker.association["trk"]["cost_thresh"]
        )

        # Update Latent Trajectories w.r.t. Residual Detections
        if len(self.latent_trks) == 0:
            for det_idx, det in enumerate(trk_detections["dets"]):
                # Wrap Detection Numpy Array into BBOX Object
                det_bbox = bbox.BBOX(
                    bbox_format="LTRB", lt_x=det[0], lt_y=det[1], rb_x=det[2], rb_y=det[3]
                )
                det_conf, label = trk_detections["confs"][det_idx], trk_detections["labels"][det_idx]

                # Initialize Latent Trajectory
                latent_trk = LATENT_TRAJECTORY(
                    frame=sync_data_dict[modal], modal=modal,
                    det_bbox=det_bbox, det_conf=det_conf, label=label
                )

                # Append
                self.latent_trks.append(latent_trk)
                del latent_trk
        else:
            # TODO: Update Latent Trajectories, returning residual detections as in TRAJECTORY Class
            # TODO: for the residual detections, generate new latent trajectories as above
            pass






















if __name__ == "__main__":
    pass
