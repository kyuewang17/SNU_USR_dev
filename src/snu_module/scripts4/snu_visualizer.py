"""
SNU Integrated Module v5.0
    - Visualization Class for SNU Results
"""
# Import Modules
import cv2
import numpy as np
import copy
import os

# Import Custom Modules
import utils.bounding_box as fbbox


# Define Base Visualization Object
class vis_obj(object):
    def __init__(self, vopts):
        # BBOX Variable
        # -> bbox format: [left-top (x,y) right-bottom (x,y)]
        #    (N x 4), numpy array
        self.bboxes = None

        # Set Visualization Options
        self.vopts = vopts

    # Update Objects
    def update_objects(self, *args, **kwargs):
        raise NotImplementedError

    # Draw Objects on Selected Modal Frame
    def draw_objects(self, *args, **kwargs):
        raise NotImplementedError

# Define Detection Result Visualization Object
class vis_seg_obj(vis_obj):
    def __init__(self, vopts):
        """
        vopts: visualizer_options
        """
        super(vis_seg_obj, self).__init__(vopts)

        # Initialize Mask
        self.mask = None

    # Update Detection BBOX Objects
    def update_objects(self, result_segmentation):
        self.mask = result_segmentation

    # Draw Detection Result on Selected Modal Frame
    def draw_objects(self):
        opencv_winname = "segmentation"
        return self.mask, opencv_winname


# Define Detection Result Visualization Object
class vis_det_obj(vis_obj):
    def __init__(self, vopts):
        """
        vopts: visualizer_options
        """
        super(vis_det_obj, self).__init__(vopts)

        # Custom Color and Object Linewidth Settings
        self.color_code = vopts.detection["bbox_color"]
        self.linewidth = vopts.detection["linewidth"]

        # Initialize Detection Result Variable
        self.detections = {}

    # Update Detection Result
    def update_objects(self, result_detections):
        self.detections = result_detections

    # Draw Detection Result on Selected Modal Frame
    def draw_objects(self, sync_data_dict, **kwargs):
        # Get Draw Modals
        vis_frames = {}
        opencv_winnames = {}
        for modal in self.detections.keys():
            if modal not in vis_frames.keys():
                # Get Visualization Frame
                vis_frames[modal] = copy.deepcopy(sync_data_dict[modal].get_data(astype=np.uint8))

                # Set OpenCV Window Names
                opencv_winnames[modal] = "detection ({})".format(modal)

        # Draw Detections
        for modal, modal_detection in self.detections.items():
            # Get Modal Visualization Frame
            modal_vis_frame = vis_frames[modal]

            if self.detections[modal]["dets"] is not None:
                # Get Detections of Current Modal
                modal_dets = self.detections[modal]["dets"].astype(np.int32)

                # Iterate for Detections
                for det_bbox in modal_dets:
                    # Draw Rectangle BBOX
                    cv2.rectangle(
                        modal_vis_frame,
                        (det_bbox[0], det_bbox[1]), (det_bbox[2], det_bbox[3]),
                        self.color_code, self.linewidth
                    )

            # Append Draw Frame
            vis_frames[modal] = modal_vis_frame

        return vis_frames, opencv_winnames


# Define Trajectory (TRK & ACL) Result Visualization Object
class vis_trk_acl_obj(vis_obj):
    def __init__(self, vopts):
        """
        vopts: visualizer_options
        """
        super(vis_trk_acl_obj, self).__init__(vopts)

        # Delete BBOX Attribute
        delattr(self, "bboxes")

        # Initialize Trajectory (List for Trajectory Class)
        self.trks = []

        # Object Linewidth Settings
        self.linewidth = vopts.tracking["linewidth"]

    # Update Trajectory Object
    def update_objects(self, trajectories):
        self.trks = trajectories

    # Draw Tracking Result on Selected Modal Frame
    def draw_objects(self, sync_data_dict, **kwargs):
        # Get Detection Modalities
        modals = kwargs.get("det_modals")
        assert modals is not None

        # Initialize Modal-wise Trajectories
        modalwise_trks = {}
        for modal in modals:
            modalwise_trks[modal] = []

        # Get Modal-wise Trajectories
        for trk in self.trks:
            for modal in modals:
                if trk.modal == modal:
                    modalwise_trks[modal].append(trk)

        # Set Visualization Frames and OpenCV Window Names
        vis_frames = {}
        opencv_winnames = {}
        for modal in modals:
            if modal not in vis_frames.keys():
                # Get Modal Visualization Frame
                vis_frames[modal] = copy.deepcopy(sync_data_dict[modal].get_data(astype=np.uint8))

                # Set OpenCV Window Names
                if self.vopts.aclassifier["is_draw"] is False:
                    opencv_winnames[modal] = "tracking ({})".format(modal)
                else:
                    opencv_winnames[modal] = "tracking | action ({})".format(modal)

        # Destroy Modal Key if Empty
        destroy_modals = []
        for modal, trk in modalwise_trks.items():
            if len(trk) == 0:
                destroy_modals.append(modal)
        for destroy_modal in destroy_modals:
            modalwise_trks.pop(destroy_modal, None)

        # Draw Trajectories
        for modal in modals:
            # Get Visualization Frame
            modal_vis_frame = vis_frames[modal]

            # If modal is not destroyed,
            if modal not in destroy_modals:

                # Get Trajectories
                trks = modalwise_trks[modal]

                for trk in trks:
                    # Convert State BBOX coordinate type and precision
                    state_bbox, _ = fbbox.zx_to_bbox(trk.states[-1])
                    state_bbox = state_bbox.astype(np.int32)

                    # Draw Rectangle BBOX
                    cv2.rectangle(
                        modal_vis_frame,
                        (state_bbox[0], state_bbox[1]), (state_bbox[2], state_bbox[3]),
                        (trk.color[0], trk.color[1], trk.color[2]), self.linewidth
                    )

                    # Unpack Visualization Options
                    font = self.vopts.font
                    font_size = self.vopts.font_size
                    pad_pixels = self.vopts.pad_pixels
                    info_interval = self.vopts.info_interval

                    # Visualize Trajectory ID
                    trk_id_str = "id:" + str(trk.id) + ""
                    (tw, th) = cv2.getTextSize(trk_id_str, font, fontScale=font_size, thickness=2)[0]
                    text_x = int((state_bbox[0] + state_bbox[2]) / 2.0 - tw / 2.0)
                    text_y = int(state_bbox[1] + th)
                    box_coords = ((int(text_x - pad_pixels / 2.0), int(text_y - th - pad_pixels / 2.0)),
                                  (int(text_x + tw + pad_pixels / 2.0), int(text_y + pad_pixels / 2.0)))
                    cv2.rectangle(modal_vis_frame, box_coords[0], box_coords[1], (trk.color[0], trk.color[1], trk.color[2]), cv2.FILLED)
                    cv2.putText(
                        modal_vis_frame, trk_id_str, (text_x, text_y), font, font_size,
                        (255 - trk.color[0], 255 - trk.color[1], 255 - trk.color[2]), thickness=2
                    )

                    # Visualize Trajectory Depth
                    if trk.depth is not None:
                        trk_depth_str = "d=" + str(round(trk.x3[2], 3)) + "(m)"
                        (tw, th) = cv2.getTextSize(trk_depth_str, font, fontScale=1.2, thickness=2)[0]
                        text_x = int((state_bbox[0] + state_bbox[2]) / 2.0 - tw / 2.0)
                        text_y = int((state_bbox[1] + state_bbox[3]) / 2.0 - th / 2.0)

                        # Put Depth Text (Tentative)
                        cv2.putText(modal_vis_frame, trk_depth_str, (text_x, text_y), font, 1.2,
                                    (255 - trk.color[0], 255 - trk.color[1], 255 - trk.color[2]), thickness=2)

                    # Visualize Action Classification Result
                    if trk.pose is not None and self.vopts.aclassifier["is_draw"] is True:
                        # Draw Action Results only when Human
                        # TODO: (later, handle this directly on action classification module)
                        if trk.label == 1:
                            # Get Image Height and Width
                            H, W = modal_vis_frame.shape[0], modal_vis_frame.shape[1]

                            # Convert Action Classification Result Text in Words
                            if trk.pose == 1:
                                pose_word = "Lie"
                            elif trk.pose == 2:
                                pose_word = "Sit"
                            elif trk.pose == 3:
                                pose_word = "Stand"
                            else:
                                pose_word = "NAN"

                            # Put Result Text in the frame
                            cv2.putText(modal_vis_frame, pose_word,
                                        (min(int(trk.x3[0] + (trk.x3[5] / 2)), W - 1), min(int(trk.x3[1] + (trk.x3[6] / 2)), H - 1)),
                                        font, 1.5, (255 - trk.color[0], 255 - trk.color[1], 255 - trk.color[2]), thickness=2)

            vis_frames[modal] = modal_vis_frame

        return vis_frames, opencv_winnames


# Define Visualizer Class
class visualizer(object):
    def __init__(self, opts):
        # Get Visualizer Option for SNU options
        self.vopts = opts.visualization

        # Initialize Visualization Objects
        self.SEG_VIS_OBJ = vis_seg_obj(vopts=self.vopts)
        self.DET_VIS_OBJ = vis_det_obj(vopts=self.vopts)
        self.TRK_ACL_VIS_OBJ = vis_trk_acl_obj(vopts=self.vopts)

        # Screen Geometry Imshow Position
        self.screen_imshow_x = opts.screen_imshow_x
        self.screen_imshow_y = opts.screen_imshow_y
        self.det_window_moved, self.trk_acl_window_moved = False, False
        self.general_window_moved = False

        # Top-view Map (in mm)
        if self.vopts.top_view["is_draw"] is True:
            self.top_view_map = np.ones(
                shape=[opts.visualization.top_view["map_size"][0], opts.visualization.top_view["map_size"][1], 3],
                dtype=np.uint8
            )
            self.top_view_window_moved = False
        else:
            self.top_view_map = None
            self.top_view_window_moved = None

        # Top-view Max Axis
        self.u_max_axis, self.v_max_axis = 0, 0
        self.u_min_axis, self.v_min_axis = 0, 0

    # Save Image Frame
    @staticmethod
    def save_frame(save_base_dir, frame, fidx):
        assert os.path.isdir(save_base_dir)
        modal_save_dir = save_base_dir
        if os.path.isdir(modal_save_dir) is False:
            os.makedirs(modal_save_dir)

        frame_filename = "{:08d}".format(fidx)
        frame_filepath = os.path.join(modal_save_dir, frame_filename + ".png")
        if os.path.isfile(frame_filepath) is True:
            frame_filename += "_X_"
            frame_filepath = os.path.join(modal_save_dir, frame_filename + ".png")

        # OpenCV Save Image
        if frame.shape[-1] == 3:
            cv2.imwrite(frame_filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(frame_filepath, frame)

    # Visualize Image Sequences Only
    def visualize_modal_frames(self, sensor_data, **kwargs):
        # Get Modal Frame
        vis_frame = sensor_data.get_data()

        if vis_frame is not None:
            # Precision Conversion
            precision = kwargs.get("precision")
            if precision is not None:
                vis_frame = vis_frame.astype(precision)

            # Get Modal Type Name
            modal_type = "{}".format(sensor_data)

            # OpenCV Window Name
            winname = "[{}]".format(modal_type)

            # Make NamedWindow
            cv2.namedWindow(winname)

            if self.general_window_moved is False:
                if self.screen_imshow_x is not None and self.screen_imshow_y is not None:
                    cv2.moveWindow(winname, self.screen_imshow_x, self.screen_imshow_y)
                    self.general_window_moved = True

            # IMSHOW
            if modal_type.__contains__("color") is True:
                cv2.imshow(winname, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            else:
                cv2.imshow(winname, vis_frame)

            cv2.waitKey(1)

    # Visualize Image Sequence with Point Cloud Drawn On
    def visualize_modal_frames_with_calibrated_pointcloud(self, sensor_data, pc_img_coord, color):
        # Get Modal Frame
        vis_frame = sensor_data.get_data()

        if vis_frame is not None:
            # Get Modal Type Name
            modal_type = "{}".format(sensor_data)

            # OpenCV Window Name
            winname = "{} + {}".format(modal_type, "Rectified-PC")

            # Make NamedWindow
            cv2.namedWindow(winname)

            if self.general_window_moved is False:
                if self.screen_imshow_x is not None and self.screen_imshow_y is not None:
                    cv2.moveWindow(winname, self.screen_imshow_x, self.screen_imshow_y)
                    self.general_window_moved = True

            # Draw Point Cloud on the Frame
            pc_len = pc_img_coord.shape[0]
            for pc_img_coord_idx in range(pc_len):
                pc_point = tuple(pc_img_coord[pc_img_coord_idx])
                cv2.circle(
                    img=vis_frame, center=pc_point, radius=3,
                    color=color[pc_img_coord_idx], thickness=-1
                )

            # IMSHOW
            if modal_type == "color":
                cv2.imshow(winname, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            else:
                cv2.imshow(winname, vis_frame)

            cv2.waitKey(1)

    # Visualizer as Function
    def __call__(self, sync_data_dict, trajectories, detections, segmentation, fidx):
        # Get Detection Result Modals
        det_modals = detections.keys()

        # Get Trajectory Result Modals
        trk_modals = []
        for trk in trajectories:
            curr_modal = trk.modal
            if curr_modal not in trk_modals:
                trk_modals.append(trk.modal)

        # Draw Segmentation Results
        if self.vopts.segmentation["is_draw"] is True and segmentation is not None:
            # Update Visualizer Object
            self.SEG_VIS_OBJ.update_objects(segmentation)

            # Draw Segmentation Results
            seg_vis_frame, seg_winname = self.SEG_VIS_OBJ.draw_objects()

            # Auto-Save
            if self.vopts.segmentation["auto_save"] is True:
                auto_save_segmentation_base_dir = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "sample_results", "segmentation"
                )
                if os.path.isdir(auto_save_segmentation_base_dir) is False:
                    os.makedirs(auto_save_segmentation_base_dir)
                self.save_frame(
                    save_base_dir=auto_save_segmentation_base_dir,
                    frame=seg_vis_frame, fidx=fidx
                )
        else:
            seg_vis_frame, seg_winname = None, None

        # Show Segmentation Results
        if self.vopts.segmentation["is_show"] is True:
            cv2.namedWindow(seg_winname)
            cv2.imshow(seg_winname, seg_vis_frame)

        # Draw Detection Results
        if self.vopts.detection["is_draw"] is True:
            # Update Visualizer Object
            self.DET_VIS_OBJ.update_objects(detections)

            # Draw Detection Results
            det_vis_frames, det_winnames = self.DET_VIS_OBJ.draw_objects(sync_data_dict=sync_data_dict)

            # Auto-Save
            for modal, det_vis_frame in det_vis_frames.items():
                if self.vopts.detection["auto_save"] is True:
                    _auto_save_detection_base_dir = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)), "sample_results", "detections"
                    )
                    auto_save_detection_base_dir = os.path.join(
                        _auto_save_detection_base_dir, modal
                    )
                    if os.path.isdir(auto_save_detection_base_dir) is False:
                        os.makedirs(auto_save_detection_base_dir)

                    # Save, give options to save frames only with detection results
                    is_save_only_objects = False
                    if is_save_only_objects is True:
                        if detections[modal]["dets"].size != 0:
                            self.save_frame(
                                save_base_dir=auto_save_detection_base_dir,
                                frame=det_vis_frame, fidx=fidx
                            )
                    else:
                        self.save_frame(
                            save_base_dir=auto_save_detection_base_dir,
                            frame=det_vis_frame, fidx=fidx
                        )
        else:
            det_vis_frames, det_winnames = None, None

        # Show Detection Results
        if self.vopts.detection["is_show"] is True:
            for idx, (modal, det_vis_frame) in enumerate(det_vis_frames.items()):
                # Get OpenCV Named Window
                det_winname = det_winnames[modal]
                cv2.namedWindow(det_winname)

                # Move Window
                if self.det_window_moved is False:
                    if self.screen_imshow_x is not None and self.screen_imshow_y is not None:
                        cv2.moveWindow(det_winname, self.screen_imshow_x, self.screen_imshow_y)
                        if idx == len(det_vis_frames)-1:
                            self.det_window_moved = True

                # Show Result (imshow)
                if len(det_vis_frame.shape) == 3:
                    cv2.imshow(det_winname, cv2.cvtColor(det_vis_frame, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imshow(det_winname, det_vis_frame)

        # Draw Tracking & Action Classification Results
        if self.vopts.tracking["is_draw"] is True:
            # Update Visualizer Object
            self.TRK_ACL_VIS_OBJ.update_objects(trajectories)

            # Draw Tracking & Action Classification Results
            trk_acl_frames, trk_acl_winnames = self.TRK_ACL_VIS_OBJ.draw_objects(
                sync_data_dict=sync_data_dict, det_modals=detections.keys()
            )

            # Auto-Save
            for modal, trk_acl_frame in trk_acl_frames.items():
                if self.vopts.tracking["auto_save"] is True:
                    _auto_save_tracking_base_dir = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)), "sample_results", "tracking_action"
                    )
                    auto_save_tracking_base_dir = os.path.join(
                        _auto_save_tracking_base_dir, modal
                    )
                    if os.path.isdir(auto_save_tracking_base_dir) is False:
                        os.makedirs(auto_save_tracking_base_dir)

                    # Save, give options to save frames only with detection results
                    is_save_only_objects = False
                    if is_save_only_objects is True:
                        if len(trajectories) != 0:
                            self.save_frame(
                                save_base_dir=auto_save_tracking_base_dir,
                                frame=trk_acl_frame, fidx=fidx
                            )
                    else:
                        self.save_frame(
                            save_base_dir=auto_save_tracking_base_dir,
                            frame=trk_acl_frame, fidx=fidx
                        )
        else:
            trk_acl_frames, trk_acl_winnames = None, None

        # Show Tracking and Action Classification Results
        if self.vopts.tracking["is_show"] is True:
            for idx, (modal, trk_acl_frame) in enumerate(trk_acl_frames.items()):
                # Get OpenCV Named Window
                trk_acl_winname = trk_acl_winnames[modal]
                cv2.namedWindow(trk_acl_winname)

                # Move Window
                if self.trk_acl_window_moved is False:
                    if self.screen_imshow_x is not None and self.screen_imshow_y is not None:
                        cv2.moveWindow(trk_acl_winname, self.screen_imshow_x, self.screen_imshow_y)
                        if idx == len(trk_acl_frames)-1:
                            self.trk_acl_window_moved = True

                # Show Result (imshow)
                if len(trk_acl_frame.shape) == 3:
                    cv2.imshow(trk_acl_winname, cv2.cvtColor(trk_acl_frame, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imshow(trk_acl_winname, trk_acl_frame)

        # WaitKey for OpenCV
        if self.vopts.detection["is_show"] is True or self.vopts.tracking["is_show"] is True or \
                self.vopts.aclassifier["is_show"] is True or self.vopts.top_view["is_show"] is True or \
                self.vopts.segmentation["is_show"] is True :
            cv2.waitKey(1)

        # Show Tracking & Action Classification Results
        return {"det": det_vis_frames, "trk_acl": trk_acl_frames}


if __name__ == "__main__":
    pass
