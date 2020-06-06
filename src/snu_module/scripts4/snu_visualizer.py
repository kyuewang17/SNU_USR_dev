"""
SNU Integrated Module v3.0
    - Visualization Class for SNU Results
"""
# Import Modules
import cv2
import numpy as np
import copy
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Import Custom Modules
import snu_utils.bounding_box as fbbox


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
    def update_objects(self, result_obj):
        raise NotImplementedError

    # Draw Objects on Selected Modal Frame
    def draw_objects(self, frame, winname):
        raise NotImplementedError


# Define Detection Result Visualization Object
class vis_det_obj(vis_obj):
    def __init__(self, vopts):
        """
        vopts: visualizer_options
        """
        super(vis_det_obj, self).__init__(vopts)

        # Detection Confidence and Labels (each [Nx1] numpy array)
        self.confs, self.labels = None, None

        # Custom Color and Object Linewidth Settings
        self.color_code = vopts.detection["bbox_color"]
        self.linewidth = vopts.detection["linewidth"]

    # Update Detection BBOX Objects
    def update_objects(self, result_detections):
        self.bboxes, self.confs, self.labels = \
            result_detections["dets"], result_detections["confs"], result_detections["labels"]

    # Draw Detection Result on Selected Modal Frame
    def draw_objects(self, frame, opencv_winname):
        for bbox in self.bboxes:
            # Convert BBOX Precision
            bbox = bbox.astype(np.int32)

            # Draw Rectangle BBOX
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                self.color_code, self.linewidth
            )

        # Set OpenCV Window Name
        opencv_winname += " {DET}"

        return frame, opencv_winname


# Define Tracklet (TRK & ACL) Result Visualization Object
class vis_trk_acl_obj(vis_obj):
    def __init__(self, vopts):
        """
        vopts: visualizer_options
        """
        super(vis_trk_acl_obj, self).__init__(vopts)

        # Delete BBOX Attribute
        delattr(self, "bboxes")

        # Initialize Tracklets (List for Tracklet Class)
        self.trks = []

        # Object Linewidth Settings
        self.linewidth = vopts.tracking["linewidth"]

    # Update Tracklet Object
    def update_objects(self, tracklets):
        self.trks = tracklets

    # Draw Tracking Result on Selected Modal Frame
    def draw_objects(self, frame, opencv_winname):
        for trk in self.trks:
            # Convert State BBOX coordinate type and precision
            state_bbox, _ = fbbox.zx_to_bbox(trk.states[-1])
            state_bbox = state_bbox.astype(np.int32)

            # Draw Rectangle BBOX
            cv2.rectangle(
                frame,
                (state_bbox[0], state_bbox[1]), (state_bbox[2], state_bbox[3]),
                (trk.color[0], trk.color[1], trk.color[2]), self.linewidth
            )

            # Unpack Visualization Options
            font = self.vopts.font
            font_size = self.vopts.font_size
            pad_pixels = self.vopts.pad_pixels
            info_interval = self.vopts.info_interval

            # Visualize Tracklet ID
            trk_id_str = "id:" + str(trk.id) + ""
            (tw, th) = cv2.getTextSize(trk_id_str, font, fontScale=font_size, thickness=2)[0]
            text_x = int((state_bbox[0] + state_bbox[2]) / 2.0 - tw / 2.0)
            text_y = int(state_bbox[1] + th)
            box_coords = ((int(text_x - pad_pixels / 2.0), int(text_y - th - pad_pixels / 2.0)),
                          (int(text_x + tw + pad_pixels / 2.0), int(text_y + pad_pixels / 2.0)))
            cv2.rectangle(frame, box_coords[0], box_coords[1], (trk.color[0], trk.color[1], trk.color[2]), cv2.FILLED)
            cv2.putText(
                frame, trk_id_str, (text_x, text_y), font, font_size,
                (255 - trk.color[0], 255 - trk.color[1], 255 - trk.color[2]), thickness=2
            )

            # Visualize Tracklet Depth
            if trk.depth is not None:
                trk_depth_str = "d=" + str(round(trk.x3[2], 3)) + "(m)"
                (tw, th) = cv2.getTextSize(trk_depth_str, font, fontScale=1.2, thickness=2)[0]
                text_x = int((state_bbox[0] + state_bbox[2]) / 2.0 - tw / 2.0)
                text_y = int((state_bbox[1] + state_bbox[3]) / 2.0 - th / 2.0)

                # Put Depth Text (Tentative)
                cv2.putText(frame, trk_depth_str, (text_x, text_y), font, 1.2,
                            (255 - trk.color[0], 255 - trk.color[1], 255 - trk.color[2]), thickness=2)

            # Visualize Action Classification Result
            if trk.pose is not None and self.vopts.aclassifier["is_draw"] is True:
                # Draw Action Results only when Human
                # TODO: (later, handle this directly on action classification module)
                if trk.label == 1:
                    # Get Image Height and Width
                    H, W = frame.shape[0], frame.shape[1]

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
                    cv2.putText(frame, pose_word,
                                (min(int(trk.x[0] + (trk.x[4] / 2)), W - 1), min(int(trk.x[1] + (trk.x[5] / 2)), H - 1)),
                                font, 1.5, (255 - trk.color[0], 255 - trk.color[1], 255 - trk.color[2]), thickness=2)

        # Set OpenCV Window Name
        if self.vopts.tracking["is_draw"] is True and self.vopts.aclassifier["is_draw"] is True:
            opencv_winname += " {TRK+ACL}"
        elif self.vopts.tracking["is_draw"] is True:
            opencv_winname += " {TRK}"
        elif self.vopts.aclassifier["is_draw"] is True:
            opencv_winname += " {ACL}"

        return frame, opencv_winname


# Define Visualizer Class
class visualizer(object):
    def __init__(self, opts):
        # OpenCV Window Base Name
        self.winname = "SNU Result"

        # Get Visualizer Option for SNU options
        self.vopts = opts.visualization

        # Allocate Modal Frame Spaces (except for LiDAR)
        self.modal_frames = {}
        for modal, modal_switch in opts.modal_switch_dict.items():
            if modal_switch is True and modal != "lidar":
                self.modal_frames[modal] = None

        # Initialize Visualization Objects
        self.DET_VIS_OBJ = vis_det_obj(vopts=self.vopts)
        self.TRK_ACL_VIS_OBJ = vis_trk_acl_obj(vopts=self.vopts)

        # Screen Geometry Imshow Position
        self.screen_imshow_x = opts.screen_imshow_x
        self.screen_imshow_y = opts.screen_imshow_y
        self.det_window_moved, self.trk_acl_window_moved = False, False

        # Top-view Map (in mm)
        self.top_view_map = np.ones(
            shape=[opts.visualization.top_view["map_size"][0], opts.visualization.top_view["map_size"][1], 3],
            dtype=np.uint8
        )
        self.top_view_window_moved = False

        # Top-view Max Axis
        self.u_max_axis, self.v_max_axis = 0, 0
        self.u_min_axis, self.v_min_axis = 0, 0

    # Visualize Image Sequences Only
    @staticmethod
    def visualize_modal_frames(sensor_data):
        if sensor_data is not None:
            # Get Visualization Sensor Data Modality
            modal_type = sensor_data.modal_type

            # Get Image Frame
            vis_frame = sensor_data.frame

            # OpenCV Window Name
            winname = "[%s]" % modal_type

            # Make NamedWindow
            cv2.namedWindow(winname)

            # IMSHOW
            if modal_type == "color":
                cv2.imshow(winname, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            else:
                cv2.imshow(winname, vis_frame)

            cv2.waitKey(1)

    # Functional Visualizer Call
    def __call__(self, sensor_data, tracklets, detections, _check_run_time=False):
        # Get Visualization Sensor Data Modality
        modal_type = sensor_data.modal_type

        # Get Image Frame
        vis_frame = sensor_data.frame

        # OpenCV Window Name
        winname = self.winname + "(%s)" % modal_type

        # Update Module Results
        self.DET_VIS_OBJ.update_objects(detections)

        # Draw Detection Results on Frame
        det_vis_frame, det_winname = self.DET_VIS_OBJ.draw_objects(
            frame=copy.deepcopy(vis_frame), opencv_winname=winname
        )

        # Update Module Results
        self.TRK_ACL_VIS_OBJ.update_objects(tracklets)

        # Draw Tracking/Action Classification Results on Frame
        trk_acl_frame, trk_acl_winname = self.TRK_ACL_VIS_OBJ.draw_objects(
            frame=copy.deepcopy(vis_frame), opencv_winname=winname
        )

        # Visualize Detection Results
        if self.vopts.detection["is_draw"] is True:
            # Make NamedWindow
            cv2.namedWindow(det_winname)

            # Move Window
            if self.det_window_moved is False:
                if self.screen_imshow_x is not None and self.screen_imshow_y is not None:
                    cv2.moveWindow(det_winname, self.screen_imshow_x, self.screen_imshow_y)
                    self.det_window_moved = True

            # IMSHOW
            # If Visualization Frame is Color Modality, convert RGB to BGR
            cv2.imshow(det_winname, cv2.cvtColor(det_vis_frame, cv2.COLOR_RGB2BGR))

        if self.vopts.tracking["is_draw"] is True or self.vopts.aclassifier["is_draw"] is True:
            # Make NamedWindow
            cv2.namedWindow(trk_acl_winname)

            # Move Window
            if self.trk_acl_window_moved is False:
                if self.screen_imshow_x is not None and self.screen_imshow_y is not None:
                    cv2.moveWindow(trk_acl_winname, self.screen_imshow_x, self.screen_imshow_y)
                    self.trk_acl_window_moved = True

            # IMSHOW
            # If Visualization Frame is Color Modality, convert RGB to BGR
            cv2.imshow(trk_acl_winname, cv2.cvtColor(trk_acl_frame, cv2.COLOR_RGB2BGR))

        # Visualize Top-view Tracklets
        if self.vopts.top_view["is_draw"] is True:
            self.visualize_top_view_tracklets(tracklets=tracklets)

        if self.vopts.detection["is_draw"] is True or self.vopts.tracking["is_draw"] is True or \
                self.vopts.aclassifier["is_draw"] is True or self.vopts.top_view["is_draw"] is True:
            cv2.waitKey(1)

        return {"det": det_vis_frame, "trk_acl": trk_acl_frame}

    # Visualize Top-view Tracklets
    # TODO: Cost-down Calculation
    def visualize_top_view_tracklets(self, tracklets):
        # Top-view Map Coordinate to Row/Column Axis Value
        def cam_coord_to_top_view_coord(x, y):
            row = int(self.top_view_map.shape[0] - y)
            col = int(self.top_view_map.shape[1] / 2.0 + x)
            return row, col

        # Top-view OpenCV Plot Window Name
        top_view_winname = "Top-View Tracklet Results"

        # For Drawing Top-view Tracklets
        top_view_map = copy.deepcopy(self.top_view_map)

        # Draw Robot in the Top-view Map
        robot_coord_row, robot_coord_col = cam_coord_to_top_view_coord(0, 0)
        cv2.circle(
            img=top_view_map, center=(robot_coord_row, robot_coord_col),
            radius=int(top_view_map.shape[1]/100),
            color=(0, 0, 0), thickness=cv2.FILLED
        )

        # Draw Tracklet Coordinates to Top-view Map
        for trk_idx, trk in enumerate(tracklets):
            # Tracklet Coordinates < (-y) and (+x) Coordinates >
            top_view_x, top_view_y = -trk.c3[1][0]*1000, trk.c3[0][0]*1000

            # Convert to Top-view Map Row/Column
            top_view_row, top_view_col = \
                cam_coord_to_top_view_coord(top_view_x, top_view_y)

            # Draw Tracklets as Circles
            cv2.circle(
                img=top_view_map, center=(top_view_col, top_view_row),
                radius=100,
                color=(trk.color[0], trk.color[1], trk.color[2]), thickness=cv2.FILLED
            )

        # Reshape Image
        scale_percent = 5
        width = int(top_view_map.shape[1] * scale_percent / 100)
        height = int(top_view_map.shape[0] * scale_percent / 100)
        resized_top_view_map = cv2.resize(
            top_view_map, (width, height), interpolation=cv2.INTER_AREA
        )

        # Plot as OpenCV Window
        cv2.namedWindow(top_view_winname)
        if self.top_view_window_moved is False:
            cv2.moveWindow(top_view_winname, self.screen_imshow_x, self.screen_imshow_y)
            self.top_view_window_moved = True
        cv2.imshow(top_view_winname, cv2.cvtColor(resized_top_view_map, cv2.COLOR_RGB2BGR))

        cv2.waitKey(1)


def main():
    pass


if __name__ == "__main__":
    pass
