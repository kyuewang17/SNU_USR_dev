"""
SNU Integrated Module v5.0
  - Code which defines Coordinate Objects for Object Tracking
  - Coordinate Format Array Information

    (1) 'observation' ( 1x4 )
        [x, y, w, h]
    (2) 'state_image' ( 1x8 )
        [x, y, dx, dy, w, h, depth, d_depth]
    (3) 'state_camera' ( 1x6 )
        [x_cam, y_cam, z_cam, dx_cam, dy_cam, dz_cam]

"""
import numpy as np
from module_lib.v4_5._TRK.objects.bbox import *


class COORD(object):
    def __init__(self, **kwargs):
        # Designate COORD Format
        coord_format = kwargs.get("coord_format")
        assert coord_format in ["observation", "state_image", "state_camera"]
        self.coord_format = coord_format

    def __repr__(self):
        return self.coord_format

    def __iter__(self):
        return self


class OBSERVATION_COORD(COORD):
    def __init__(self, **kwargs):
        super(OBSERVATION_COORD, self).__init__(coord_format="observation")

        # Get Input Coordinates
        bbox_object = kwargs.get("bbox_object")
        if bbox_object is not None:
            assert isinstance(bbox_object, BBOX)
            self.x, self.y, self.w, self.h = \
                bbox_object.x, bbox_object.y, bbox_object.w, bbox_object.h
        else:
            self.x, self.y, self.w, self.h = \
                kwargs.get("x"), kwargs.get("y"), kwargs.get("w"), kwargs.get("h")
            assert self.w > 0 and self.h > 0

        # Set Iteration Counter
        self.__iter_counter = 0

    def __getitem__(self, idx):
        assert isinstance(idx, int) and 0 <= idx <= 3
        ret_list = [self.x, self.y, self.w, self.h]
        return ret_list[idx]

    def __setitem__(self, idx, value):
        attr_str_list = ["x", "y", "w", "h"]
        setattr(self, attr_str_list[idx], value)

    def next(self):
        try:
            iter_item = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter = 0
        return iter_item

    def get_bbox(self, conversion_fmt="XYWH"):
        assert conversion_fmt in ["XYWH", "LTRB", "LTWH"]
        bbox_obj = BBOX(
            bbox_format="XYWH",
            x=self.x, y=self.y, w=self.w, h=self.h
        )
        bbox_obj.convert_bbox_fmt(conversion_fmt)
        return bbox_obj

    def get_intersection_bbox(self, other, **kwargs):
        assert isinstance(other, (BBOX, OBSERVATION_COORD, STATE_IMAGE_COORD))

        # Get Return Format
        return_fmt = kwargs.get("return_fmt", "LTRB")
        assert return_fmt in ["LTRB", "LTWH", "XYWH"]

        # Get Left-Top Right-Bottom Coordinates
        ltrb_bbox = self.get_bbox(conversion_fmt="LTRB")

        if isinstance(other, BBOX):
            # Get Min-Max Coordinates
            uu1 = np.maximum(ltrb_bbox.lt_x, other.lt_x)
            vv1 = np.maximum(ltrb_bbox.lt_y, other.lt_y)
            uu2 = np.minimum(ltrb_bbox.rb_x, other.rb_x)
            vv2 = np.minimum(ltrb_bbox.rb_y, other.rb_y)

        elif isinstance(other, OBSERVATION_COORD):
            other_ltrb_bbox = other.get_bbox(conversion_fmt="LTRB")

            # Get Min-Max Coordinates
            uu1 = np.maximum(ltrb_bbox.lt_x, other_ltrb_bbox.lt_x)
            vv1 = np.maximum(ltrb_bbox.lt_y, other_ltrb_bbox.lt_y)
            uu2 = np.minimum(ltrb_bbox.rb_x, other_ltrb_bbox.rb_x)
            vv2 = np.minimum(ltrb_bbox.rb_y, other_ltrb_bbox.rb_y)

        elif isinstance(other, STATE_IMAGE_COORD):




            pass



class STATE_IMAGE_COORD(COORD):
    def __init__(self, **kwargs):
        super(STATE_IMAGE_COORD, self).__init__(coord_format="state_image")


class STATE_CAMERA_COORD(COORD):
    def __init__(self, **kwargs):
        super(STATE_CAMERA_COORD, self).__init__(coord_format="state_camera")


# Object Class Container for Multiple Coordinates
class COORDS(object):
    def __init__(self, **kwargs):
        pass




























if __name__ == "__main__":
    pass
