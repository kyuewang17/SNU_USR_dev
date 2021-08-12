"""
SNU Integrated Module v5.0
  - Code which defines Coordinate Objects for Object Tracking
  - Coordinate Format Array Information

    (1) 'observation' ( 1x7 )
        [x, y, dx, dy, w, h, depth]
    (2) 'state_image' ( 1x7 )
        [x, y, dx, dy, w, h, depth]
    (3) 'state_camera' ( 1x6 )
        [x_cam, y_cam, z_cam, dx_cam, dy_cam, dz_cam]

"""
import numpy as np
from module_lib.v4_5._TRK.objects.bbox import *


class COORD(object):
    def __init__(self, **kwargs):
        # Designate COORD Format
        coord_format = kwargs.get("coord_format")
        self.coord_format = coord_format

        # Initialize Coordinates
        self.x, self.y, self.dx, self.dy, self.w, self.h, self.depth = \
            None, None, None, None, None, None, None

        # Set Iteration Counter
        self.__iter_counter = 0

    def __len__(self):
        return 7

    def __getitem__(self, idx):
        ret_list = [self.x, self.y, self.dx, self.dy, self.w, self.h, self.depth]
        return ret_list[idx]

    def __setitem__(self, idx, value):
        attr_str_list = ["x", "y", "dx", "dy", "w", "h", "depth"]
        setattr(self, attr_str_list[idx], value)

    def __repr__(self):
        return self.coord_format

    def __iter__(self):
        return self

    def next(self):
        try:
            iter_item = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return iter_item

    def get_size(self):
        return self.w * self.h

    def get_patch(self, frame, patch_size_factor=1.0):
        return self.to_bbox().get_patch(frame=frame, patch_size_factor=patch_size_factor)

    def numpify(self):
        ret_arr = np.empty(len(self))
        for arr_idx in self:
            ret_arr[arr_idx] = self[arr_idx]
        return ret_arr

    def to_bbox(self, **kwargs):
        # Get Conversion Format
        conversion_fmt = kwargs.get("conversion_fmt", "XYWH")
        assert conversion_fmt in ["XYWH", "LTRB", "LTWH"]

        # Get Resize Ratio
        x_ratio, y_ratio = kwargs.get("x_ratio", 1.0), kwargs.get("y_ratio", 1.0)

        # Initialize BBOX Object
        bbox_obj = BBOX(
            bbox_format="XYWH",
            x=self.x, y=self.y, w=self.w, h=self.h
        )

        # Resize
        if x_ratio != 1 or y_ratio != 1:
            bbox_obj.resize(x_ratio=x_ratio, y_ratio=y_ratio)

        # Format Conversion
        bbox_obj.convert_bbox_fmt(conversion_fmt)
        return bbox_obj

    def get_intersection_bbox(self, other, **kwargs):
        assert isinstance(other, (BBOX, COORD))

        # Get Return Format
        return_fmt = kwargs.get("return_fmt", "LTRB")
        assert return_fmt in ["LTRB", "LTWH", "XYWH"]

        # Get Left-Top Right-Bottom Coordinates
        ltrb_bbox = self.to_bbox(conversion_fmt="LTRB")

        if isinstance(other, BBOX):
            # Get Min-Max Coordinates
            uu1 = np.maximum(ltrb_bbox.lt_x, other.lt_x)
            vv1 = np.maximum(ltrb_bbox.lt_y, other.lt_y)
            uu2 = np.minimum(ltrb_bbox.rb_x, other.rb_x)
            vv2 = np.minimum(ltrb_bbox.rb_y, other.rb_y)

        elif isinstance(other, COORD):
            # Get LTRB Coordinates of other object
            other__lt_x, other__lt_y = other.x - other.w / 2.0, other.y - other.h / 2.0
            other__rb_x, other__rb_y = other.x + other.w / 2.0, other.y + other.h / 2.0

            # Get Min-Max Coordinates
            uu1 = np.maximum(ltrb_bbox.lt_x, other__lt_x)
            vv1 = np.maximum(ltrb_bbox.lt_y, other__lt_y)
            uu2 = np.minimum(ltrb_bbox.rb_x, other__rb_x)
            vv2 = np.minimum(ltrb_bbox.rb_y, other__rb_y)

        else:
            raise AssertionError()

        # Initialize Common BBOX
        common_bbox = BBOX(
            bbox_format="LTRB", lt_x=uu1, lt_y=vv1, rb_x=uu2, rb_y=vv2
        )

        # Format Conversion and Return BBOX
        common_bbox.convert_bbox_fmt(return_fmt)
        return common_bbox

    def get_iou(self, other):
        assert isinstance(other, (BBOX, COORD))

        # Get Intersection BBOX
        common_bbox = self.get_intersection_bbox(other)

        # Get Intersection Area
        common_area = common_bbox.get_size()

        # Get Union Area
        union_area = self.get_size() + other.get_size() - common_area

        # Return
        if union_area == 0:
            return 0.0
        else:
            return float(common_area) / float(union_area)

    def get_ioc(self, other, denom_comp="other"):
        assert isinstance(other, (BBOX, COORD))

        # Get Intersection BBOX
        common_bbox = self.get_intersection_bbox(other)

        # Get Intersection Area
        common_area = common_bbox.get_size()

        # Get Denominator Component
        assert denom_comp in ["other", "self"]

        # Get Denominator Area
        if denom_comp == "other":
            denom_area = other.get_size()
        else:
            denom_area = self.get_size()

        # Return
        return float(common_area) / float(denom_area)


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

        # Get Velocities
        self.dx, self.dy = kwargs.get("dx", 0.0), kwargs.get("dy", 0.0)

        # Get Depth
        self.depth = kwargs.get("depth", 0.0)

        # Set Iteration Counter
        self.__iter_counter = 0

    def next(self):
        try:
            iter_item = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return iter_item

    def to_state(self):
        # Return State Image Coordinate Object
        return STATE_IMAGE_COORD(
            x=self.x, y=self.y, dx=self.dx, dy=self.dy,
            w=self.w, h=self.h, depth=self.depth
        )


class STATE_IMAGE_COORD(COORD):
    def __init__(self, **kwargs):
        super(STATE_IMAGE_COORD, self).__init__(coord_format="state_image")

        # Get Array
        input_arr = kwargs.get("input_arr")

        # Initialize Coordinates
        if input_arr is None:
            self.x, self.y = kwargs.get("x"), kwargs.get("y")
            self.dx, self.dy = kwargs.get("dx"), kwargs.get("dy")
            self.w, self.h = kwargs.get("w"), kwargs.get("h")
            self.depth = kwargs.get("depth")
        else:
            assert isinstance(input_arr, np.ndarray) and input_arr.size == 8
            input_arr_vec = input_arr.reshape(-1)
            self.x, self.y, self.dx, self.dy = \
                input_arr_vec[0], input_arr_vec[1], input_arr_vec[2], input_arr_vec[3]
            self.w, self.h = input_arr_vec[4], input_arr_vec[5]
            self.depth = input_arr_vec[6]

        # Set Iteration Counter
        self.__iter_counter = 0

    def next(self):
        try:
            iter_item = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return iter_item

    def to_observation_coord(self):
        return OBSERVATION_COORD(
            x=self.x, y=self.y, dx=self.dx, dy=self.dy, w=self.w, h=self.h, depth=self.depth
        )

    def to_camera_coord(self, **kwargs):
        """
        Return 'STATE_CAMERA_COORD' Class Object, implemented below,

        for Fixed Agents, Project Bottom-Center Point in Image Coordinates to Ground Plane in World Coordinates
        for Moving Agents, Project BBOX Center Point in Image Coordinates to Camera Coordinates

        """
        raise NotImplementedError()


class STATE_CAMERA_COORD(object):
    def __init__(self, **kwargs):
        super(STATE_CAMERA_COORD, self).__init__(coord_format="state_camera")

        # Initialize Coordinates
        self.x, self.y, self.z = kwargs.get("x"), kwargs.get("y"), kwargs.get("z")
        self.dx, self.dy, self.dz = kwargs.get("dx"), kwargs.get("dy"), kwargs.get("dz")

        # Set Iteration Counter
        self.__iter_counter = 0

    def __len__(self):
        return 6

    def __getitem__(self, idx):
        ret_list = [self.x, self.y, self.z, self.dx, self.dy, self.dz]
        return ret_list[idx]

    def __setitem__(self, idx, value):
        attr_str_list = ["x", "y", "z", "dx", "dy", "dz"]
        setattr(self, attr_str_list[idx], value)

    def __iter__(self):
        return self

    def next(self):
        try:
            iter_item = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return iter_item

    def numpify(self):
        ret_arr = np.empty(len(self))
        for arr_idx in range(len(self)):
            ret_arr[arr_idx] = self[arr_idx]
        return ret_arr

    def to_image_coord(self, **kwargs):
        """
        Return 'STATE_IMAGE_COORD' Class Object, implemented above
        width, height, d_depth values are set to "zero"

        """
        raise NotImplementedError()


if __name__ == "__main__":
    pass
