"""
SNU Integrated Module v5.0
  - Code which defines Bounding Box Objects for Object Tracking

"""
import numpy as np


class BBOX(object):
    def __init__(self, **kwargs):
        # Designate BBOX Format
        bbox_format = kwargs.get("bbox_format")
        assert bbox_format in ["LTRB", "LTWH", "XYWH"]
        self.bbox_format = bbox_format

        # Case-wise Input of BBOX Coordinates
        if bbox_format == "LTRB":
            self.lt_x, self.lt_y = kwargs.get("lt_x"), kwargs.get("lt_y")
            self.rb_x, self.rb_y = kwargs.get("rb_x"), kwargs.get("rb_y")

            assert self.lt_x < self.rb_x and self.lt_y < self.rb_y

            self.x, self.y = (self.lt_x + self.rb_x) / 2.0, (self.lt_y + self.rb_y) / 2.0
            self.w, self.h = (self.rb_x - self.lt_x), (self.rb_y - self.lt_y)

        elif bbox_format == "LTWH":
            self.lt_x, self.lt_y = kwargs.get("lt_x"), kwargs.get("lt_y")
            self.w, self.h = kwargs.get("w"), kwargs.get("h")

            assert self.w > 0 and self.h > 0

            self.rb_x, self.rb_y = (self.lt_x + self.w), (self.lt_y + self.h)
            self.x, self.y = (self.lt_x + self.w / 2.0), (self.lt_y + self.h / 2.0)

        elif bbox_format == "XYWH":
            self.x, self.y, self.w, self.h = \
                kwargs.get("x"), kwargs.get("y"), kwargs.get("w"), kwargs.get("h")

            assert self.w > 0 and self.h > 0

            self.lt_x, self.lt_y = (self.x - self.w / 2.0), (self.y - self.h / 2.0)
            self.rb_x, self.rb_y = (self.x + self.w / 2.0), (self.y + self.h / 2.0)

        # Iteration Counter
        self.__iter_counter = 0

    def __repr__(self):
        return self.bbox_format

    def __getitem__(self, idx):
        assert isinstance(idx, int) and 0 <= idx <= 3
        if self.bbox_format == "LTRB":
            ret_list = [self.lt_x, self.lt_y, self.rb_x, self.rb_y]
        elif self.bbox_format == "LTWH":
            ret_list = [self.lt_x, self.lt_y, self.w, self.h]
        elif self.bbox_format == "XYWH":
            ret_list = [self.x, self.y, self.w, self.h]
        else:
            raise AssertionError()
        return ret_list[idx]

    def __setitem__(self, idx, value):
        if self.bbox_format == "LTRB":
            attr_str_list = ["lt_x", "lt_y", "rb_x", "rb_y"]
        elif self.bbox_format == "LTWH":
            attr_str_list = ["lt_x", "lt_y", "w", "h"]
        elif self.bbox_format == "XYWH":
            attr_str_list = ["x", "y", "w", "h"]
        else:
            raise AssertionError()
        setattr(self, attr_str_list[idx], value)
        self.__adjust_coordinates(pivot_fmt=self.bbox_format)

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

    def __sub__(self, other):
        if isinstance(other, (list, tuple)):
            assert len(other) == 2
            v_x, v_y = self.x - other[0], self.y - other[1]
        elif isinstance(other, np.ndarray):
            assert other.size == 2
            other_vec = other.reshape(2)
            v_x, v_y = self.x - other_vec[0], self.y - other_vec[1]
        elif isinstance(other, BBOX):
            v_x, v_y = self.x - other.x, self.y - other.y
        else:
            raise NotImplementedError()

        return np.array([v_x, v_y])

    def get_size(self):
        return np.maximum(self.w * self.h, 0.0)

    def get_diagonal_length(self):
        return np.sqrt(self.w ** 2 + self.h ** 2)

    def get_intersection_bbox(self, other, **kwargs):
        assert isinstance(other, BBOX)

        # Get Conversion Format
        conversion_fmt = kwargs.get("conversion_fmt", self.bbox_format)
        assert conversion_fmt in ["LTRB", "LTWH", "XYWH"]

        # Get Min-Max Coordinates
        uu1 = np.maximum(self.lt_x, other.lt_x)
        vv1 = np.maximum(self.lt_y, other.lt_y)
        uu2 = np.minimum(self.rb_x, other.rb_x)
        vv2 = np.minimum(self.rb_y, other.rb_y)

        # Initialize Common BBOX of LTRB, Convert to Conversion Format
        common_bbox = BBOX(
            bbox_format="LTRB", lt_x=uu1, lt_y=vv1, rb_x=uu2, rb_y=vv2
        )
        common_bbox.convert_bbox_fmt(conversion_fmt)

        # Finally Return
        return common_bbox

    def to_ndarray(self, dtype=None):
        ret_arr = np.array([self[0], self[1], self[2], self[3]])
        if dtype is not None:
            ret_arr = ret_arr.astype(dtype=dtype)
        return ret_arr

    def to_list(self):
        return [self[0], self[1], self[2], self[3]]

    def __adjust_coordinates(self, pivot_fmt=None):
        if pivot_fmt is None:
            pivot_fmt = self.bbox_format
        else:
            assert pivot_fmt in ["LTRB", "LTWH", "XYWH"]

        # Change Other Coordinates, according to pivot format
        if pivot_fmt == "LTRB":
            self.x = (self.lt_x + self.rb_x) / 2.0
            self.y = (self.lt_y + self.rb_y) / 2.0
            self.w = (self.rb_x - self.lt_x)
            self.h = (self.rb_y - self.lt_y)

        elif pivot_fmt == "LTWH":
            self.x = (self.lt_x + self.w / 2.0)
            self.y = (self.lt_y + self.h / 2.0)
            self.rb_x = self.lt_x + self.w
            self.rb_y = self.lt_y + self.h

        elif pivot_fmt == "XYWH":
            self.lt_x = self.x - self.w / 2.0
            self.lt_y = self.y - self.h / 2.0
            self.rb_x = self.x + self.w / 2.0
            self.rb_y = self.y + self.h / 2.0

    def resize(self, x_ratio, y_ratio):
        # Get LTRB Points
        a, b, c, d = self.lt_x, self.lt_y, self.rb_x, self.rb_y

        # Set Coefficients
        K_alpha_plus, K_alpha_minus = 0.5 * (1 + x_ratio), 0.5 * (1 - x_ratio)
        K_beta_plus, K_beta_minus = 0.5 * (1 + y_ratio), 0.5 * (1 - y_ratio)

        # Convert
        self.lt_x = K_alpha_plus * a + K_alpha_minus * c
        self.lt_y = K_beta_plus * b + K_beta_minus * d
        self.rb_x = K_alpha_minus * a + K_alpha_plus * c
        self.rb_y = K_beta_minus * b + K_beta_plus * d

        # Adjust Coordinates
        self.__adjust_coordinates("LTRB")

    def get_iou(self, other):
        assert isinstance(other, BBOX)

        # Get Intersection BBOX
        common_bbox = self.get_intersection_bbox(other, conversion_fmt="LTRB")

        # Get Intersection Area
        common_area = common_bbox.get_size()

        # Get Union Area (AUB = A + B - A^B)
        union_area = self.get_size() + other.get_size() - common_area

        # Return
        if union_area == 0:
            return 0.0
        else:
            return float(common_area) / float(union_area)

    def get_ioc(self, other, denom_comp="other"):
        """
        Denominator is the 'other' component

        """
        # Get Intersection BBOX
        common_bbox = self.get_intersection_bbox(other, conversion_fmt="LTRB")

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

    def convert_bbox_fmt(self, dest_fmt):
        assert dest_fmt in ["LTRB", "LTWH", "XYWH"]
        self.bbox_format = dest_fmt


class BBOXES(object):
    def __init__(self, **kwargs):
        # Initialize Placeholder List for BBOXES
        self.bboxes = []

        # Get List of BBOX Objects
        bboxes = kwargs.get("bboxes")
        if isinstance(bboxes, (list, tuple)):
            for bbox in bboxes:
                assert isinstance(bbox, BBOX)
                self.bboxes.append(bbox)
        else:
            if bboxes is not None:
                raise NotImplementedError()

        # Initialize Iteration Counter
        self.__iter_counter = 0

    def __add__(self, other):
        assert isinstance(other, (BBOX, BBOXES))
        if isinstance(other, BBOX):
            self.bboxes.append(other)
        else:
            for other_bbox in other:
                self.bboxes.append(other_bbox)
        return self

    def __len__(self):
        return len(self.bboxes)

    def __iter__(self):
        return self

    def next(self):
        try:
            return_bbox = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return return_bbox

    def __getitem__(self, idx):
        return self.bboxes[idx]

    def append(self, bbox):
        self.bboxes.append(bbox)

    def check_format_integrity(self):
        fmt_list = [self.bboxes[idx].bbox_format for idx in range(len(self))]
        fmt_net_list = list(set(fmt_list))
        if len(fmt_net_list) == 0:
            raise AssertionError("BBOXES Data is Empty")
        elif len(fmt_net_list) > 1:
            raise AssertionError("Formats of BBOXES do not match...!")

    def get_ious(self, **kwargs):
        bbox, bboxes = kwargs.get("bbox"), kwargs.get("bboxes")
        assert np.logical_xor(bbox is None, bboxes is None)
        if bbox is not None:
            assert isinstance(bbox, BBOX)

            # Prepare IOU Array of Same Length With BBOX Objects
            iou_array = np.empty(len(self.bboxes))

            # Iterate for Self BBOXES, Calculate IOU
            for self_idx, self_bbox in enumerate(self):
                iou_array[self_idx] = bbox.get_iou(self_bbox)

        elif bboxes is not None:
            assert isinstance(bboxes, BBOXES)

            # Prepare IOU Array of Row: self-bboxes and Column: other-bboxes
            iou_array = np.empty((len(self.bboxes), len(bboxes)))

            # Iterate for BBOXES objects
            for self_idx, self_bbox in enumerate(self):
                for other_idx, other_bbox in enumerate(bboxes):
                    iou_array[self_idx, other_idx] = other_bbox.get_iou(self_bbox)

        else:
            raise AssertionError()

        return iou_array


if __name__ == "__main__":
    pass
