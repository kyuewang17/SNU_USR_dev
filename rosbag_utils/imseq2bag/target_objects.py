#!/usr/bin/env python
"""


"""
import numpy as np


class BBOX(object):
    def __init__(self, **kwargs):
        # Get Coordinates
        self.lt_x = kwargs.get("LT_X")
        self.lt_y = kwargs.get("LT_Y")
        self.rb_x = kwargs.get("RB_X")
        self.rb_y = kwargs.get("RB_Y")

        # Iteration Counter
        self.__iter_counter = 0

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        var_list = [self.lt_x, self.lt_y, self.rb_x, self.rb_y]
        return var_list[idx]

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

    # def convert_coord_precision(self, precision=int):
    #     self.lt_x = precision(self.lt_x)
    #     self.lt_y = precision(self.lt_y)
    #     self.rb_x = precision(self.rb_x)
    #     self.rb_y = precision(self.rb_y)

    def numpify(self, type_conversion="ltrb"):
        assert type_conversion in ["ltrb", "ltwh", "xywh"]
        ret_arr = np.empty(len(self))
        for arr_idx in range(len(self)):
            if arr_idx == 0 or arr_idx == 1:
                if type_conversion in ["ltrb", "ltwh"]:
                    ret_arr[arr_idx] = self[arr_idx]
                else:
                    ret_arr[arr_idx] = np.floor((self[arr_idx] + self[arr_idx+2]) / 2.0)
            elif arr_idx == 2 or arr_idx == 3:
                if type_conversion == "ltrb":
                    ret_arr[arr_idx] = self[arr_idx]
                else:
                    ret_arr[arr_idx] = self[arr_idx] - self[arr_idx-2]
        return ret_arr

    def get_intersection_bbox(self, other):
        assert isinstance(other, (BBOX, np.ndarray))

        if isinstance(other, BBOX):
            # Get Min-Max Coordinates
            uu1 = np.maximum(self.lt_x, other.lt_x)
            vv1 = np.maximum(self.lt_y, other.lt_y)
            uu2 = np.minimum(self.rb_x, other.rb_x)
            vv2 = np.minimum(self.rb_y, other.rb_y)

        elif isinstance(other, np.ndarray):
            # Get Min-Max Coordinates
            uu1 = np.maximum(self.lt_x, other[0])
            vv1 = np.maximum(self.lt_y, other[1])
            uu2 = np.minimum(self.rb_x, other[2])
            vv2 = np.minimum(self.rb_y, other[3])

        else:
            raise NotImplementedError()

        # Initialize Common BBOX
        return BBOX(LT_X=uu1, LT_Y=vv1, RB_X=uu2, RB_Y=vv2)

    def get_iou(self, other):
        raise NotImplementedError()


class TARGET(object):
    def __init__(self, **kwargs):
        # ID Label
        id = kwargs.get("id")
        if id is not None:
            assert isinstance(id, int)
        self.id = id

        # BBOX ( Left-Top, Right-Bottom )
        bbox = kwargs.get("bbox")
        if bbox is not None:
            assert isinstance(bbox, BBOX)
        self.bbox = bbox

        # Class Label
        self.cls = kwargs.get("cls")

        # Pose Label
        self.pose = kwargs.get("pose") if self.cls == "Human" else None

        # Target Modality
        self.modal = kwargs.get("modal")

        # Initialize Target Timing String
        self.timing = {
            "date": kwargs.get("date"),
            "time": kwargs.get("time"),
            "fidx": kwargs.get("fidx")
        }

    def get_timing_string(self):
        return "{}_{}_{}".format(self.timing["date"], self.timing["time"], self.timing["fidx"])

    def get_bbox(self, fmt="LTRB"):
        assert fmt in ["LTRB", "XYWH", "LTWH"]
        _bbox = self.bbox.numpify()

        if fmt == "LTRB":
            return _bbox

        elif fmt == "LTWH":
            bbox = np.empty(4)
            bbox[0] = _bbox[0]
            bbox[1] = _bbox[1]
            bbox[2] = _bbox[2] - _bbox[0]
            bbox[3] = _bbox[3] - _bbox[1]
            return bbox

        else:
            bbox = np.empty(4)
            bbox[0] = np.floor((_bbox[0] + _bbox[2]) / 2.0)
            bbox[1] = np.floor((_bbox[1] + _bbox[3]) / 2.0)
            bbox[2] = _bbox[2] - _bbox[0]
            bbox[3] = _bbox[3] - _bbox[1]
            return bbox




if __name__ == "__main__":
    pass
