"""
SNU Integrated Module v5.0
  - Code which defines Coordinate Base Block for Object Tracking

"""
import numpy as np


class BBOX(object):
    def __init__(self, **kwargs):
        # Define Width and Height
        self.width, self.height = None, None

        # Designate BBOX Format
        format = kwargs.get("bbox_format")
        self.format = format

    def get_size(self):
        assert self.width is not None and self.height is not None
        return self.width * self.height

    def get_diagonal_length(self):
        assert self.width is not None and self.height is not None
        return np.sqrt(self.width ** 2 + self.height ** 2)

    def get_iou(self, other):
        raise NotImplementedError()


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
            for other_bbox in other:
                self.bboxes.append(other_bbox)
        else:
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
