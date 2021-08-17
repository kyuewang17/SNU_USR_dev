"""
SNU Integrated Module v5.0
  - Code which defines Basic Object Class for Object Tracking

"""
import numpy as np
import copy


# Object Instance Class (Single)
class object_instance(object):
    def __init__(self, **kwargs):
        # Initialize Frame Index List that indicates the times that current instance existed
        init_fidx = kwargs.get("init_fidx")
        assert isinstance(init_fidx, int) and init_fidx >= 0
        self.frame_indices = [init_fidx]

        # Initialize Instance ID
        inst_id = kwargs.get("id")
        assert isinstance(inst_id, int) and inst_id >= 0
        self.id = inst_id

        # Initialize Instance Label
        inst_label = kwargs.get("label")
        assert inst_label is not None
        self.label = inst_label

    def __repr__(self):
        return "OBJECT ID - [{}]".format(self.id)

    def __add__(self, other):
        assert isinstance(other, (object_instance, object_instances))
        if isinstance(other, object_instance):
            return object_instances(objects=[self, other])
        else:
            return other + self

    def __eq__(self, other):
        assert isinstance(other, object_instance)
        assert self.label == other.label

        if self.id == other.id:
            return True
        else:
            return False

    def __len__(self):
        return len(self.frame_indices)

    def __iter__(self):
        return self

    def next(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __del__(self):
        pass

    def update(self, *args, **kwargs):
        raise NotImplementedError()


# Object Instance Classes
class object_instances(object):
    def __init__(self, **kwargs):
        # Initialize Object ID List
        self.object_ids = []

        # Initialize List Placeholder for Object Instances
        self.objects = []

        # Get List of Object Instances
        objects = kwargs.get("objects")
        if isinstance(objects, list):
            for obj in objects:
                assert isinstance(obj, object_instance)
                self.objects.append(obj)
                self.object_ids.append(obj.id)

            # Arrange Objects w.r.t. ID
            self.arrange_objects(arrange_method="id")

        else:
            if objects is not None:
                raise NotImplementedError()

        # Iteration Counter
        self.__iteration_counter = 0

    def __len__(self):
        return len(self.objects)

    def __add__(self, other):
        self.append(other=other)
        return self

    def __getitem__(self, idx):
        return self.objects[idx]

    def __iter__(self):
        return self

    def next(self):
        try:
            return_value = self[self.__iteration_counter]
        except IndexError:
            self.__iteration_counter = 0
            raise StopIteration
        self.__iteration_counter += 1
        return return_value

    def append(self, other):
        assert isinstance(other, (object_instance, object_instances))
        if isinstance(other, object_instance):
            self.objects.append(other)
            self.object_ids.append(other.id)
            self.arrange_objects(arrange_method="id")
        else:
            for other_object in other:
                self.objects.append(other_object)
                self.object_ids.append(other_object.id)
            self.arrange_objects(arrange_method="id")

    def get_ids(self):
        return self.object_ids

    def get_labels(self):
        return [self.objects[obj_idx].label for obj_idx in range(len(self.objects))]

    def get_object_with_id(self, id):
        if id not in self.object_ids:
            return None
        else:
            return self.objects[self.object_ids.index(id)]

    def arrange_objects(self, **kwargs):
        if len(self.objects) == 0:
            return

        # Get Arrange Method
        arrange_method = kwargs.get("arrange_method", "id")

        # Get Unsorted List w.r.t. Arrange Method
        if arrange_method == "id":
            unsorted_list = self.get_ids()
        elif arrange_method == "init_fidx":
            unsorted_list = [self.objects[obj_idx].frame_indices[0] for obj_idx in range(len(self.objects))]
        else:
            raise NotImplementedError()

        # Get Sorted Index
        sorted_index = np.argsort(unsorted_list)

        # Sort Objects
        sorted_objects = []
        for i, sidx in enumerate(sorted_index):
            sorted_objects.append(self.objects[sidx])
        self.objects = sorted_objects
        self.object_ids = [self.objects[obj_idx].id for obj_idx in range(len(self.objects))]

    def gather_objects(self, **kwargs):
        # Get Condition KWARGS
        id, label, fidx = kwargs.get("id"), kwargs.get("label"), kwargs.get("fidx")

        # Return All Objects when there are no conditions
        if id is None and label is None and fidx is None:
            return self.objects

        # Condition Dictionary (conditions are collected with logical "and" method)
        condition_dict = {}

        # Accumulate Conditions
        if id is not None:
            assert isinstance(id, (int, list, tuple))
            if isinstance(id, int):
                assert id >= 0
                condition_dict["id"] = id
            else:
                assert set(id).issubset(set(self.get_ids()))
                condition_dict["id"] = list(id)
        if label is not None:
            if isinstance(label, (list, tuple)):
                assert set(label).issubset(set(self.get_labels()))
                condition_dict["label"] = list(label)
            else:
                condition_dict["label"] = label
        if fidx is not None:
            if isinstance(fidx, int) and fidx >= 0:
                condition_dict["fidx"] = fidx
            elif isinstance(fidx, (list, tuple)):
                assert len(fidx) == 2 and fidx[0] <= fidx[1]
                condition_dict["fidx"] = list(fidx)
            else:
                raise NotImplementedError()

        # Collect Object Indices according to Conditions
        selected_object_indices = []
        for obj_idx, obj in enumerate(self):
            decision_counter_flag = 0
            for method, condition in condition_dict.items():
                # Frame Index Condition
                if method == "fidx":
                    if isinstance(condition, list):
                        min_fidx, max_fidx = fidx[0], fidx[1]
                        if obj.frame_indices[0] >= min_fidx and obj.frame_indices[-1] <= max_fidx:
                            decision_counter_flag += 1
                    else:
                        if fidx in obj.frame_indices:
                            decision_counter_flag += 1
                # ID Condition
                elif method in ["id", "label"]:
                    if isinstance(condition, list):
                        if getattr(obj, method) in condition:
                            decision_counter_flag += 1
                    else:
                        if getattr(obj, method) == condition:
                            decision_counter_flag += 1
                else:
                    raise NotImplementedError()

            # Decide to Append
            if decision_counter_flag == len(condition_dict):
                selected_object_indices.append(obj_idx)

        # Return Selected Objects
        return [self.objects[idx] for idx in selected_object_indices]

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, *args, **kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    obj_01 = object_instance(init_fidx=0, label=0, id=4)
    obj_02 = object_instance(init_fidx=3, label=1, id=1)
    obj_03 = object_instance(init_fidx=4, label=1, id=3)

    s1 = obj_01 + obj_02
    s11 = s1 + obj_03

    for s11_obj in s11:
        print(s11_obj.id)

    pass
