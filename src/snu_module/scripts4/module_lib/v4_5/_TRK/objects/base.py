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
        # Initialize List Placeholder for Object Instances
        self.objects = []

        # Get List of Object Instances
        objects = kwargs.get("objects")
        if isinstance(objects, list):
            for obj in objects:
                assert isinstance(obj, object_instance)
                self.objects.append(obj)
        else:
            if objects is not None:
                raise NotImplementedError()

        # Iteration Counter
        self.__iteration_counter = 0

    def __len__(self):
        return len(self.objects)

    def __add__(self, other):
        assert isinstance(other, (object_instance, object_instances))
        if isinstance(other, object_instance):
            self.objects.append(other)
            self.arrange_objects(arrange_method="id")
        else:
            for other_object in other:
                self.objects.append(other_object)
            self.arrange_objects(arrange_method="id")
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

    def get_ids(self):
        return [self.objects[obj_idx].id for obj_idx in range(len(self.objects))]

    def get_labels(self):
        return [self.objects[obj_idx].label for obj_idx in range(len(self.objects))]

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

    def gather_objects(self, **kwargs):
        # Condition Dictionary
        condition_dict = {}

        # Accumulate Conditions
        id = kwargs.get("id", self.get_ids())
        if isinstance(id, (list, tuple)):
            assert set(id).issubset(set(self.get_ids()))
            condition_dict["id"] = list(id)
        else:
            assert isinstance(id, int) and id >= 0
            condition_dict["id"] = [id]

        label = kwargs.get("label", list(set(self.get_labels())))
        if isinstance(label, (list, tuple)):
            assert set(label).issubset(set(self.get_labels()))
            condition_dict["label"] = list(label)
        else:
            condition_dict["label"] = [label]

        fidx = kwargs.get("fidx")
        if fidx is None:
            min_fidx, max_fidx = None, None
            for obj in self:
                if min_fidx is None:
                    min_fidx, max_fidx = obj.frame_indices[0], obj.frame_indices[-1]
                else:
                    if min_fidx > obj.frame_indices[0]:
                        min_fidx = obj.frame_indices[0]
                    if max_fidx < obj.frame_indices[-1]:
                        max_fidx = obj.frame_indices[-1]
            condition_dict["fidx"] = [min_fidx, max_fidx]
        else:
            if isinstance(fidx, int) and fidx >= 0:
                condition_dict["fidx"] = [fidx]
            elif isinstance(fidx, (list, tuple)):
                condition_dict["fidx"] = fidx
            else:
                raise NotImplementedError()

        # For Conditions, selectively gather objects
        if id == self.get_ids() and label == list(set(self.get_labels())) and fidx is None:
            return self.objects
        else:
            selected_object_indices = []
            for obj_idx, obj in enumerate(self):
                decision_counter_flag = 0
                for method, condition in condition_dict.items():
                    if method == "fidx":
                        # Check Min-fidx
                        if condition[0] == -1 or obj.frame_indices[0] >= condition[0]:
                            min_fidx_flag = True
                        else:
                            min_fidx_flag = False

                        # Check Max-fidx
                        if condition[-1] == -1 or obj.frame_indices[-1] <= condition[-1]:
                            max_fidx_flag = True
                        else:
                            max_fidx_flag = False

                        if min_fidx_flag is True and max_fidx_flag is True:
                            decision_counter_flag += 1

                    elif method in ["id", "label"]:
                        if getattr(obj, method) in condition:
                            decision_counter_flag += 1

                    else:
                        raise NotImplementedError()

                # Decide to Append
                if decision_counter_flag == len(condition_dict):
                    selected_object_indices.append(obj_idx)

            # Return Selected Objects
            return [self.objects[idx] for idx in selected_object_indices]

    def associate(self, *args, **kwargs):
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
