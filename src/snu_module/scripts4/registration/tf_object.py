"""
SNU Integrated Module v5.0
    - TF Object Python Script for LiDAR-(RGB/Thermal) Registration

"""
import os
import yaml
import numpy as np


# TF Transform Class
class TF_TRANSFORM(object):
    def __init__(self, opts):

        # Get TF Transform File Path
        tf_filepath = os.path.join(
            os.path.dirname(__file__), "params",
            opts.agent_type, "{:02d}.yaml".format(int(opts.agent_id))
        )

        # Agent Type and ID
        self.__agent_type = opts.agent_type
        self.__agent_id = opts.agent_id

        # Read YAML File and Load Data
        with open(tf_filepath, "r") as stream:
            tf_data = yaml.safe_load(stream=stream)
        self.__tf_data = tf_data

    def get_transform(self, src_sensor, dest_sensor):
        # Get Source Sensor Transform
        tf_params = self.__tf_data[src_sensor][dest_sensor]

        # Rotation Matrix
        _r = tf_params["rotation_matrix"]
        rotation_matrix = np.array([
            [_r["r11"], _r["r12"], _r["r13"]],
            [_r["r21"], _r["r22"], _r["r23"]],
            [_r["r31"], _r["r32"], _r["r33"]],
        ])

        # Translation Vector
        _t = tf_params["translation"]
        translation_vector = np.array([
            _t["t1"], _t["t2"], _t["t3"]
        ]).reshape(3, 1)

        return rotation_matrix, translation_vector


if __name__ == "__main__":
    class _opts(object):
        def __init__(self):
            self.agent_type = "dynamic"
            self.agent_id = "01"

    opts = _opts()
    test_obj = TF_TRANSFORM(opts=opts)
    test_obj.get_transform(src_sensor="lidar", dest_sensor="thermal")

    test_yaml_path = os.path.join(
        os.path.dirname(__file__), "params", "dynamic", "01.yaml")

    # Read YAML file
    with open(test_yaml_path, "r") as stream:
        data_loaded = yaml.safe_load(stream=stream)
