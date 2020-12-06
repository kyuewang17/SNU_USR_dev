#!/usr/bin/env python
"""
- Object Class Python Script for Agent Environment (Static / Dynamic)

- Acts as the Class that Manages the Entire SNU Integrated Algorithms for Agents
  (improved from 'backbone' class from SNU-v4.5)

"""
import cv2
import numpy as np

import SENSORS


class MODALS_OBJ(object):
    def __init__(self, modal_info_dict):
        assert isinstance(modal_info_dict, dict)
        """
        'modal_info_dict' example

        - modal_info_dict = {
            'color': True,
            'depth': False,
            'infrared': True,
            'thermal': True,
            'nightvision': False,
            'lidar': True
        }

        """
        for modal, is_modal in modal_info_dict.items():
            if is_modal is True:
                if modal == "color":
                    modal_obj = SENSORS.COLOR_SENSOR_OBJ()
                elif modal == "depth":
                    modal_obj = SENSORS.DEPTH_SENSOR_OBJ()
                elif modal == "infrared":
                    modal_obj = SENSORS.INFRARED_SENSOR_OBJ()
                elif modal == "thermal":
                    modal_obj = SENSORS.THERMAL_SENSOR_OBJ()
                elif modal == "nightvision":
                    modal_obj = SENSORS.NIGHTVISION_SENSOR_OBJ()
                elif modal == "lidar":
                    modal_obj = SENSORS.LIDAR_SENSOR_OBJ()
                else:
                    raise NotImplementedError()
            else:
                modal_obj = None

            # Set Field
            setattr(self, modal, modal_obj)

    def get_available_modals(self):
        available_modals = []
        for modal, dict_value in vars(self).items():
            if dict_value is not None:
                available_modals.append(modal)
        return available_modals

    def get_all_modal_timestamps(self):
        timestamp_dict = {}
        for modal, modal_obj in vars(self).items():
            if modal_obj is not None:
                timestamp_dict[modal] = modal_obj.get_timestamp()
            else:
                timestamp_dict[modal] = None
        return timestamp_dict

    def get_modal_timestamp(self, modal):
        modal_obj = getattr(self, modal)
        return modal_obj.get_timestamp()

    def get_modal_data(self, modal):
        modal_obj = getattr(self, modal)
        return modal_obj.get_data()

    def update_modal(self, modal, data, timestamp):
        modal_obj = getattr(self, modal)
        modal_obj.update(data, timestamp)


class BASE_AGENT_OBJ(object):
    def __init__(self, agent_type, agent_id, modal_info_dict):
        # Camera Type must be String, Camera ID must be Integer
        assert isinstance(agent_type, str), "Unknown Agent Type: {}... Must be String format".format(type(agent_type))
        assert isinstance(agent_id, int), "Agent ID Must be an Integer...!"

        # Agent Type (Static / Dynamic)
        self.AGENT_TYPE = agent_type

        # Agent ID
        self.ID = agent_id

        # Existing Multimodal Sensor List
        self.MODALS = MODALS_OBJ(modal_info_dict=modal_info_dict)

    def __repr__(self):
        return "{}-{:03d}".format(self.AGENT_TYPE, self.ID)

    def update_modal_sensor_list(self, modal_sensor_list):
        self.MODALS = modal_sensor_list

    def get_agent_info(self):
        return {"type": self.AGENT_TYPE, "id": self.ID}

    def check_for_modal(self, check_modal):
        available_modals = self.MODALS.get_available_modals()
        return True if check_modal in available_modals else False


class STATIC_AGENT_OBJ(BASE_AGENT_OBJ):
    def __init__(self, agent_id, modal_info_dict):
        super(STATIC_AGENT_OBJ, self).__init__("static", agent_id, modal_info_dict)


class DYNAMIC_AGENT_OBJ(BASE_AGENT_OBJ):
    def __init__(self, agent_id, modal_info_dict):
        super(DYNAMIC_AGENT_OBJ, self).__init__("dynamic", agent_id, modal_info_dict)



if __name__ == "__main__":
    test_dict = {
        'color': True,
        'depth': False,
        'infrared': True,
        'thermal': True,
        'nightvision': False,
        'lidar': True
    }
    TEST_STATIC_AGENT_OBJ = STATIC_AGENT_OBJ(agent_id=1, modal_info_dict=test_dict)
    pass
