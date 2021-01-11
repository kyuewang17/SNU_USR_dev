#!/usr/bin/env python
"""
- Object Class Python Script for Agent Environment (Static / Dynamic)

- Acts as the Class that Manages the Entire SNU Integrated Algorithms for Agents
  (improved from 'backbone' class from SNU-v4.5)

"""
import cv2
import numpy as np

import SENSORS


class BASE_AGENT_OBJ(object):
    def __init__(self, agent_sensor_modal_dict, agent_id, agent_type, agent_loc="pohang"):
        assert isinstance(agent_id, int)
        assert isinstance(agent_sensor_modal_dict, dict)

        # Agent INFO
        self.ID = agent_id
        self.TYPE = agent_type

        # Agent Sensor Modal Dict (True if exists, False if not exist)
        for modal, existence in agent_sensor_modal_dict.items():
            if existence is True:
                setattr(self, modal, True)
            else:
                setattr(self, modal, False)

        """ Private Attributes """
        self.__LOC = agent_loc

    def __repr__(self):
        return "[{}]-{}-{:02d}".format(
            self.__LOC.upper(), self.TYPE.upper(), self.ID)

    def update_agent_loc(self, agent_loc):
        self.__LOC = agent_loc


class STATIC_AGENT_OBJ(BASE_AGENT_OBJ):
    def __init__(self, agent_sensor_modal_dict, agent_id, agent_loc="pohang"):
        super(STATIC_AGENT_OBJ, self).__init__(agent_sensor_modal_dict, agent_id, agent_type="static", agent_loc=agent_loc)


class DYNAMIC_AGENT_OBJ(BASE_AGENT_OBJ):
    def __init__(self, agent_sensor_modal_dict, agent_id, agent_loc="pohang"):
        super(DYNAMIC_AGENT_OBJ, self).__init__(agent_sensor_modal_dict, agent_id, agent_type="dynamic", agent_loc=agent_loc)


if __name__ == "__main__":
    pass
