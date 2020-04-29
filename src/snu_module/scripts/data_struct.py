"""
SNU Integrated Module v2.5
  - Generic Data Structure Definition File
"""
import copy
import numpy as np
import ros_utils


# Ordinary Struct
class STRUCT:
    def __init__(self):
        pass


# Sensor Image Frame Class
class sensor_frame(object):
    def __init__(self):
        self.raw, self.processed = None, None
        self.memo = None

    # Update Frame
    def update_raw_frame(self, frame):
        self.raw = frame

    # Update Processed Frame
    def update_processed_frame(self, proc_frame):
        self.processed = proc_frame

    # Set Memo
    def set_memo(self, memo):
        if memo is not None:
            if type(memo) is not str:
                assert 0, "memo must be <string> format!"
            if len(memo) > 100:
                print("[WARNING] Memo Length is : %5d" %(len(memo)))
        self.memo = memo


# Sensor Parameter Class for Static Camera
class static_cam_params(object):
    def __init__(self, agent_name=None, parameter_precision=np.float32):
        # Set Camera Name
        self.agent_name = agent_name

        # Set Intrinsic-related Variables
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        self.w = None

        # Set Translation-related Variables (world-coordinate translation)
        self.x, self.y, self.z = None, None, None

        # Set Pan(yaw) / Tilt(pitch) / Roll Variables
        self.pan, self.tilt, self.roll = None, None, None

        # Set Camera Parameter Matrices
        self.param_precision = parameter_precision
        self.intrinsic_matrix, self.extrinsic_matrix = None, None

        self.intrinsic_matrix = np.zeros((3, 4), dtype=parameter_precision)
        self.extrinsic_matrix = np.zeros((4, 4), dtype=parameter_precision)

    # Update Agent Name
    def update_agent_name(self, agent_name):
        self.agent_name = agent_name

    # Update Camera Parameter Variables
    def update_params(self, param_array):
        # Intrinsic-related
        self.fx, self.fy, self.cx, self.cy = \
            param_array[0], param_array[1], param_array[2], param_array[3]
        self.w = param_array[4]

        # Translation-related (world coordinate)
        self.x, self.y, self.z = param_array[5], param_array[6], param_array[7]

        # Pan / Tilt / Roll
        self.pan, self.tilt, self.roll = param_array[8], param_array[9], param_array[10]

    # Convert PTR to Rotation Matrix
    def convert_ptr_to_rotation(self):
        r11 = np.sin(self.pan) * np.cos(self.roll) - np.cos(self.pan) * np.sin(self.tilt) * np.sin(self.roll)
        r12 = -np.cos(self.pan) * np.cos(self.roll) - np.sin(self.pan) * np.sin(self.tilt) * np.sin(self.roll)
        r13 = np.cos(self.tilt) * np.sin(self.roll)
        r21 = np.sin(self.pan) * np.sin(self.roll) + np.sin(self.tilt) * np.cos(self.pan) * np.cos(self.roll)
        r22 = -np.cos(self.pan) * np.sin(self.roll) + np.sin(self.tilt) * np.sin(self.pan) * np.cos(self.roll)
        r23 = -np.cos(self.tilt) * np.cos(self.roll)
        r31 = np.cos(self.tilt) * np.cos(self.pan)
        r32 = np.cos(self.tilt) * np.sin(self.pan)
        r33 = np.sin(self.tilt)

        rotation_matrix = np.array([[r11, r12, r13],
                                    [r21, r22, r23],
                                    [r31, r32, r33]], dtype=self.param_precision)
        return rotation_matrix

    # Update Camera Parameters (intrinsic / extrinsic)
    def get_camera_params(self):
        # < 3 x 4 >
        self.intrinsic_matrix = np.array([[self.fx, self.w, self.cx, 0],
                                          [0, self.fy, self.cy, 0],
                                          [0, 0, 1, 0]], dtype=self.param_precision)

        rotation_matrix = self.convert_ptr_to_rotation()
        translation_vector = np.matmul(
            rotation_matrix,
            np.array([self.x, self.y, self.z], dtype=self.param_precision).reshape((3, 1))
        )

        # < 4 x 4 >
        self.extrinsic_matrix = np.block(
            [np.vstack((rotation_matrix, np.zeros((1, 3)))), np.append(translation_vector, 1).reshape(-1, 1)]
        )


# Image Struct Class
class image_struct(object):
    def __init__(self, modal, agent_type, agent_name=None, is_ros_switch=False):
        """
        :param modal: modality of the sensor image input.
        :param agent_type: "static", if the sensor equipment is non-moving. "dynamic" if else.
        :param agent_name: The name of the multimodal sensor agent.
        :param is_ros_switch: "True", if the sensor is transmitted via ROS embedding. "False", otherwise.
        """
        # Sensor Modal Type
        self.modal = modal

        # Agent Type and Name
        self.agent_type = agent_type.lower()
        self.agent_name = agent_name

        # Is ROS embedded?
        self.is_ros_switch = is_ros_switch

        # Frame Index Count
        self.fidx = None

        # Sensor Image Frame Variables
        self.frame = sensor_frame()

        # Image Struct Dictionary
        # (e.g. make dictionary of something like "clipping distance", etc.)
        self.info_dict = {}

        # Sensor Image Seqstamp Initialization
        # (if is_ros_switch==False, then <seqstamp> is the time in "seconds"
        if is_ros_switch is False:
            self.prev_seqstamp, self.seqstamp = None, None
        else:
            self.prev_seqstamp = ros_utils.seqstamp(modal)
            self.seqstamp = ros_utils.seqstamp(modal)

        # Sensor Parameters (e.g. Intrinsics, Calibration Parameters, etc.)
        if agent_type.lower() == "static":
            self.sensor_params = static_cam_params()
            if agent_name is not None:
                self.sensor_params.update_agent_name(agent_name)
        elif agent_type.lower() == "dynamic":
            self.sensor_params = {}
        else:
            self.sensor_params = None

    # Set Dictionary
    def set_image_struct_info(self, key, value):
        if type(key) is not str:
            assert 0, "key must be <string> format!"
        self.info_dict[key] = value

    # Update Raw Frame
    def update_raw_frame(self, frame, fidx, seqstamp):
        # Update Raw Frame
        self.frame.update_raw_frame(frame)

        # Update Frame Index Count
        self.fidx = fidx

        # Update Seqstamp
        if self.is_ros_switch is True:
            if type(seqstamp).__name__ is not "seqstamp":
                assert 0, "input method <seqstamp> type needs to be verified!"
            self.prev_seqstamp = copy.deepcopy(self.seqstamp)
            self.seqstamp.update(seqstamp.seq, seqstamp.timestamp)
        else:
            if seqstamp is not None:
                if type(seqstamp).__name__ not in ["int", "float"]:
                    assert 0, "input method <seqstamp> type needs to be verified!"
                self.prev_seqstamp = copy.copy(self.seqstamp)
                self.seqstamp = seqstamp
            else:
                self.prev_seqstamp = copy.copy(self.seqstamp)
                self.seqstamp = None

    # Update Processed Frame
    def update_processed_frame(self, frame):
        self.frame.update_processed_frame(frame)

    # Set Frame Memo
    def set_frame_memo(self, memo):
        self.frame.set_memo(memo)

    # Update Sensor(camera) Parameters
    def update_sensor_params(self, param_input):
        """
        :param param_input: Type-ambiguous input method. Differs to the agent type of the class
            (1) - for the static method, input method is the sensor parameter numpy array from the YAML file
            (2) - for the dynamic method, the input method is a python dictionary which contains sensor parameters
        """
        if self.is_ros_switch is False:
            # Implement it if possible later on....!!!
            pass
        else:
            if self.agent_type == "static":
                if type(param_input).__name__ is not "ndarray":
                    assert 0, "input method <param_input> type needs to be 'ndarray'!"
                self.sensor_params.update(param_input)
            elif self.agent_type == "dynamic":
                if type(param_input) is not dict:
                    assert 0, "input method <param_input> type needs to be 'dict'!"
                self.sensor_params.update(param_input)
            else:
                assert 0, "To-Be-Implemented!"

    #



    # Seqstamp Difference
    def seqstamp_interval(self):
        if self.is_ros_switch is False:
            return self.seqstamp - self.prev_seqstamp
        else:
            if self.seqstamp.seq is not None and self.prev_seqstamp.seq is not None:
                dseq = self.seqstamp.seq - self.prev_seqstamp.seq
            else:
                dseq = None

            seqstamp_time, prev_seqstamp_time = self.seqstamp.get_time(), self.prev_seqstamp.get_time()
            if seqstamp_time is not None and prev_seqstamp_time is not None:
                dtime = seqstamp_time - prev_seqstamp_time
            else:
                dtime = None

            return dseq, dtime











