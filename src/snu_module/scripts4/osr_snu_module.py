#!/usr/bin/env python
"""
Outdoor Surveillance Robot SNU Module

- Algorithm Switcher

    - Switches algorithm of below types

        [1] ROS-embedded SNU Module
        [2] SNU Module which runs on Image Sequences


- TBA

"""
import os
import argparse
import time
import yaml
import rospy
import logging
import tf2_ros
import numpy as np

# import snu_visualizer
# from options_v4_5 import snu_option_class as options
# from utils.ros.base import backbone
# from utils.ros.sensors import snu_SyncSubscriber
# from snu_algorithms_v4_5 import snu_algorithms
# from utils.profiling import Timer
#
# from module_detection import load_model as load_det_model
# from module_action import load_model as load_acl_model
#
# from config import cfg


# Set Logger
def set_logger(logging_level=logging.INFO):
    # Define Logger
    _logger = logging.getLogger()

    # Set Logger Display Level
    _logger.setLevel(level=logging_level)

    # Set Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("[%(levelname)s] | %(asctime)s : %(message)s")
    )
    _logger.addHandler(stream_handler)

    return _logger


# Argument Parser
def argument_parser(logger, dev_version=4.5):
    # Attempt to Find the Agent Identification File from Computer
    agent_id_file_base_path = "/agent"
    agent_id_file_list = \
        os.listdir(agent_id_file_base_path) if os.path.isdir(agent_id_file_base_path) else []
    if len(agent_id_file_list) == 0:
        logger.info("Agent Identification File Not Found...!")
        is_agent_flag = False
        agent_type, agent_id = None, None
    elif len(agent_id_file_list) > 1:
        raise AssertionError("More than 1 Agent Identification Files...!")
    else:
        logger.info("Agent Identification File Found at PATH: {}".format(agent_id_file_base_path))
        is_agent_flag = True
        agent_id_file = agent_id_file_list[0]
        agent_type, agent_id = agent_id_file.split("_")[0], int(agent_id_file.split("_")[1])
        if agent_type.lower() not in ["static", "dynamic", "fixed", "moving"]:
            raise AssertionError("Agent Identification File [{}] is Erroneous...!".format(agent_id_file))
        else:
            agent_type = "static" if agent_type.lower() in ["static", "fixed"] else "dynamic"
    time.sleep(0.5)

    # Declare Argument Parser
    parser = argparse.ArgumentParser(
        prog="osr_snu_module.py",
        description="SNU Integrated Algorithm (DEV-[{:f}])".format(dev_version)
    )
    parser.add_argument(
        "--dev-version", "-V", default=dev_version,
        help="Integrated Algorithm Development Version"
    )

    # Add Sub-Parser
    subparser = parser.add_subparsers(help="Sub-Parser Commands")

    # ROS Bag Files
    """ Create Sub-Parsing Command for Testing this Code on ROS Bag File """
    rosbag_parser = subparser.add_parser(
        "bag", help="for executing this code with ROS bag file"
    )
    rosbag_parser.add_argument(
        "--cfg-file-name", "-C", type=str,
        default="base.yaml",
        # default=os.path.join(os.path.dirname(__file__), "config", "rosbag", "base.yaml"),
        help="Configuration File Name, which matches the currently playing ROS bag file"
    )
    rosbag_parser.add_argument("--arg-opts", "-A", default="rosbag", help="Argument Option - ROS Bag File")
    """"""

    # Image Sequences
    """ Create Sub-Parsing Command for Testing this Code on Image Sequences
        (generated from ROS bag file, using 'bag2seq.py') """
    imseq_parser = subparser.add_parser(
        "imseq", help="for executing this code with Image Sequences, generated from the given 'bag2seq.py' python package"
    )
    imseq_parser.add_argument(
        "--imseq-base-path", "-I", type=str,
        help="Image Sequence Base Path, which is generated from ROS bag file using the given 'bag2seq.py'"
    )
    imseq_parser.add_argument(
        "--cfg-file-name", "-C", type=str,
        default="base.yaml",
        # default=os.path.join(os.path.dirname(__file__), "config", "imseq", "base.yaml"),
        help="Configuration File Name, which matches the designated Image Sequence Folder Name"
    )
    imseq_parser.add_argument("--arg-opts", "-A", default="imseq", help="Argument Option - Image Sequence")
    """"""

    # Agents
    """ Create Sub-Parsing Command for Testing this Code on Outdoor Surveillance Agents """
    agent_parser = subparser.add_parser(
            "agent", help="for executing this code on Outdoor Surveillance Robot Agents"
    )
    if is_agent_flag is True:
        agent_parser.add_argument("--agent-type", default=agent_type)
        agent_parser.add_argument("--agent-id", default=agent_id)
    else:
        agent_parser.add_argument(
            "--agent_type", "-T", type=str, choices=["static", "dynamic"],
            help="Agent Type (choose btw 'static' and 'dynamic')"
        )
        agent_parser.add_argument("--agent-id", "-I", type=int, help="Agent ID Number")
    agent_parser.add_argument("--arg-opts", "-A", default="agents", help="Argument Option - Agent Robot")
    """"""

    # Parse Arguments and Return
    args = parser.parse_args()
    return args


def cfg_loader(logger, args):
    # Load Configuration File, regarding the input arguments
    if args.arg_opts == "rosbag":
        from configs.rosbag.config_rosbag import cfg
        cfg_file_path = os.path.join(os.path.dirname(__file__), "config", "rosbag", args.cfg_file_name)
        logger.info("Loading Configuration File from {}".format(cfg_file_path))
    elif args.arg_opts == "imseq":
        from configs.imseq.config_imseq import cfg
        cfg_file_path = os.path.join(os.path.dirname(__file__), "config", "imseq", args.cfg_file_name)
        logger.info("Loading Configuration File from {}".format(cfg_file_path))
    elif args.arg_opts == "agents":
        from configs.agents.config_agents import cfg
        cfg_file_path = os.path.join(
            os.path.dirname(__file__), "config", "imseq", args.agent_type, "{:02d}.yaml".format(args.agent_id)
        )
        logger.info("Loading Configuration File from {}".format(cfg_file_path))
    else:
        raise NotImplementedError("Current Argument Option [{}] not Defined...!".format(args.arg_opts))
    time.sleep(0.5)
    cfg.merge_from_file(cfg_filename=cfg_file_path)
    return cfg










def main():
    pass





if __name__ == "__main__":
    # Set Logger
    logger = set_logger(logging_level=logging.INFO)

    # Argument Parser
    args = argument_parser(logger=logger, dev_version=4.5)

    # Load Configuration from File
    cfg = cfg_loader(logger=logger, args=args)



    pass