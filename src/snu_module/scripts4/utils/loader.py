"""
SNU Integrated Module v5.0

    - Algorithm Loading Functions

"""
import os
import logging
import time
import argparse
import importlib


# Set Logger
def set_logger(logging_level=logging.INFO):
    # Define Logger
    logger = logging.getLogger()

    # Set Logger Display Level
    logger.setLevel(level=logging_level)

    # Set Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("[%(levelname)s] | %(asctime)s : %(message)s")
    )
    logger.addHandler(stream_handler)

    return logger


# Argument Parser
def argument_parser(logger, dev_version=4.5, mode_selection=None):
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

    # Detect Mode Selection
    if mode_selection is None:
        logger.info("Searching Argument Declaration...!")
    else:
        assert isinstance(mode_selection, str)
        assert mode_selection in ["bag", "imseq", "agent"], "Manual Mode [{}] is Undefined...!".format(mode_selection)
        logger.info("Manual Mode [{}] Selected...!".format(mode_selection))

    # Declare Argument Parser
    parser = argparse.ArgumentParser(
        prog="osr_snu_module.py", description="SNU Integrated Algorithm"
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
        "--cfg-file-name", "-C", type=str, default="base.yaml",
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
        "--cfg-file-name", "-C", type=str, default="base.yaml",
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
    if mode_selection is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(["{}".format(mode_selection)])
    return args


# Configuration File Loader
def cfg_loader(logger, args):
    # Load Configuration File, regarding the input arguments
    if args.arg_opts == "rosbag":
        from configs.rosbag.config_rosbag import cfg
        cfg_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "rosbag", args.cfg_file_name)
        if os.path.isfile(cfg_file_path) is False:
            logger.warn("")
        else:
            logger.info("Loading Configuration File from {}".format(cfg_file_path))
    elif args.arg_opts == "imseq":
        from configs.imseq.config_imseq import cfg
        cfg_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "imseq", args.cfg_file_name)
        logger.info("Loading Configuration File from {}".format(cfg_file_path))
        raise NotImplementedError("Image Sequence-based Code Not Implemented Yet...!")
    elif args.arg_opts == "agents":
        from configs.agents.config_agents import cfg
        cfg_file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "configs", "agents",
            args.agent_type, "{:02d}.yaml".format(args.agent_id)
        )
        logger.info("Loading Configuration File from {}".format(cfg_file_path))
    else:
        raise NotImplementedError("Current Argument Option [{}] not Defined...!".format(args.arg_opts))
    time.sleep(0.5)
    cfg.merge_from_file(cfg_filename=cfg_file_path)
    return cfg


# Load Options
def load_options(logger, args, cfg):
    # Get Module Version and Selected Base Path
    dev_version = str(args.dev_version)
    dev_main_version, dev_sub_version = dev_version.split(".")[0], dev_version.split(".")[-1]
    module_version_base_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "module_lib", "v{}_{}".format(dev_main_version, dev_sub_version)
    )
    if os.path.isdir(module_version_base_path) is False:
        raise AssertionError("Module Version [v{}] NOT Found...!".format(args.dev_version))
    else:
        logger.info("Module Version [v{}] Targeted...!".format(args.dev_version))
    time.sleep(0.5)

    # Select Option based on Module Version
    options = importlib.import_module("module_lib.v{}_{}.options".format(dev_main_version, dev_sub_version))
    opts = options.snu_option_class(cfg=cfg, run_type=args.arg_opts, dev_version=args.dev_version)
    return opts


if __name__ == "__main__":
    pass
