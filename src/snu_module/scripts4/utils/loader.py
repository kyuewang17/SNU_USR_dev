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
def argument_parser(logger, script_name, dev_version=None, mode_selection=None):
    # Assertion
    assert isinstance(script_name, str), "Argument 'script_name' should be a <str> type...!"

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
        agent_type, agent_id = agent_id_file.split("-")[0], int(agent_id_file.split("-")[1])
        if agent_type.lower() not in ["static", "dynamic", "fixed", "moving"]:
            raise AssertionError("Agent Identification File [{}] is Erroneous...!".format(agent_id_file))
        else:
            agent_type = "static" if agent_type.lower() in ["static", "fixed"] else "dynamic"

    # Sleep
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
        prog=script_name, description="SNU Integrated Algorithm"
    )
    if dev_version is not None:
        parser.add_argument(
            "--dev-version", "-V", default=dev_version,
            help="Integrated Algorithm Development Version"
        )
    else:
        parser.add_argument(
            "--dev-version", "-V",
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
    rosbag_parser.add_argument("--arg-opts", "-A", default="bag", help="Argument Option - ROS Bag File")
    """"""

    # Image Sequences
    """ Create Sub-Parsing Command for Testing this Code on Image Sequences
        (generated from ROS bag file, using 'bag2seq.py') """
    imseq_parser = subparser.add_parser(
        "imseq", help="for executing this code with Image Sequences, generated from the given 'bag2seq.py' python package"
    )
    if mode_selection is not None:
        imseq_parser.add_argument(
            "--imseq-base-path", "-I", type=str,
            default="/home/kyle/PycharmProjects/SNU_USR_dev/src/snu_module/bag2imseq/_cvt_data__[lidar_error]",
            help="Image Sequence Base Path, which is generated from ROS bag file using the given 'bag2seq.py'"
        )
    else:
        imseq_parser.add_argument(
            "--imseq-base-path", "-I", type=str,
            help="Image Sequence Base Path, which is generated from ROS bag file using the given 'bag2seq.py'"
        )
    imseq_parser.add_argument("--arg-opts", "-A", default="imseq", help="Argument Option - Image Sequence")
    """"""

    # Agents
    """ Create Sub-Parsing Command for Testing this Code on Outdoor Surveillance Agents """
    agent_parser = subparser.add_parser(
            "agent", help="for executing this code on Outdoor Surveillance Robot Agents"
    )
    if is_agent_flag is True:
        agent_parser.add_argument("--agent-type", "-T", default=agent_type)
        agent_parser.add_argument("--agent-id", "-I", type=int, default=agent_id)
    else:
        agent_parser.add_argument(
            "--agent-type", "-T", type=str, choices=["static", "dynamic"],
            help="Agent Type (choose btw 'static' and 'dynamic')"
        )
        agent_parser.add_argument("--agent-id", "-I", type=int, help="Agent ID Number")
    agent_parser.add_argument("--arg-opts", "-A", default="agent", help="Argument Option - Agent Robot")
    """"""

    # Parse Arguments and Return
    if mode_selection is None:
        args = parser.parse_args()
    else:
        # Selective Parse Arguments
        args = parser.parse_args(["{}".format(mode_selection)])
    # args = parser.parse_args()
    return args


# Configuration File Loader
def cfg_loader(logger, args):
    # Load Configuration File, regarding the input arguments
    if args.arg_opts == "bag":
        from configs.rosbag.config_rosbag import cfg
        cfg_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "rosbag", args.cfg_file_name)
        if os.path.isfile(cfg_file_path) is False:
            logger.warn("Could Not Find the Configuration File for Current Playing Bag File...!")
            logger.warn("Loading Base Configuration File Instead...!")
            cfg_file_path = os.path.join(os.path.dirname(cfg_file_path), "base.yaml")
            time.sleep(0.5)
        else:
            logger.info("Loading Configuration File from {}".format(cfg_file_path))
    elif args.arg_opts == "imseq":
        from configs.imseq.config_imseq import cfg
        imseq_folder_name = args.imseq_base_path.split("/")[-1]
        cfg_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "imseq", imseq_folder_name)
        if os.path.isfile(cfg_file_path) is False:
            logger.warn("Could Not Find the Configuration File for Current Targeted Image Sequence...!")
            logger.warn("Loading Base Configuration File Instead...!")
            cfg_file_path = os.path.join(os.path.dirname(cfg_file_path), "base.yaml")
            time.sleep(0.5)
        else:
            logger.info("Loading Configuration File from {}".format(cfg_file_path))
    elif args.arg_opts == "agent":
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
    opts = options.snu_option_class(cfg=cfg, dev_version=args.dev_version)
    return opts


if __name__ == "__main__":
    pass
