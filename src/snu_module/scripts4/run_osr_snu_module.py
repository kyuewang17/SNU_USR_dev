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
import logging

from utils.loader import set_logger, argument_parser, cfg_loader, load_options


# Module Loader
def load_snu_module(logger, opts):
    # Import SNU Module, Initialize, and Return
    if opts.env_type in ["bag", "static", "dynamic"]:
        from ros__run_snu_module import snu_module
        return snu_module(logger=logger, opts=opts)
    elif opts.env_type == "imseq":
        from imseq__run_snu_module import snu_module
        raise NotImplementedError()
    else:
        raise AssertionError()


def main():
    pass


if __name__ == "__main__":
    # Set Logger
    logger = set_logger(logging_level=logging.INFO)

    # Argument Parser
    args = argument_parser(
        logger=logger,
        script_name=os.path.basename(__file__),
        dev_version=4.5
    )

    # Load Configuration from File
    cfg = cfg_loader(logger=logger, args=args)

    # Load Options
    opts = load_options(logger=logger, args=args, cfg=cfg)
    opts.visualization.correct_flag_options()

    # Load SNU Module
    snu_usr = load_snu_module(logger=logger, opts=opts)

    # Run SNU Module
    snu_usr(module_name="snu_module")
