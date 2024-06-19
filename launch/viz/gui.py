import sys
from PySide6 import QtWidgets
import argparse

import os
import logging
import torch
import numpy as np
import cv2
import itertools
import pickle
import json

from config_reg import ConfigRegistry, ConfigEntrySource, ConfigEntryCallback
from config_reg import ConfigEntryCommandlineBoolPattern, ConfigEntryCommandlineSeqPattern
from config_reg.callback import abspath_callback
from oakink2_preview.util.upkeep.opt import argdict_to_string
from oakink2_preview.util.console_io import suppress_trimesh_logging
from oakink2_preview.util.upkeep import log as log_upkeep
from oakink2_preview.viz.gui import MainWindow


PROG = os.path.splitext(os.path.basename(__file__))[0]
PARAM_PREFIX = "viz"
THIS_FILE = os.path.normcase(os.path.normpath(__file__))
THIS_DIR = os.path.dirname(THIS_FILE)
CURR_WORKING_DIR = os.getcwd()

# global vars
_logger = logging.getLogger(__name__)


def reg_entry(config_reg: ConfigRegistry):
    config_reg.register(
        "stream_prefix",
        prefix=PARAM_PREFIX,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
    )

    config_reg.register(
        "anno_prefix",
        prefix=PARAM_PREFIX,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
    )

    config_reg.register(
        "object_prefix",
        prefix=PARAM_PREFIX,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
    )

    config_reg.register(
        "program_prefix",
        prefix=PARAM_PREFIX,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
    )


def reg_extract(config_reg: ConfigRegistry):
    cfg = config_reg.select(PARAM_PREFIX)
    return cfg


def main():
    # region: program setup
    log_upkeep.log_init()
    log_upkeep.enable_console()

    config_reg = ConfigRegistry(prog=PROG)
    reg_entry(config_reg)

    parser = argparse.ArgumentParser(prog=PROG)
    config_reg.hook(parser)
    config_reg.parse(parser)

    run_cfg = reg_extract(config_reg)

    _logger.info("run_cfg: %s", argdict_to_string(run_cfg))

    suppress_trimesh_logging()
    # endregion

    # region: prepare app
    app = QtWidgets.QApplication([])

    widget = MainWindow(
        stream_prefix=run_cfg["stream_prefix"],
        anno_prefix=run_cfg["anno_prefix"],
        object_prefix=run_cfg["object_prefix"],
        program_prefix=run_cfg["program_prefix"],
    )
    widget.showMaximized()

    sys.exit(app.exec())
    # endregion


if __name__ == "__main__":
    main()
