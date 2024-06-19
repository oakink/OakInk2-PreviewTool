import os
import logging

import ctypes
import sys
import io
import contextlib
import warnings
import pprint as pprint_module
import functools


def suppress_trimesh_logging():
    logger = logging.getLogger("trimesh")
    logger.setLevel(logging.ERROR)


def suppress_gym_logging():
    try:
        import gym

        gym.logger.set_level(logging.ERROR)
    except ImportError:
        pass


class RedirectStream(object):
    @staticmethod
    def _flush_c_stream(stream):
        streamname = stream.name[1:-1]
        libc = ctypes.CDLL(None)
        libc.fflush(ctypes.c_void_p.in_dll(libc, streamname))

    def __init__(self, stream=sys.stdout, file=os.devnull):
        self.stream = stream
        self.file = file

    def __enter__(self):
        self.stream.flush()  # ensures python stream unaffected
        self.fd = open(self.file, "w+")
        self.dup_stream = os.dup(self.stream.fileno())
        os.dup2(self.fd.fileno(), self.stream.fileno())  # replaces stream

    def __exit__(self, type, value, traceback):
        RedirectStream._flush_c_stream(self.stream)  # ensures C stream buffer empty
        os.dup2(self.dup_stream, self.stream.fileno())  # restores stream
        os.close(self.dup_stream)
        self.fd.close()


def filter_warnings(cata=Warning, module=""):
    warnings.filterwarnings("ignore", category=cata, module=module)


pprint = functools.partial(pprint_module.pprint, sort_dicts=False, width=120, compact=True)
pformat = functools.partial(pprint_module.pformat, sort_dicts=False, width=120, compact=True)
