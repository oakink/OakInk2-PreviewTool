import sys
import shlex
from copy import deepcopy

import logging

_logger = logging.getLogger("__name__")


def arg_to_string(arg):
    res = "{\n"
    for k, v in vars(arg).items():
        res += f"  {k:<20}: {v}\n"
    res += "}"
    return res


def argdict_to_string(argdict):
    # TODO: support recursive print to depth
    if argdict is None:
        return r"{}"
    res = "{\n"
    for k, v in argdict.items():
        if isinstance(v, dict):
            res += f"  {k:<20}: {{\n"
            for sub_k, sub_v in v.items():
                res += f"    {sub_k:<18}: {sub_v}\n"
            res += "  }\n"
        elif isinstance(v, list):
            res += f"  {k:<20}: [\n"
            for el in v:
                res += f"    - {el}\n"
            res += "  ]\n"
        else:
            res += f"  {k:<20}: {v}\n"
    res += "}"
    return res


def get_command():
    return shlex.join(deepcopy(sys.argv))
