import contextlib
import os
import sys

# ! warning ! make sure the with context ends in one critical section. remember to lock!


@contextlib.contextmanager
def modify_cuda_visible_devices(gpu_id):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        yield
    finally:
        os.environ.pop("CUDA_VISIBLE_DEVICES")


@contextlib.contextmanager
def modify_sys_path(entry):
    try:
        sys.path.append(entry)
        yield
    finally:
        sys.path.pop(sys.path.index(entry))
