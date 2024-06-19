from __future__ import annotations

import torch
import numpy as np

import typing
from typing import Union

if typing.TYPE_CHECKING:
    from torch import Tensor
    from numpy import ndarray as Array
    from typing import Mapping, MutableMapping


def to_tensor(array: Union[Array, Tensor], dtype=torch.float32) -> Tensor:
    if torch.is_tensor(array):
        return array.to(dtype)
    else:
        return torch.tensor(array, dtype=dtype)


def to_array(array: Union[Array, Tensor], dtype=np.float32) -> Array:
    return np.array(array, dtype=dtype)


def as_array(array: Union[Array, Tensor], dtype=np.float32) -> Array:
    return np.asarray(array, dtype=dtype)


def map_to_device(mapping, device):
    for k in mapping.keys():
        mapping[k] = mapping[k].to(device)
    return mapping


def map_to_tensor(mapping: MutableMapping, dtype=torch.float32):
    for k in mapping.keys():
        mapping[k] = to_tensor(mapping[k], dtype=dtype)
    return mapping


def map_to_array(mapping: MutableMapping, dtype=np.float32):
    for k in mapping.keys():
        mapping[k] = to_array(mapping[k], dtype=dtype)
    return mapping


def map_copy_to_device(mapping, device):
    res = {}
    for k in mapping.keys():
        res[k] = mapping[k].to(device)
    return res


def map_copy_to_tensor(mapping: Mapping, dtype=torch.float32):
    res = {}
    for k in mapping.keys():
        res[k] = to_tensor(mapping[k], dtype=dtype)
    return res


def map_copy_to_array(mapping: Mapping, dtype=np.float32):
    res = {}
    for k in mapping.keys():
        res[k] = to_array(mapping[k], dtype=dtype)
    return res


def map_deepcopy_to_device(mapping: Mapping, device):
    res = {}
    for k in mapping.keys():
        res[k] = mapping[k].detach().clone().to(device)
    return res


def map_copy_select_to(mapping, device=None, dtype=None, select=None):
    res = {}
    if select is None:
        select = mapping
    for k in mapping.keys():
        if k in select:
            res[k] = mapping[k].to(device=device, dtype=dtype)
        else:
            res[k] = mapping[k]
    return res
