from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing import Optional
    from typing import Union

    Device = Union[str, torch.device]

import torch
import torch.nn.functional as F

from .rotation import copysign
from .rotation import quat_to_rotmat


def random_quat_n(n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.
    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.
    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_quat(dtype: Optional[torch.dtype] = None, device: Optional[Device] = None) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.
    Args:
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.
    Returns:
        Quaternions as tensor of shape (4, ).
    """
    return random_quat_n(1, dtype=dtype, device=device)[0]


def random_rotmat_n(n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.
    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.
    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quat = random_quat_n(n, dtype=dtype, device=device)
    return quat_to_rotmat(quat)


def random_rotmat(dtype: Optional[torch.dtype] = None, device: Optional[Device] = None) -> torch.Tensor:
    """
    Generate a single random 3x3 rotation matrix.
    Args:
        dtype: Type to return
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type
    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return random_rotmat_n(1, dtype, device)[0]
