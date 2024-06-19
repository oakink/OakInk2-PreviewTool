from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing import Optional
    from typing import Union

import numpy as np

from .rotation_np import copysign_np
from .rotation_np import quat_to_rotmat_np


def random_quat_n_np(n: int, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.
    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    o = np.random.randn(n, 4).astype(dtype)
    s = (o * o).sum(1)
    o = o / copysign_np(np.sqrt(s), o[:, 0])[:, None]
    return o


def random_quat_np(dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.
    Args:
        dtype: Type to return.
    Returns:
        Quaternions as tensor of shape (4, ).
    """
    return random_quat_n_np(1, dtype=dtype)[0]


def random_rotmat_n_np(n: int, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Generate random rotations as 3x3 rotation matrices.
    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quat = random_quat_n_np(n, dtype=dtype)
    return quat_to_rotmat_np(quat)


def random_rotmat_np(dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Generate a single random 3x3 rotation matrix.
    Args:
        dtype: Type to return
    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return random_rotmat_n_np(1, dtype)[0]
