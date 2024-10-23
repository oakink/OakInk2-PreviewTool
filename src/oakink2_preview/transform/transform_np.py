from __future__ import annotations

import numpy as np

from .rotation_np import rotvec_to_rotmat_np, rotmat_to_rotvec_np, rotmat_to_quat_np, quat_to_rotmat_np
from .rotation_np import quat_invert_np, quat_multiply_np, quat_to_rotvec_np
from .rotation_np import rot6d_to_rotmat_np, rotmat_to_rot6d_np


def assemble_T_np(tsl, rotmat):
    # tsl [..., 3]
    # rotmat [..., 3, 3]
    leading_shape = tsl.shape[:-1]
    # leading_dim = len(leading_shape)

    res = np.zeros((*leading_shape, 4, 4), like=tsl)
    res[..., 3, 3] = 1.0
    res[..., :3, 3] = tsl
    res[..., :3, :3] = rotmat
    return res


def inv_transf_np(transf: np.ndarray):
    leading_shape = transf.shape[:-2]
    leading_dim = len(leading_shape)

    R_inv = np.swapaxes(transf[..., :3, :3], leading_dim, leading_dim + 1)
    t_inv = -R_inv @ transf[..., :3, 3:]
    res = np.zeros_like(transf)
    res[..., :3, :3] = R_inv
    res[..., :3, 3:] = t_inv
    res[..., 3, 3] = 1
    return res


def transf_point_array_np(transf: np.ndarray, point: np.ndarray):
    # transf: [..., 4, 4]
    # point: [..., X, 3]
    leading_shape = point.shape[:-2]
    leading_dim = len(leading_shape)

    res = (
        np.swapaxes(
            np.matmul(
                transf[..., :3, :3],
                np.swapaxes(point, leading_dim, leading_dim + 1),
            ),
            leading_dim,
            leading_dim + 1,
        )  # [..., X, 3]
        + transf[..., :3, 3][..., np.newaxis, :]  # [..., 1, 3]
    )
    return res


def project_point_array_np(cam_intr: np.ndarray, point: np.ndarray, eps=1e-7):
    leading_shape = point.shape[:-2]
    leading_dim = len(leading_shape)

    hom_2d = np.swapaxes(
        np.matmul(
            cam_intr,
            np.swapaxes(point, leading_dim, leading_dim + 1),
        ),
        leading_dim,
        leading_dim + 1,
    )  # [..., N, 3]
    xy = hom_2d[..., 0:2]
    z = hom_2d[..., 2:]
    z[np.abs(z) < eps] = eps
    uv = xy / z
    return uv


def se3_to_transf_np(se3: np.ndarray):
    tsl = se3[..., 0:3]
    rot = se3[..., 3:6]

    prev_shape = se3.shape[:-1]
    transf_shape = prev_shape + (4, 4)
    transf_dtype = se3.dtype

    res = np.zeros(shape=transf_shape, dtype=transf_dtype)
    rot_mat = rotvec_to_rotmat_np(rot)
    res[..., :3, :3] = rot_mat
    res[..., :3, 3] = tsl
    res[..., 3, 3] = 1.0

    return res


def transf_to_se3_np(transf: np.ndarray):
    tsl = transf[..., 0:3, 3]  # [..., 3]
    rotmat = transf[..., 0:3, 0:3]
    rotvec = rotmat_to_rotvec_np(rotmat)  # [..., 3]
    se3 = np.concatenate((tsl, rotvec), axis=-1)
    return se3


def approx_avg_transf_np(transf_list: list[np.ndarray]):
    tsl_list = []
    quat_list = []
    for transf in transf_list:
        tsl = transf[..., 0:3, 3]  # [..., 3]
        rotmat = transf[..., 0:3, 0:3]
        quat = rotmat_to_quat_np(rotmat)  # [..., 4]
        tsl_list.append(tsl)
        quat_list.append(quat)

    tsl = np.stack(tsl_list, axis=-1)
    tsl = np.mean(tsl, axis=-1)  # [..., 3]
    quat = np.stack(quat_list, axis=-1)
    quat = np.mean(quat, axis=-1)  # [..., 4] # approx

    rotmat = quat_to_rotmat_np(quat)

    transf = assemble_T_np(tsl, rotmat)
    return transf


def transf_to_posevec_np(transf: np.ndarray) -> np.ndarray:
    # transf: [..., 4, 4]
    # posevec: [..., 6]
    tsl = transf[..., :3, 3]  # [..., 3]
    rotmat = transf[..., :3, :3]
    quat = rotmat_to_quat_np(rotmat)  # [..., 4]
    posevec = np.concatenate((tsl, quat), axis=-1)
    return posevec


def posevec_to_transf_np(posevec):
    tsl = posevec[:3]
    rot = posevec[3:]
    rotmat = quat_to_rotmat_np(rot)
    transf = assemble_T_np(tsl, rotmat)
    return transf


def posevec_diff_np(posevec_a, posevec_b):
    pos_a = posevec_a[:3]
    quat_a = posevec_a[3:]
    pos_b = posevec_b[:3]
    quat_b = posevec_b[3:]

    pos_diff = pos_a - pos_b
    quat_diff = quat_multiply_np(quat_a, quat_invert_np(quat_b))
    return np.concatenate((pos_diff, quat_diff), axis=0)


def posevec_norm_np(posevec):
    pos = posevec[:3]
    quat = posevec[3:]
    pos_norm = np.linalg.norm(pos)
    rotvec = quat_to_rotvec_np(quat)
    rotangle = np.linalg.norm(rotvec)
    return pos_norm, rotangle


def transf_to_tslrot6d_np(transf: np.ndarray) -> np.ndarray:
    # tsl: [..., 3]
    # rot6d: [..., 6]
    tsl = transf[..., :3, 3]
    rotmat = transf[..., :3, :3]
    rot6d = rotmat_to_rot6d_np(rotmat)
    tslrot6d = np.concatenate((tsl, rot6d), axis=-1)
    return tslrot6d


def tslrot6d_to_transf_np(tslrot6d: np.ndarray) -> np.ndarray:
    # tsl: [..., 3]
    # rot6d: [..., 6]
    tsl = tslrot6d[..., 0:3]  # [..., 3]
    rot6d = tslrot6d[..., 3:9]  # [..., 6]
    rotmat = rot6d_to_rotmat_np(rot6d)  # [..., 3, 3]
    return assemble_T_np(tsl, rotmat)
