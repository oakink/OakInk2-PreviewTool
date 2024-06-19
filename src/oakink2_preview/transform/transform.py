from __future__ import annotations

import torch

from .rotation import rotvec_to_rotmat, rotmat_to_rotvec, rotmat_to_quat, quat_to_rotmat
from .rotation import quat_to_rotvec, quat_multiply, quat_invert
from .rotation import rot6d_to_rotmat, rotmat_to_rot6d


def assemble_T(tsl: torch.Tensor, rotmat: torch.Tensor):
    # tsl [..., 3]
    # rotmat [..., 3, 3]
    leading_shape = tsl.shape[:-1]
    # leading_dim = len(leading_shape)

    res = torch.zeros((*leading_shape, 4, 4)).to(tsl)
    res[..., 3, 3] = 1.0
    res[..., :3, 3] = tsl
    res[..., :3, :3] = rotmat
    return res


def inv_transf(transf: torch.Tensor):
    leading_shape = transf.shape[:-2]
    leading_dim = len(leading_shape)

    R_inv = torch.transpose(transf[..., :3, :3], leading_dim, leading_dim + 1)
    t_inv = -R_inv @ transf[..., :3, 3:]
    res = torch.zeros_like(transf)
    res[..., :3, :3] = R_inv
    res[..., :3, 3:] = t_inv
    res[..., 3, 3] = 1
    return res


def transf_point_array(transf: torch.Tensor, point: torch.Tensor):
    # transf: [..., 4, 4]
    # point: [..., X, 3]
    leading_shape = point.shape[:-2]
    leading_dim = len(leading_shape)

    res = (
        torch.transpose(
            torch.matmul(
                transf[..., :3, :3],
                torch.transpose(point, leading_dim, leading_dim + 1),
            ),
            leading_dim,
            leading_dim + 1,
        )  # [..., X, 3]
        + transf[..., :3, 3][..., None, :]  # [..., 1, 3]
    )
    return res


def project_point_array(cam_intr: torch.Tensor, point: torch.Tensor, eps=1e-7) -> torch.Tensor:
    leading_shape = point.shape[:-2]
    leading_dim = len(leading_shape)

    hom_2d = torch.transpose(
        torch.matmul(
            cam_intr,
            torch.transpose(point, leading_dim, leading_dim + 1),
        ),
        leading_dim,
        leading_dim + 1,
    )  # [..., X, 3]
    xy = hom_2d[..., 0:2]
    z = hom_2d[..., 2:]
    z[torch.abs(z) < eps] = eps
    uv = xy / z
    return uv


def se3_to_transf(se3: torch.Tensor) -> torch.Tensor:
    rotvec = se3[..., 0:3]
    tsl = se3[..., 3:6]
    rotmat = rotvec_to_rotmat(rotvec)
    return assemble_T(tsl, rotmat)


def transf_to_se3(transf: torch.Tensor) -> torch.Tensor:
    rotmat = transf[..., :3, :3]
    tsl = transf[..., :3, 3]
    rotvec = rotmat_to_rotvec(rotmat)
    return torch.cat([rotvec, tsl], dim=-1)


def approx_avg_transf(transf_list: list[torch.Tensor]) -> torch.Tensor:
    tsl_list = []
    quat_list = []
    for transf in transf_list:
        tsl = transf[..., 0:3, 3]  # [..., 3]
        rotmat = transf[..., 0:3, 0:3]
        quat = rotmat_to_quat(rotmat)  # [..., 4]
        tsl_list.append(tsl)
        quat_list.append(quat)

    tsl = torch.stack(tsl_list, dim=-1)
    tsl = torch.mean(tsl, axis=-1)  # [..., 3]
    quat = torch.stack(quat_list, dim=-1)
    quat = torch.mean(quat, axis=-1)  # [..., 4] # approx

    rotmat = quat_to_rotmat(quat)
    return assemble_T(tsl, rotmat)


def transf_to_posevec(transf: torch.Tensor) -> torch.Tensor:
    # transf: [..., 4, 4]>
    # posevec: [..., 6]
    tsl = transf[..., :3, 3]  # [..., 3]
    rotmat = transf[..., :3, :3]
    quat = rotmat_to_quat(rotmat)  # [..., 4]
    posevec = torch.cat([tsl, quat[..., 1:]], dim=-1)
    return posevec


def posevec_diff(posevec_a: torch.Tensor, posevec_b: torch.Tensor) -> torch.Tensor:
    pos_a = posevec_a[:3]
    quat_a = posevec_a[3:]
    pos_b = posevec_b[:3]
    quat_b = posevec_b[3:]

    pos_diff = pos_a - pos_b
    quat_diff = quat_multiply(quat_a, quat_invert(quat_b))
    return torch.cat([pos_diff, quat_diff], dim=0)


def posevec_norm(posevec: torch.Tensor) -> torch.Tensor:
    pos = posevec[:3]
    quat = posevec[3:]
    pos_norm = torch.norm(pos, dim=-1)
    rotvec = quat_to_rotvec(quat)
    rotangle = torch.norm(rotvec, dim=-1)
    return pos_norm, rotangle


def transf_to_tslrot6d(transf: torch.Tensor) -> torch.Tensor:
    # tsl: [..., 3]
    # rot6d: [..., 6]
    tsl = transf[..., :3, 3]  # [..., 3]
    rotmat = transf[..., :3, :3]  # [..., 3, 3]
    rot6d = rotmat_to_rot6d(rotmat)  # [..., 6]
    tslrot6d = torch.cat((tsl, rot6d), dim=-1)
    return tslrot6d


def tslrot6d_to_transf(tslrot6d: torch.Tensor) -> torch.Tensor:
    # tsl: [..., 3]
    # rot6d: [..., 6]
    tsl = tslrot6d[..., 0:3]  # [..., 3]
    rot6d = tslrot6d[..., 3:9]  # [..., 6]
    rotmat = rot6d_to_rotmat(rot6d)  # [..., 3, 3]
    return assemble_T(tsl, rotmat)


def inv_rotmat(rotmat: torch.Tensor):
    leading_shape = rotmat.shape[:-2]
    leading_dim = len(leading_shape)

    rotmat_inv = torch.transpose(rotmat[..., :3, :3], leading_dim, leading_dim + 1)
    return rotmat_inv


def rotate_point_array(rotmat: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    # rotmat: [..., 3, 3]
    # point: [..., X, 3]
    leading_shape = point.shape[:-2]
    leading_dim = len(leading_shape)

    res = torch.transpose(
        torch.matmul(
            rotmat,
            torch.transpose(point, leading_dim, leading_dim + 1),
        ),
        leading_dim,
        leading_dim + 1,
    )
    return res
