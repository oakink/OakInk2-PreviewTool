from typing import Optional
import numpy as np


def copysign_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return a ndarray where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.
    Args:
        a: source ndarray.
        b: ndarray whose signs will be used, of the same shape as a.
    Returns:
        ndarray of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return np.where(signs_differ, -a, a)


def standardize_quat_np(quat: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.
    Args:
        quat: quaternions in form (w,x,y,z),
              tensor of shape (..., 4).
    Returns:
        standardized quaternions in form (w,x,y,z),
        tensor of shape (..., 4).
    """
    return np.where(quat[..., 0:1] < 0, -quat, quat)


def normalize_quat_np(quat: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
    if eps is None:
        eps = np.finfo(quat.dtype).eps
    quat_norm = np.linalg.norm(quat, ord=2, axis=-1, keepdims=True)
    quat_nromalized = quat / np.maximum(quat_norm, eps)
    # standardize the quaternion to have non-negative real part
    return standardize_quat_np(quat_nromalized)


def quat_raw_multiply_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.
    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.
    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return np.stack((ow, ox, oy, oz), -1)


def quat_multiply_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.
    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.
    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quat_raw_multiply_np(a, b)
    return standardize_quat_np(ab)


def quat_invert_np(quaternion: np.ndarray) -> np.ndarray:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.
    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).
    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = np.array([1, -1, -1, -1], dtype=quaternion.dtype)
    return quaternion * scaling


def quat_apply_np(quat: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.
    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).
    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.shape[-1] != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = np.zeros((point.shape[:-1] + (1,)), dtype=point.dtype)
    point_as_quat = np.concatenate((real_parts, point), -1)
    out = quat_raw_multiply_np(
        quat_raw_multiply_np(quat, point_as_quat),
        quat_invert_np(quat),
    )
    return out[..., 1:]


def _sqrt_positive_part(x: np.ndarray) -> np.ndarray:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = np.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = np.sqrt(x[positive_mask])
    return ret


def _one_hot(idx: np.ndarray, num_classes: int) -> np.ndarray:
    identity = np.eye(num_classes)
    return identity[idx]


def quat_to_rotmat_np(quat: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quat: quaternions in form (w,x,y,z),
              ndarray of shape (..., 4).
    Returns:
        rotmat: ndarray of shape (..., 3, 3).
    """
    r, i, j, k = (el.squeeze(-1) for el in np.split(quat, 4, axis=-1))
    two_s = 2.0 / (quat * quat).sum(-1)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quat.shape[:-1] + (3, 3)).astype(quat.dtype)


def rotmat_to_quat_np(rotmat: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        rotmat: tensor of shape (..., 3, 3).
    Returns:
        quaternions in form (w,x,y,z),
        tensor of shape (..., 4).
    """
    if rotmat.shape[-1] != 3 or rotmat.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {rotmat.shape}.")

    batch_dim = rotmat.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = (
        el.squeeze(-1) for el in np.split(rotmat.reshape(batch_dim + (9,)), 9, axis=-1)
    )

    q_abs = _sqrt_positive_part(
        np.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            axis=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = np.stack(
        [
            np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = np.array(0.1).astype(q_abs.dtype)
    quat_candidates = quat_by_rijk / (2.0 * np.maximum(q_abs[..., None], flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[_one_hot(q_abs.argmax(axis=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    return standardize_quat_np(out)


def rotvec_to_rotmat_np(rotvec: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as axis/angle to rotation matrices.
    Args:
        rotvec: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quat_to_rotmat_np(rotvec_to_quat_np(rotvec))


def rotmat_to_rotvec_np(rotmat: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as rotation matrices to axis/angle.
    Args:
        rotmat: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quat_to_rotvec_np(rotmat_to_quat_np(rotmat))


def rotvec_to_quat_np(rotvec: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as axis/angle to quaternions.
    Args:
        rotvec: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = np.linalg.norm(rotvec, ord=2, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = np.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    quat = np.concatenate([np.cos(half_angles), rotvec * sin_half_angles_over_angles], axis=-1)
    return quat


def quat_to_rotvec_np(quat: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quat: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = np.linalg.norm(quat[..., 1:], ord=2, axis=-1, keepdims=True)
    half_angles = np.arctan2(norms, quat[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = np.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quat[..., 1:] / sin_half_angles_over_angles


def _axis_angle_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angle_to_rotmat_np(euler_angles: np.ndarray, convention: str) -> np.ndarray:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.ndim == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, (el.squeeze(-1) for el in np.split(euler_angles, euler_angles.shape[-1], axis=-1)))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return np.matmul(np.matmul(matrices[0], matrices[1]), matrices[2])


def euler_angle_to_quat_np(euler_angles: np.ndarray, convention: str) -> np.ndarray:
    """
    Convert rotations given as Euler angles in radians to quaternions.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return rotmat_to_quat_np(euler_angle_to_rotmat_np(euler_angles, convention))


def euler_angle_to_rotvec_np(euler_angles: np.ndarray, convention: str) -> np.ndarray:
    """
    Convert rotations given as Euler angles in radians to axis/angle.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return rotmat_to_rotvec_np(euler_angle_to_rotmat_np(euler_angles, convention))


def _angle_from_tan(axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool) -> np.ndarray:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.
    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.
    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return np.arctan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return np.arctan2(-data[..., i2], data[..., i1])
    return np.arctan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def rotmat_to_euler_angle_np(matrix: np.ndarray, convention: str) -> np.ndarray:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.
    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        eps = 1e-6
        _v = matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        _v = np.clip(_v, -1.0 + eps, 1.0 - eps)
        central_angle = np.arcsin(_v)
    else:
        central_angle = np.arccos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(convention[0], convention[1], matrix[..., i2], False, tait_bryan),
        central_angle,
        _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True, tait_bryan),
    )
    return np.stack(o, -1)


def quat_to_euler_angle_np(quat: np.ndarray, convention: str) -> np.ndarray:
    """
    Convert rotations given as quaternions to Euler angles in radians.
    Args:
        quat: Quaternions as tensor of shape (..., 4).
        convention: Convention string of three uppercase letters.
    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    return rotmat_to_euler_angle_np(quat_to_rotmat_np(quat), convention)


def rotvec_to_euler_angle_np(rotvec: np.ndarray, convention: str) -> np.ndarray:
    """
    Convert rotations given as axis/angle to Euler angles in radians.
    Args:
        rotvec: Quaternions as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters.
    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    return rotmat_to_euler_angle_np(rotvec_to_rotmat_np(rotvec), convention)


def _normalize(mat: np.ndarray, axis: int, eps: Optional[float] = None) -> np.ndarray:
    if eps is None:
        eps = np.finfo(mat.dtype).eps
    norm = np.linalg.norm(mat, ord=2, axis=axis, keepdims=True)
    return mat / np.maximum(norm, eps)


def rot6d_to_rotmat_np(d6: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1, axis=-1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize(b2, axis=-1)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def rotmat_to_rot6d_np(matrix: np.ndarray) -> np.ndarray:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].copy().reshape(batch_dim + (6,))
