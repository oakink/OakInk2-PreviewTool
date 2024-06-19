import torch
import numpy as np

from ..transform.cast import to_tensor
from ..transform.rotation import quat_to_rotmat, rotvec_to_rotmat

import logging

_logger = logging.getLogger(__name__)


class RotationConvert(torch.nn.Module):
    def __init__(self, dtype=torch.float32, rot_mode="rotmat"):
        super().__init__()

        self.dtype = dtype
        if rot_mode not in ["rotmat", "rotvec", "quat", "ortho6d"]:
            _logger.error("unsupported rot_mode: %s", rot_mode)
            raise RuntimeError(f"unsupported rot_mode: {rot_mode}")
        self.rot_mode = rot_mode
        self.update_rot_fn()

    def update_rot_fn(self):
        if self.rot_mode == "quat":
            self.rot_fn = self.quat_fn
        elif self.rot_mode == "rotvec":
            self.rot_fn = self.rotvec_fn
        elif self.rot_mode == "ortho6d":
            raise NotImplementedError()
        else:
            # self.rot_mode == "rotmat":
            self.rot_fn = self.identity_fn

    @staticmethod
    def identity_fn(rot: torch.Tensor, dtype: torch.dtype):
        return rot.to(dtype)

    @staticmethod
    def rotvec_fn(rot: torch.Tensor, dtype: torch.dtype):
        res = rotvec_to_rotmat(rot.to(dtype))
        return res.to(dtype)

    @staticmethod
    def quat_fn(rot: torch.Tensor, dtype: torch.dtype):
        res = quat_to_rotmat(rot.to(dtype))
        return res.to(dtype)

    def forward(self, rot: torch.Tensor) -> torch.Tensor:
        return self.rot_fn(rot, self.dtype)


# TODO: this works well in eager mode, but if more performance is wanted from jit, this seems awkward
# TODO: flat_hand_mean: convert to corresponding format
class HandRotationInterface(torch.nn.Module):
    NUM_HAND_JOINTS = 15

    def __init__(
        self,
        data_struct,
        dtype=torch.float32,
        rot_mode="rotmat",
        side="both",
        hand_use_pca=False,
        hand_num_pca_comps=10,
        hand_flat_hand_mean=True,
    ) -> None:
        super().__init__()

        self.dtype = dtype
        self.rot_if = RotationConvert(dtype=self.dtype, rot_mode=rot_mode)
        self.rot_mode = rot_mode
        if side not in ["both", "left", "right"]:
            _logger.error("unsupported side: %s", side)
            raise RuntimeError(f"unsupported side: {side}")
        self.side = side

        self.setup_hand(data_struct, hand_use_pca, hand_num_pca_comps, hand_flat_hand_mean)

    def setup_hand(self, data_struct, hand_use_pca, hand_num_pca_comps, hand_flat_hand_mean):
        self.use_pca = hand_use_pca
        self.num_pca_comps = hand_num_pca_comps
        self.flat_hand_mean = hand_flat_hand_mean

        if self.side == "both":
            setup_left, setup_right = True, True
        elif self.side == "left":
            setup_left, setup_right = True, False
        else:
            # self.side == "right":
            setup_left, setup_right = False, True

        if setup_left:
            left_hand_components = data_struct.hands_componentsl[:hand_num_pca_comps]

            self.np_left_hand_components = left_hand_components
            if self.use_pca:
                self.register_buffer("left_hand_components", torch.tensor(left_hand_components, dtype=self.dtype))

            if self.flat_hand_mean:
                left_hand_mean = np.zeros_like(data_struct.hands_meanl)
            else:
                left_hand_mean = data_struct.hands_meanl
            left_hand_mean = left_hand_mean.reshape((self.NUM_HAND_JOINTS, 3))
            self.register_buffer("left_hand_mean", to_tensor(left_hand_mean, dtype=self.dtype))

        if setup_right:
            right_hand_components = data_struct.hands_componentsr[:hand_num_pca_comps]

            self.np_right_hand_components = right_hand_components
            if self.use_pca:
                self.register_buffer("right_hand_components", torch.tensor(right_hand_components, dtype=self.dtype))

            if self.flat_hand_mean:
                right_hand_mean = np.zeros_like(data_struct.hands_meanr)
            else:
                right_hand_mean = data_struct.hands_meanr
            right_hand_mean = right_hand_mean.reshape((self.NUM_HAND_JOINTS, 3))
            self.register_buffer("right_hand_mean", to_tensor(right_hand_mean, dtype=self.dtype))

    def forward(self, pose, pose_=None):
        if self.side == "both":
            left_hand_pose, right_hand_pose = pose, pose_
        elif self.side == "left":
            left_hand_pose, right_hand_pose = pose, None
        else:
            # self.side == "right":
            left_hand_pose, right_hand_pose = None, pose

        if self.rot_mode == "rotvec":
            if self.use_pca:
                # deal with pca and mean pose
                if left_hand_pose is not None:
                    batch_size = left_hand_pose.shape[0]
                    left_hand_pose = torch.einsum("bi,ij->bj", [left_hand_pose, self.left_hand_components])
                    left_hand_pose = left_hand_pose.reshape((batch_size, self.NUM_HAND_JOINTS, 3))
                if right_hand_pose is not None:
                    batch_size = right_hand_pose.shape[0]
                    right_hand_pose = torch.einsum("bi,ij->bj", [right_hand_pose, self.right_hand_components])
                    right_hand_pose = right_hand_pose.reshape((batch_size, self.NUM_HAND_JOINTS, 3))

            if left_hand_pose is not None:
                left_hand_pose = left_hand_pose + self.left_hand_mean

            if right_hand_pose is not None:
                right_hand_pose = right_hand_pose + self.right_hand_mean

        if self.side == "both":
            return left_hand_pose, right_hand_pose
        elif self.side == "left":
            return left_hand_pose
        else:
            # self.side == "right":
            return right_hand_pose
