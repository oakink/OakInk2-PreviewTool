from __future__ import annotations

import os
import torch
import numpy as np
from dataclasses import dataclass, fields
from collections import namedtuple
import pickle

import typing
import logging

if typing.TYPE_CHECKING:
    from typing import Optional, Union
    from torch import Tensor
    from numpy import ndarray as Array

from .type_def import NamedData, Struct
from .lbs import lbs, find_dynamic_lmk_idx_and_bcoords, vertices2landmarks
from ..transform.cast import to_tensor, to_array
from .vjsel import VertexJointSelector
from .rot_if import RotationConvert, HandRotationInterface

_logger = logging.getLogger(__name__)

VERTEX_IDS = {
    "nose": 9120,
    "reye": 9929,
    "leye": 9448,
    "rear": 616,
    "lear": 6,
    "rthumb": 8079,
    "rindex": 7669,
    "rmiddle": 7794,
    "rring": 7905,
    "rpinky": 8022,
    "lthumb": 5361,
    "lindex": 4933,
    "lmiddle": 5058,
    "lring": 5169,
    "lpinky": 5286,
    "LBigToe": 5770,
    "LSmallToe": 5780,
    "LHeel": 8846,
    "RBigToe": 8463,
    "RSmallToe": 8474,
    "RHeel": 8635,
}


POSE_LOWER_BODY_IDX = [0, 1, 3, 4, 6, 7, 9, 10]
POSE_UPPER_BODY_IDX = [i for i in range(21) if i not in POSE_LOWER_BODY_IDX]
POSE_NECK_IDX = [8, 11, 14]
POSE_TORSO_IDX = [i for i in POSE_UPPER_BODY_IDX if i not in POSE_NECK_IDX]


@dataclass
class SMPLXInput(NamedData):
    world_rot: Optional[Tensor] = None
    world_tsl: Optional[Tensor] = None
    body_shape: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None
    left_hand_pose: Optional[Tensor] = None
    right_hand_pose: Optional[Tensor] = None
    expr_shape: Optional[Tensor] = None
    jaw_pose: Optional[Tensor] = None
    leye_pose: Optional[Tensor] = None
    reye_pose: Optional[Tensor] = None


SMPLXTupleInput = namedtuple(
    "SMPLXTupleOutput",
    [el.name for el in fields(SMPLXInput)],
)

SMPLXInputKeyList = [el.name for el in fields(SMPLXInput)]


@dataclass
class SMPLXOutput(NamedData):
    vertices: Tensor
    joints: Tensor

    world_rot: Tensor
    world_tsl: Tensor

    betas: Tensor
    expression: Tensor

    body_pose: Tensor
    left_hand_pose: Tensor
    right_hand_pose: Tensor
    jaw_pose: Tensor

    full_pose: Tensor
    transform_abs: Tensor


SMPLXTupleOutput = namedtuple(
    "SMPLXTupleOutput",
    [el.name for el in fields(SMPLXOutput)],
)

SMPLXOutputKeyList = [el.name for el in fields(SMPLXOutput)]


def find_joint_kin_chain(joint_id, kinematic_tree):
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain


# TODO: support ortho6d input
class SMPLXLayer(torch.nn.Module):
    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
    SHAPE_SPACE_DIM = 300
    EXPRESSION_SPACE_DIM = 100
    NECK_IDX = 12

    def __init__(
        self,
        model_path: str,
        dtype=torch.float32,
        ext: str = "npz",
        gender: str = "neutral",
        vertex_ids: dict[str, int] = None,
        joint_mapper=None,
        rot_mode="rotmat",
        num_betas: int = 10,
        hand_use_pca: bool = False,
        hand_num_pca_comps: int = 6,
        hand_flat_hand_mean: bool = True,
        expr_num_coeffs: int = 10,
        expr_use_face_contour: bool = False,
        # custom ext
        use_jit: bool = False,
        use_body_upper_asset: Optional[str] = None,
        use_body_left_hand_asset: Optional[str] = None,
        use_body_right_hand_asset: Optional[str] = None,
    ) -> None:
        super().__init__()

        # === load the model
        self.gender = gender
        self.dtype = dtype
        data_struct = self.load_model(model_path, self.gender, ext)
        if vertex_ids is None:
            vertex_ids = VERTEX_IDS.copy()

        # === basic (body)
        self.rotation_cvt = RotationConvert(dtype=self.dtype, rot_mode=rot_mode)
        self.rot_mode = self.rotation_cvt.rot_mode
        self.setup_basic(data_struct, num_betas, vertex_ids, joint_mapper=joint_mapper)

        # === hand
        self.setup_hand(data_struct, hand_use_pca, hand_num_pca_comps, hand_flat_hand_mean)

        # === face
        self.setup_expression(data_struct, expr_num_coeffs, expr_use_face_contour)

        # === io mode
        self.use_jit = use_jit
        if self.use_jit:
            self.ret_type: type = SMPLXTupleOutput
            self.determine_batch_size = self._determine_batch_size_jit
        else:
            self.ret_type = SMPLXOutput
            self.determine_batch_size = self._determine_batch_size

        # === body upper asset
        if use_body_upper_asset is not None:
            _asset = torch.load(use_body_upper_asset, map_location=torch.device("cpu"))
            self.register_buffer("body_upper_vert_idx", _asset["vert_idx"])
            # self.register_buffer("body_upper_face_idx", _asset["face_idx"])
            self.register_buffer("body_upper_faces", _asset["faces"])

        # === body left hand asset
        if use_body_left_hand_asset is not None:
            _asset = torch.load(use_body_left_hand_asset, map_location=torch.device("cpu"))
            self.register_buffer("body_left_hand_vert_idx", _asset["vert_idx"])
            # self.register_buffer("body_left_hand_face_idx", _asset["face_idx"])
            self.register_buffer("body_left_hand_faces", _asset["faces"])

        # === body right hand asset
        if use_body_right_hand_asset is not None:
            _asset = torch.load(use_body_right_hand_asset, map_location=torch.device("cpu"))
            self.register_buffer("body_right_hand_vert_idx", _asset["vert_idx"])
            # self.register_buffer("body_right_hand_face_idx", _asset["face_idx"])
            self.register_buffer("body_right_hand_faces", _asset["faces"])

    @staticmethod
    def load_model(model_path, gender, ext):
        if os.path.isdir(model_path):
            model_fn = "SMPLX_{}.{ext}".format(gender.upper(), ext=ext)
            smplx_path = os.path.join(model_path, model_fn)
        else:
            smplx_path = model_path
        if not os.path.exists(smplx_path):
            _logger.error("model path {} does not exist!".format(smplx_path))
            raise RuntimeError("model path {} does not exist!".format(smplx_path))

        if ext == "pkl":
            with open(smplx_path, "rb") as smplx_file:
                model_data = pickle.load(smplx_file, encoding="latin1")
        elif ext == "npz":
            model_data = np.load(smplx_path, allow_pickle=True)
        else:
            _logger.error("Unknown extension: {}".format(ext))
            raise ValueError("Unknown extension: {}".format(ext))

        data_struct = Struct(**model_data)
        return data_struct

    def setup_basic(self, data_struct, num_betas, vertex_ids, joint_mapper):
        shapedirs = data_struct.shapedirs
        if shapedirs.shape[-1] < self.SHAPE_SPACE_DIM:
            _logger.warning(f"using {self.name()} model, with only 10 shape coefficients.")
            num_betas = min(num_betas, 10)
        else:
            num_betas = min(num_betas, self.SHAPE_SPACE_DIM)

        self.num_betas = num_betas
        shapedirs = shapedirs[:, :, :num_betas]
        # The shape components
        self.register_buffer("shapedirs", to_tensor(to_array(shapedirs), dtype=self.dtype))

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(vertex_ids=vertex_ids)

        self.faces = data_struct.f
        self.register_buffer("faces_tensor", to_tensor(to_array(self.faces, dtype=np.int64), dtype=torch.long))

        v_template = data_struct.v_template
        if not torch.is_tensor(v_template):
            v_template = to_tensor(to_array(v_template), dtype=self.dtype)
        # the vertices of the template model
        self.register_buffer("v_template", v_template)

        j_regressor = to_tensor(to_array(data_struct.J_regressor), dtype=self.dtype)
        self.register_buffer("J_regressor", j_regressor)

        # pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_array(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_array(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

        lbs_weights = to_tensor(to_array(data_struct.weights), dtype=self.dtype)
        self.register_buffer("lbs_weights", lbs_weights)

    def setup_hand(self, data_struct, hand_use_pca, hand_num_pca_comps, hand_flat_hand_mean):
        self.hand_rotation_if__left = HandRotationInterface(
            data_struct,
            side="left",
            dtype=self.dtype,
            rot_mode=self.rot_mode,
            hand_use_pca=hand_use_pca,
            hand_num_pca_comps=hand_num_pca_comps,
            hand_flat_hand_mean=hand_flat_hand_mean,
        )
        self.hand_rotation_if__right = HandRotationInterface(
            data_struct,
            side="right",
            dtype=self.dtype,
            rot_mode=self.rot_mode,
            hand_use_pca=hand_use_pca,
            hand_num_pca_comps=hand_num_pca_comps,
            hand_flat_hand_mean=hand_flat_hand_mean,
        )

    def setup_expression(self, data_struct, num_expression_coeffs, use_face_contour):
        lmk_faces_idx = data_struct.lmk_faces_idx
        self.register_buffer("lmk_faces_idx", torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = data_struct.lmk_bary_coords
        self.register_buffer("lmk_bary_coords", torch.tensor(lmk_bary_coords, dtype=self.dtype))

        self.use_face_contour = use_face_contour
        if self.use_face_contour:
            dynamic_lmk_faces_idx = data_struct.dynamic_lmk_faces_idx
            dynamic_lmk_faces_idx = torch.tensor(dynamic_lmk_faces_idx, dtype=torch.long)
            self.register_buffer("dynamic_lmk_faces_idx", dynamic_lmk_faces_idx)

            dynamic_lmk_bary_coords = data_struct.dynamic_lmk_bary_coords
            dynamic_lmk_bary_coords = torch.tensor(dynamic_lmk_bary_coords, dtype=self.dtype)
            self.register_buffer("dynamic_lmk_bary_coords", dynamic_lmk_bary_coords)

            neck_kin_chain = find_joint_kin_chain(self.NECK_IDX, self.parents)
            self.register_buffer("neck_kin_chain", torch.tensor(neck_kin_chain, dtype=torch.long))

        shapedirs = data_struct.shapedirs
        if len(shapedirs.shape) < 3:
            shapedirs = shapedirs[:, :, None]
        if shapedirs.shape[-1] < self.SHAPE_SPACE_DIM + self.EXPRESSION_SPACE_DIM:
            _logger.warning(f"using {self.name()} model with only 10 shape and 10 expression coefficients.")
            expr_start_idx = 10
            expr_end_idx = 20
            num_expression_coeffs = min(num_expression_coeffs, 10)
        else:
            expr_start_idx = self.SHAPE_SPACE_DIM
            expr_end_idx = self.SHAPE_SPACE_DIM + num_expression_coeffs
            num_expression_coeffs = min(num_expression_coeffs, self.EXPRESSION_SPACE_DIM)

        self.num_expression_coeffs = num_expression_coeffs

        expr_dirs = shapedirs[:, :, expr_start_idx:expr_end_idx]
        self.register_buffer("expr_dirs", to_tensor(to_array(expr_dirs), dtype=self.dtype))

    def forward(
        self,
        world_rot: Optional[Tensor] = None,
        world_tsl: Optional[Tensor] = None,
        body_shape: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        expr_shape: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
    ) -> Union[SMPLXOutput, SMPLXTupleOutput]:
        device: torch.device = self.shapedirs.device
        dtype: torch.dtype = self.shapedirs.dtype
        rot_mode: str = self.rot_mode

        # determine batchsize
        batch_size = self.determine_batch_size(
            (
                world_rot,
                world_tsl,
                body_shape,
                body_pose,
                left_hand_pose,
                right_hand_pose,
                expr_shape,
                jaw_pose,
                leye_pose,
                reye_pose,
            )
        )

        # TODO: move to init value pad fn
        if world_rot is None:
            _world_rot = (
                torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        if body_pose is None:
            _body_pose = (
                torch.eye(3, device=device, dtype=dtype)
                .view(1, 1, 3, 3)
                .expand(batch_size, self.NUM_BODY_JOINTS, -1, -1)
                .contiguous()
            )
        if left_hand_pose is None:
            _left_hand_pose = (
                torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
            )
        if right_hand_pose is None:
            _right_hand_pose = (
                torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
            )
        if jaw_pose is None:
            _jaw_pose = (
                torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        if leye_pose is None:
            _leye_pose = (
                torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        if reye_pose is None:
            _reye_pose = (
                torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        if expr_shape is None:
            expr_shape = torch.zeros([batch_size, self.num_expression_coeffs], dtype=dtype, device=device)
        if body_shape is None:
            body_shape = torch.zeros([batch_size, self.num_betas], dtype=dtype, device=device)
        if world_tsl is None:
            world_tsl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        # call rot_if
        world_rot = self.rotation_cvt(world_rot) if world_rot is not None else _world_rot
        body_pose = self.rotation_cvt(body_pose) if body_pose is not None else _body_pose
        jaw_pose = self.rotation_cvt(jaw_pose) if jaw_pose is not None else _jaw_pose
        leye_pose = self.rotation_cvt(leye_pose) if leye_pose is not None else _leye_pose
        reye_pose = self.rotation_cvt(reye_pose) if reye_pose is not None else _reye_pose

        # FIXME: should have better support for jit
        if left_hand_pose is not None:
            left_hand_pose = self.hand_rotation_if__left(left_hand_pose)
            left_hand_pose = self.rotation_cvt(left_hand_pose)
        else:
            left_hand_pose = _left_hand_pose
        if right_hand_pose is not None:
            right_hand_pose = self.hand_rotation_if__right(right_hand_pose)
            right_hand_pose = self.rotation_cvt(right_hand_pose)
        else:
            right_hand_pose = _right_hand_pose

        # Concatenate all pose vectors
        full_pose = torch.cat(
            [
                world_rot.reshape(-1, 1, 3, 3),
                body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3),
                jaw_pose.reshape(-1, 1, 3, 3),
                leye_pose.reshape(-1, 1, 3, 3),
                reye_pose.reshape(-1, 1, 3, 3),
                left_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3),
                right_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3),
            ],
            dim=1,
        )
        shape_components = torch.cat([body_shape, expr_shape], dim=-1)

        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints, transform_abs = lbs(
            shape_components,
            full_pose,
            self.v_template,
            shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices,
                full_pose,
                self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords

            lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat([lmk_bary_coords.expand(batch_size, -1, -1), dyn_lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)
        # Map the joints to the current dataset

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if world_tsl is not None:
            joints += world_tsl.unsqueeze(dim=1)
            vertices += world_tsl.unsqueeze(dim=1)
            # transform_abs apply world_tsl
            transform_abs = transform_abs.clone()
            transform_abs[..., :3, 3] += world_tsl.unsqueeze(dim=1)

        output = self.ret_type(
            # output
            vertices=vertices,
            joints=joints,
            # loc
            world_rot=world_rot,
            world_tsl=world_tsl,
            # shape
            betas=body_shape,
            expression=expr_shape,
            # pose
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            # for other layers
            full_pose=full_pose,
            transform_abs=transform_abs,
        )
        return output

    @staticmethod
    def _determine_batch_size(model_vars):
        batch_size = None
        for var in model_vars:
            if var is None:
                continue

            if batch_size is None:
                batch_size = var.shape[0]
            else:
                if var.shape[0] != batch_size:
                    var_name = f"{var=}".partition("=")[0]
                    _logger.error(f"incompatible batch_size at variable {var_name}")
                    raise RuntimeError(f"incompatible batch_size at variable {var_name}")
        if batch_size is None:
            batch_size = 1
        return batch_size

    @staticmethod
    def _determine_batch_size_jit(model_vars):
        return model_vars[0].shape[0]


def SMPLXLayerTraced(
    device: torch.device,
    batch_size: int,
    model_path: str,
    dtype=torch.float32,
    ext: str = "npz",
    gender: str = "neutral",
    vertex_ids: dict[str, int] = None,
    joint_mapper=None,
    rot_mode="rotmat",
    num_betas: int = 10,
    hand_use_pca: bool = False,
    hand_num_pca_comps: int = 6,
    hand_flat_hand_mean: bool = True,
    expr_num_coeffs: int = 10,
    expr_use_face_contour: bool = False,
):
    if rot_mode != "rotmat":
        raise NotImplementedError()

    smplx_layer = SMPLXLayer(
        model_path=model_path,
        dtype=dtype,
        ext=ext,
        gender=gender,
        vertex_ids=vertex_ids,
        joint_mapper=joint_mapper,
        rot_mode=rot_mode,
        num_betas=num_betas,
        hand_use_pca=hand_use_pca,
        hand_num_pca_comps=hand_num_pca_comps,
        hand_flat_hand_mean=hand_flat_hand_mean,
        expr_num_coeffs=expr_num_coeffs,
        expr_use_face_contour=expr_use_face_contour,
        use_jit=True,
    ).to(device)

    # if rot_mode == "rotmat":
    from ..transform.init.smplx import init_random_input

    input_map = init_random_input(device=device, dtype=dtype, batch_size=batch_size, grad=False)

    # trace module
    traced_smplx_layer = torch.jit.trace(smplx_layer, example_inputs=tuple(input_map.values()))

    def traced_fn(*arg, **kwarg):
        res = traced_smplx_layer(*arg, **kwarg)
        return SMPLXOutput(*res)

    return traced_fn


def load_extra_asset(filepath):
    return torch.load(filepath, map_location=torch.device("cpu"))
