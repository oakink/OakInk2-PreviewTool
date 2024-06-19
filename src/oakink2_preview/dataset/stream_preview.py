import os
import logging
import numpy as np
import cv2
import pickle
import typing
import dataclasses
import torch
from copy import deepcopy

from ..layer.type_def import NamedData

FRAME_SHAPE = (480, 848, 3)  # may differ
FPS_MOCAP = 120
FPS_COLOR = 30
FPS = FPS_COLOR
CAMERA_LAYOUT_DESC = [
    "allocentric_top",
    "allocentric_left",
    "allocentric_right",
    "egocentric",
]
_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FrameData(NamedData):
    frame_id: int

    color_allocentric_top: typing.Optional[np.ndarray] = None
    color_allocentric_left: typing.Optional[np.ndarray] = None
    color_allocentric_right: typing.Optional[np.ndarray] = None
    color_egocentric: typing.Optional[np.ndarray] = None

    cam_intr_allocentric_top: typing.Optional[np.ndarray] = None
    cam_intr_allocentric_left: typing.Optional[np.ndarray] = None
    cam_intr_allocentric_right: typing.Optional[np.ndarray] = None
    cam_intr_egocentric: typing.Optional[np.ndarray] = None

    cam_extr_allocentric_top: typing.Optional[np.ndarray] = None
    cam_extr_allocentric_left: typing.Optional[np.ndarray] = None
    cam_extr_allocentric_right: typing.Optional[np.ndarray] = None
    cam_extr_egocentric: typing.Optional[np.ndarray] = None

    optitrack_obj_transf: typing.Optional[typing.Mapping] = None
    smplx_result: typing.Optional[typing.Mapping] = None


class StreamDataset:
    def __init__(
        self,
        stream_filedir: str,
        anno_filepath: str,
    ):
        self.cam_selection = deepcopy(CAMERA_LAYOUT_DESC)
        self.ret_type = FrameData

        self.stream_filedir = stream_filedir
        self.anno_filepath = anno_filepath

        with open(self.anno_filepath, "rb") as ifs:
            self.anno = pickle.load(ifs)

        self.cam_def = self.anno["cam_def"]
        self.rev_cam_def = {v: k for k, v in self.cam_def.items()}
        self.cam_selection = self.anno["cam_selection"]
        self.frame_id_list = self.anno["frame_id_list"]
        self.object_list = self.anno["obj_list"]

        self.len = len(self.frame_id_list)

    def __getitem__(self, image_id):
        frame_id = self.frame_id_list[image_id]
        res_content = {"frame_id": frame_id}

        color_prefix = self.stream_filedir
        for cam_layout_desc in self.cam_selection:
            color_frame = cv2.imread(
                os.path.join(color_prefix, f"{self.rev_cam_def[cam_layout_desc]}/{frame_id:0>6}.png")
            )
            res_content[f"color_{cam_layout_desc}"] = color_frame

        for cam_layout_desc in self.cam_selection:
            cam_intr = self.anno["cam_intr"][cam_layout_desc][frame_id]
            res_content[f"cam_intr_{cam_layout_desc}"] = cam_intr

        for cam_layout_desc in self.cam_selection:
            cam_extr = self.anno["cam_extr"][cam_layout_desc][frame_id]
            res_content[f"cam_extr_{cam_layout_desc}"] = cam_extr

        obj_transf_map = {}
        for obj in self.object_list:
            obj_transf_map[obj] = self.anno["obj_transf"][obj][frame_id]
        res_content["optitrack_obj_transf"] = obj_transf_map

        res_content["smplx_result"] = self.anno["raw_smplx"][frame_id]

        res = self.ret_type(**res_content)
        return res
    
    def __len__(self) -> int:
        return self.len

    def frame_shape(self):
        return FRAME_SHAPE

    def frame_id_to_index(self, mocap_frame_id):
        return self.frame_id_list.index(mocap_frame_id)
