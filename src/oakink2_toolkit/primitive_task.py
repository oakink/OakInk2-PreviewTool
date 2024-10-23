from __future__ import annotations

from typing import Optional
from .type_def import NamedData

import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class OakInk2__PrimitiveTask(NamedData):
    instantiated = False

    seq_key: str = None
    seq_token: str = None

    primitive_task: str = None
    task_desc: str = None
    transient: bool = None

    hand_involved: str = None
    interaction_mode: str = None

    frame_range: tuple[int] = None
    frame_range_lh: Optional[tuple[int]] = None
    frame_range_rh: Optional[tuple[int]] = None

    scene_obj_list: list[str] = None
    task_obj_list: list[str] = None
    lh_obj_list: list[str] = None
    rh_obj_list: list[str] = None

    # field to be instantiated
    smplx_param: Optional[dict[int, dict[str, torch.Tensor]]] = None
    lh_hand_param: Optional[dict[int, dict[str, torch.Tensor]]] = None
    rh_param: Optional[dict[int, dict[str, torch.Tensor]]] = None
    lh_in_range_mask: Optional[torch.Tensor] = None
    rh_in_range_mask: Optional[torch.Tensor] = None
    obj_transf: Optional[dict[str, dict[int, torch.Tensor]]] = None
