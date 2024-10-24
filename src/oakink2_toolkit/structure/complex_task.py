from __future__ import annotations

from typing import Optional
from ..type_def import NamedData

import numpy as np
import torch
import networkx as nx
from dataclasses import dataclass


@dataclass
class OakInk2__ComplexTask(NamedData):
    instantiated: bool = False

    seq_key: str = None
    seq_token: str = None

    is_complex: bool = None  # num_affordance > 1
    exec_path: list[str] = None
    exec_path_affordance: list[str] = None  # no transient
    exec_range_map: dict[str, tuple[int, int]] = None
    pdg: nx.Graph = None

    task_target: str = None
    scene_desc: Optional[str] = None
    recipe: Optional[str] = None

    frame_range: Optional[list[int]] = None

    exec_dialog_list: Optional[tuple[str, str]] = None

    scene_obj_list: list[str] = None

    # field to be instantiated
    smplx_param: Optional[dict[int, dict[str, torch.Tensor]]] = None
    lh_param: Optional[dict[int, dict[str, torch.Tensor]]] = None
    rh_param: Optional[dict[int, dict[str, torch.Tensor]]] = None
    obj_transf: Optional[dict[str, dict[int, torch.Tensor]]] = None
