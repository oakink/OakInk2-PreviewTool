from __future__ import annotations


from typing import Optional
from .type_def import NamedData
import trimesh
from dataclasses import dataclass


@dataclass
class OakInk2__Affordance(NamedData):
    instantiated: bool = False

    obj_id: str = None
    obj_part_id: list[str] = None
    obj_urdf_filepath: Optional[str] = None

    # field to be instantiated
    obj_part_mesh: Optional[dict[str, trimesh.Trimesh]] = None
