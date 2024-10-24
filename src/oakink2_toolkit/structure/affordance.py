from __future__ import annotations


from typing import Optional, Union
from ..type_def import NamedData
import trimesh
from dataclasses import dataclass


@dataclass
class OakInk2__Affordance(NamedData):
    instantiated: bool = False

    obj_id: str = None
    obj_name: str = None

    is_part: bool = False
    obj_instance_id: Optional[str] = None
    obj_part_id: Optional[list[str]] = None
    obj_urdf_filepath: Optional[str] = None

    affordance_list: list[str] = None

    # field to be instantiated
    obj_mesh: Optional[
        Union[
            trimesh.Trimesh,
            dict[str, trimesh.Trimesh],
        ]
    ] = None
