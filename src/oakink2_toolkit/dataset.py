from __future__ import annotations

import os
import numpy as np
import torch
import trimesh
import networkx as nx
import json
import pickle
import ast
import typing

if typing.TYPE_CHECKING:
    from typing import Optional

from . import meta
from . import program
from .affordance import OakInk2__Affordance
from .primitive_task import OakInk2__PrimitiveTask
from .complex_task import OakInk2__ComplexTask


def load_obj(obj_prefix: str, obj_id: str):
    obj_filedir = os.path.join(obj_prefix, obj_id)
    candidate_list = [el for el in os.listdir(obj_filedir) if os.path.splitext(el)[-1] in [".obj", ".ply"]]
    assert len(candidate_list) == 1
    obj_filename = candidate_list[0]
    obj_filepath = os.path.join(obj_filedir, obj_filename)
    if os.path.splitext(obj_filename)[-1] == ".obj":
        mesh = trimesh.load_mesh(obj_filepath, process=False, skip_materials=True, force="mesh")
    else:
        mesh = trimesh.load(obj_filepath, process=False)
    return mesh


def load_json(json_filepath: str):
    with open(json_filepath, "r") as f:
        data = json.load(f)
    return data


def try_load_json(json_filepath: str):
    if os.path.exists(json_filepath):
        with open(json_filepath, "r") as ifs:
            data = json.load(ifs)
    else:
        data = None
    return data


class OakInk2__Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_prefix: str,
        return_instantiated: bool = False,
    ):
        self.dataset_prefix = dataset_prefix

        self.data_prefix = os.path.join(self.dataset_prefix, "data")
        self.anno_prefix = os.path.join(self.dataset_prefix, "anno_preview")
        self.obj_prefix = os.path.join(self.dataset_prefix, "obj_preview")
        self.program_prefix = os.path.join(self.dataset_prefix, "program")
        self.program_extension_prefix = os.path.join(self.dataset_prefix, "program_extension")

        task_target_filepath = os.path.join(self.program_prefix, "task_target.json")
        self.task_target = load_json(task_target_filepath)
        self.program_info_filedir = os.path.join(self.program_prefix, "program_info")
        self.pdg_filedir = os.path.join(self.program_prefix, "pdg")
        self.desc_info_filedir = os.path.join(self.program_prefix, "desc_info")
        self.initial_condition_info_filedir = os.path.join(self.program_prefix, "initial_condition_info")

        self.all_seq_list = list(self.task_target.keys())

        # mode
        self.return_instantiated = return_instantiated

    def __getitem__(self, index):
        seq_key = self.all_seq_list[index]
        res = self.load_complex_task(seq_key)
        if self.return_instantiated:
            self.instantiate_complex_task(res)
        return res

    def __len__(self):
        return len(self.all_seq_list)

    def instantiate_affordance(self, affordance_data: OakInk2__Affordance):
        if not affordance_data.instantiated:
            affordance_data.obj_part_mesh = {}
            for obj_part_id in affordance_data.obj_part_id:
                obj_part_mesh = load_obj(self.obj_prefix, obj_part_id)
                affordance_data.obj_part_mesh[obj_part_id] = obj_part_mesh
            affordance_data.instantiated = True
        return affordance_data

    def instantiate_primitive_task(
        self, primitive_task_data: OakInk2__PrimitiveTask, complex_task_data: Optional[OakInk2__ComplexTask] = None
    ):
        if not primitive_task_data.instantiated:
            if complex_task_data is not None and complex_task_data.instantiated:
                # use frame_range_lh and frame_range_rh to quick index the tensor
                # and pad with zeros to be the same length with frame_range
                frame_range_lh, frame_range_rh = primitive_task_data.frame_range_lh, primitive_task_data.frame_range_rh
                frame_list_lh = list(range(frame_range_lh[0], frame_range_lh[1]))
                frame_list_rh = list(range(frame_range_rh[0], frame_range_rh[1]))
            else:
                # load...
                pass
            primitive_task_data.instantiated = True
        return primitive_task_data

    def instantiate_complex_task(self, complex_task_data: OakInk2__ComplexTask):
        if not complex_task_data.instantiated:
            # load annotation file
            anno_filepath = os.path.join(self.anno_prefix, f"{complex_task_data.seq_token}.pkl")
            with open(anno_filepath, "rb") as ifs:
                anno_data = pickle.load(ifs)
            # frame range
            mocap_frame_id_list = anno_data["mocap_frame_id_list"]
            frame_range = (min(mocap_frame_id_list), max(mocap_frame_id_list))
            complex_task_data.frame_range = frame_range
            # obj list
            obj_list = anno_data["obj_list"]
            complex_task_data.scene_obj_list = obj_list
            # smplx param
            smplx_param = self._collect_smplx(anno_data, mocap_frame_id_list)
            complex_task_data.smplx_param = smplx_param
            # left_hand param
            lh_param = self._collect_mano(anno_data, "lh", mocap_frame_id_list)
            complex_task_data.lh_param = lh_param
            # right hand param
            rh_param = self._collect_mano(anno_data, "rh", mocap_frame_id_list)
            complex_task_data.rh_param = rh_param
            # obj transf
            obj_transf = self._collect_obj_transf(anno_data, obj_list, mocap_frame_id_list)
            complex_task_data.obj_transf = obj_transf
            # conclude
            complex_task_data.instantiated = True
        return complex_task_data

    def _collect_smplx(self, anno_data, mocap_frame_id_list):
        smplx_handle = anno_data["raw_smplx"]
        smplx_key_list = list(next(iter(smplx_handle.values())).keys())
        smplx_param = {}
        for k in smplx_key_list:
            _param_tensor = []
            for fid in mocap_frame_id_list:
                _param_tensor.append(smplx_handle[fid][k])
            _param_tensor = torch.cat(_param_tensor, dim=0)
            smplx_param[k] = _param_tensor
        return smplx_param

    def _collect_mano(self, anno_data, hand_side, mocap_frame_id_list):
        mano_bh_handle = anno_data["raw_mano"]
        mano_bh_key_list = list(next(iter(mano_bh_handle.values())).keys())
        # only pick key prefixed with hand_side. also remove f"{hand_side}__" to get underlying key
        ori_key_list, key_list, key_prefix = [], [], f"{hand_side}__"
        for k in mano_bh_key_list:
            if k.startswith(key_prefix):
                ori_key_list.append(k)
                key_list.append(k[len(key_prefix) :])
        # collect res
        mano_param = {}
        for k, ori_key in zip(key_list, ori_key_list):
            _param_tensor = []
            for fid in mocap_frame_id_list:
                _param_tensor.append(mano_bh_handle[fid][ori_key])
            _param_tensor = torch.cat(_param_tensor, dim=0)
            mano_param[k] = _param_tensor
        return mano_param

    def _collect_obj_transf(self, anno_data, obj_list, mocap_frame_id_list):
        obj_transf_handle = anno_data["obj_transf"]
        res = {}
        for obj_id in obj_list:
            obj_transf_curr = obj_transf_handle[obj_id]
            _res = []
            for fid in mocap_frame_id_list:
                _res.append(obj_transf_curr[fid])
            _res = np.stack(_res, axis=0)
            res[obj_id] = _res
        return res

    def load_complex_task(self, seq_key, return_instantiated=None):
        if return_instantiated is None:
            return_instantiated = self.return_instantiated

        seq_token = seq_key.replace("/", "++")

        # preparation
        program_info_filepath = os.path.join(self.program_info_filedir, f"{seq_token}.json")
        program_info = load_json(program_info_filepath)
        affordance_task_namemap = program.suffix_affordance_primitive_segment(program_info)
        transient_task_namemap = program.suffix_transient_primitive_segment(program_info)
        _full_task_namemap = {**affordance_task_namemap, **transient_task_namemap}
        # reorder the tasks according to program_info
        full_task_namemap = {}
        for k in program_info.keys():
            full_task_namemap[k] = _full_task_namemap[k]
        rev_full_task_namemap = {v: ast.literal_eval(k) for k, v in full_task_namemap.items()}

        is_complex = len(affordance_task_namemap) > 0
        exec_path = list(full_task_namemap.values())
        exec_path_affordance = list(affordance_task_namemap.values())
        exec_range_map = rev_full_task_namemap
        with open(os.path.join(self.pdg_filedir, f"{seq_token}.json"), "r") as ifs:
            _g_info = json.load(ifs)
            _g = nx.DiGraph()
            for k in _g_info["id_map"]:
                _g.add_node(affordance_task_namemap[k])
            _rev_id_map = {v: k for k, v in _g_info["id_map"].items()}
            for e in _g_info["e"]:
                _e_from, _e_to = e
                _seg_from, _seg_to = _rev_id_map[_e_from], _rev_id_map[_e_to]
                _g.add_edge(affordance_task_namemap[_seg_from], affordance_task_namemap[_seg_to])
        pdg = _g

        task_target = self.task_target[seq_key]
        # scene_desc & recipe
        _fpath = os.path.join(self.initial_condition_info_filedir, f"{seq_token}.json")
        if os.path.exists(_fpath):
            with open(_fpath, "r") as ifs:
                _i_info = json.load(ifs)
            scene_desc, recipe = _i_info["initial_condition"], _i_info["recipe"]
        else:
            scene_desc, recipe = None, None
        # frame_id
        if not return_instantiated:
            _fpath = os.path.join(self.program_extension_prefix, "frame_id", f"{seq_token}.pkl")
            if os.path.exists(_fpath):
                with open(_fpath, "rb") as ifs:
                    _fid_info = pickle.load(ifs)
                    frame_list = _fid_info["mocap_frame_id_list"]
                    frame_range = (min(frame_list), max(frame_list))
            else:
                frame_range = None
        else:
            frame_range = None
        # scene_obj_list
        if not return_instantiated:
            _fpath = os.path.join(self.program_extension_prefix, "obj_list", f"{seq_token}.json")
            scene_obj_list = try_load_json(_fpath)
        else:
            scene_obj_list = None

        res = OakInk2__ComplexTask(
            seq_key=seq_key,
            seq_token=seq_token,
            is_complex=is_complex,
            exec_path=exec_path,
            exec_path_affordance=exec_path_affordance,
            exec_range_map=exec_range_map,
            pdg=pdg,
            task_target=task_target,
            scene_desc=scene_desc,
            recipe=recipe,
            frame_range=frame_range,
            scene_obj_list=scene_obj_list,
        )
        return res

    def load_primitive_task(self, complex_task_data, primitive_identifier=None, return_instantiated=None):
        # if primitive_identifier is None, load all primitive tasks
        # else if it is a list, load the primitive tasks with the identifiers in the list
        # else if it is a string, load the primitive task with the identifier
        if return_instantiated is None:
            return_instantiated = self.return_instantiated

    def _load_primitive_task_from_def(
        self,
        seq_key,
        frame_range_def,
        return_instantiated=None,
        program_info=None,
        desc_info=None,
    ):
        # internal method to load a primitive task *from ground up*
        if return_instantiated is None:
            return_instantiated = self.return_instantiated
        seq_token = seq_key.replace("/", "++")
        # preparation
        if program_info is None:
            program_info_filepath = os.path.join(self.program_info_filedir, f"{seq_token}.json")
            program_info = load_json(program_info_filepath)
        if desc_info is None:
            desc_info_filepath = os.path.join(self.desc_info_filedir, f"{seq_token}.json")
            desc_info = load_json(desc_info_filepath)
        # get the program annotations
        frame_range_def_key = str(frame_range_def)
        program_item = program_info[frame_range_def_key]

        frame_range = program.frame_range_def_enclose(frame_range_def)
        frame_range_lh = frame_range_def[0]
        frame_range_rh = frame_range_def[1]

        primitive = program_item["primitive"]
        task_desc = desc_info[frame_range_def_key]["seg_desc"]
        transient = program.is_transient(primitive)

        hand_involved = program.determine_hand_involved(frame_range_def)
        interaction_mode = program_item["interaction_mode"]

        # scene_obj_list
        if not return_instantiated:
            _fpath = os.path.join(self.program_extension_prefix, "obj_list", f"{seq_token}.json")
            scene_obj_list = try_load_json(_fpath)
        else:
            scene_obj_list = None
        # task_obj_list...
        task_obj_list = program_item["obj_list"]
        lh_obj_list = program_item["obj_list_lh"]
        rh_obj_list = program_item["obj_list_rh"]
        res = OakInk2__PrimitiveTask(
            seq_key=seq_key,
            seq_token=seq_token,
            primitive_task=primitive,
            task_desc=task_desc,
            transient=transient,
            hand_involved=hand_involved,
            interaction_mode=interaction_mode,
            frame_range=frame_range,
            frame_range_lh=frame_range_lh,
            frame_range_rh=frame_range_rh,
            scene_obj_list=scene_obj_list,
            task_obj_list=task_obj_list,
            lh_obj_list=lh_obj_list,
            rh_obj_list=rh_obj_list,
        )
        return res

    def _load_primitive_task_from_identifier(
        self,
        seq_key,
        primitive_identifier,
        return_instantiated=None,
        program_info=None,
        desc_info=None,
    ):
        # internal method to load a primitive task *from ground up*
        if return_instantiated is None:
            return_instantiated = self.return_instantiated
        seq_token = seq_key.replace("/", "++")  # preparation
        if program_info is None:
            program_info_filepath = os.path.join(self.program_info_filedir, f"{seq_token}.json")
            program_info = load_json(program_info_filepath)
        affordance_task_namemap = program.suffix_affordance_primitive_segment(program_info)
        transient_task_namemap = program.suffix_transient_primitive_segment(program_info)
        _full_task_namemap = {**affordance_task_namemap, **transient_task_namemap}
        # reorder the tasks according to program_info
        full_task_namemap = {}
        for k in program_info.keys():
            full_task_namemap[k] = _full_task_namemap[k]
        rev_full_task_namemap = {v: ast.literal_eval(k) for k, v in full_task_namemap.items()}

        # handle primitive def
        frame_range_def = rev_full_task_namemap[primitive_identifier]
        # load
        return self._load_primitive_task_from_def(
            seq_key=seq_key,
            frame_range_def=frame_range_def,
            return_instantiated=return_instantiated,
            program_info=program_info,
            desc_info=desc_info,
        )

    def load_affordance(self, obj_id, return_instantiated=None):
        pass
