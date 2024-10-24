import sys
import argparse

import os
import logging
import torch
import numpy as np
import cv2
import itertools
import pickle
import json
import trimesh

from config_reg import ConfigRegistry, ConfigEntrySource, ConfigEntryCallback
from config_reg import ConfigEntryCommandlineBoolPattern, ConfigEntryCommandlineSeqPattern
from config_reg.callback import abspath_callback
from oakink2_preview.util.upkeep.opt import argdict_to_string
from oakink2_preview.util.console_io import suppress_trimesh_logging
from oakink2_preview.util.upkeep import log as log_upkeep
from oakink2_preview.transform.transform_np import transf_point_array_np
from oakink2_preview.util.vis_cv2_util import caption_combined_view
from oakink2_preview.util.vis_pyrender_util import PyMultiObjRenderer

from oakink2_toolkit.dataset import OakInk2__Dataset
from oakink2_toolkit.structure import OakInk2__Affordance, OakInk2__PrimitiveTask, OakInk2__ComplexTask
from oakink2_toolkit.meta import VIDEO_SHAPE
from oakink2_toolkit.tool import slice_param, merge_param

from manotorch.manolayer import ManoLayer, MANOOutput

PROG = os.path.splitext(os.path.basename(__file__))[0]
THIS_FILE = os.path.normcase(os.path.normpath(__file__))
THIS_DIR = os.path.dirname(THIS_FILE)
WS_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", ".."))
CURR_WORKING_DIR = os.getcwd()

# global vars
_logger = logging.getLogger(__name__)


def reg_entry(config_reg: ConfigRegistry):
    config_reg.register(
        "data.prefix",
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        required=True,
        default=os.path.join(WS_DIR, "data"),
    )
    config_reg.register(
        "seq_key",
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        required=True,
    )
    config_reg.register(
        "primitive_identifier",
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        required=True,
        default=None,
    )

    config_reg.register(
        "mano.mano_path",
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        required=True,
        default=os.path.join(WS_DIR, "asset", "mano_v1_2"),
    )


def reg_extract(config_reg: ConfigRegistry):
    cfg = config_reg.select(strip=True)
    return cfg


CAM_NAME = "allocentric_top"


def _face_lh(mano_layer_lh):
    _close_faces = torch.Tensor(
        [
            [92, 38, 122],
            [234, 92, 122],
            [239, 234, 122],
            [279, 239, 122],
            [215, 279, 122],
            [215, 122, 118],
            [215, 118, 117],
            [215, 117, 119],
            [215, 119, 120],
            [215, 120, 108],
            [215, 108, 79],
            [215, 79, 78],
            [215, 78, 121],
            [214, 215, 121],
        ]
    )
    _th_closed_faces = torch.cat([mano_layer_lh.th_faces.clone().detach().cpu(), _close_faces[:, [2, 1, 0]].long()])
    hand_faces_lh = _th_closed_faces.cpu().numpy()
    return hand_faces_lh


def run(run_cfg):
    dtype = torch.float32
    device = torch.device("cpu")

    mano_layer_rh = ManoLayer(
        mano_assets_root=run_cfg["mano"]["mano_path"],
        rot_mode="quat",
        side="right",
        center_idx=0,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)
    hand_faces_rh = mano_layer_rh.get_mano_closed_faces().cpu().numpy()
    mano_layer_lh = ManoLayer(
        mano_assets_root=run_cfg["mano"]["mano_path"],
        rot_mode="quat",
        side="left",
        center_idx=0,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)
    hand_faces_lh = _face_lh(mano_layer_lh)

    oakink2_dataset = OakInk2__Dataset(
        dataset_prefix=run_cfg["data"]["prefix"],
        return_instantiated=True,
    )
    complex_task_data = oakink2_dataset.load_complex_task(run_cfg["seq_key"])
    primitive_task_data_list = oakink2_dataset.load_primitive_task(complex_task_data, run_cfg["primitive_identifier"])
    if isinstance(primitive_task_data_list, OakInk2__PrimitiveTask):
        primitive_task_data_list = [primitive_task_data_list]
    # load all objects in the scene
    obj_map = {}
    for obj_part_id in complex_task_data.scene_obj_list:
        obj_map[obj_part_id] = oakink2_dataset.load_affordance(obj_part_id).obj_mesh

    # load cam_extr and cam_intr
    anno_filepath = os.path.join(oakink2_dataset.anno_prefix, f"{complex_task_data.seq_token}.pkl")
    with open(anno_filepath, "rb") as ifs:
        anno_info = pickle.load(ifs)
    cam_intr = next(iter(anno_info["cam_intr"][CAM_NAME].values()))
    cam_extr = next(iter(anno_info["cam_extr"][CAM_NAME].values()))

    # create renderer
    renderer = PyMultiObjRenderer(
        width=VIDEO_SHAPE[0], height=VIDEO_SHAPE[1], obj_map=obj_map, cam_intr=cam_intr, raymond=True
    )
    for ptask_data in primitive_task_data_list:
        lh_out, rh_out = {}, {}
        if ptask_data.frame_range_lh is not None:
            j_list, v_list = [], []
            lh_pose_info = {}
            # index pose_info with mask
            for k, v in ptask_data.lh_param.items():
                lh_pose_info[k] = v[ptask_data.lh_in_range_mask].to(dtype=dtype, device=device)
            for lh_pose_item in slice_param(lh_pose_info, batch_size=100):
                mano_out_sl = mano_layer_lh(pose_coeffs=lh_pose_item["pose_coeffs"], betas=lh_pose_item["betas"])
                j_sl = mano_out_sl.joints + lh_pose_item["tsl"].unsqueeze(1)
                v_sl = mano_out_sl.verts + lh_pose_item["tsl"].unsqueeze(1)
                j = j_sl.clone().cpu().numpy()
                v = v_sl.clone().cpu().numpy()
                j = transf_point_array_np(cam_extr, j)
                v = transf_point_array_np(cam_extr, v)
                j_list.append(j)
                v_list.append(v)
            j_traj = np.concatenate(j_list, axis=0)
            v_traj = np.concatenate(v_list, axis=0)
            lh_out["j"] = j_traj
            lh_out["v"] = v_traj
        if ptask_data.frame_range_rh is not None:
            j_list, v_list = [], []
            rh_pose_info = {}
            # index pose_info with mask
            for k, v in ptask_data.rh_param.items():
                rh_pose_info[k] = v[ptask_data.rh_in_range_mask].to(dtype=dtype, device=device)
            for rh_pose_item in slice_param(rh_pose_info, batch_size=100):
                mano_out_sl = mano_layer_rh(pose_coeffs=rh_pose_item["pose_coeffs"], betas=rh_pose_item["betas"])
                j_sl = mano_out_sl.joints + rh_pose_item["tsl"].unsqueeze(1)
                v_sl = mano_out_sl.verts + rh_pose_item["tsl"].unsqueeze(1)
                j = j_sl.clone().cpu().numpy()
                v = v_sl.clone().cpu().numpy()
                j = transf_point_array_np(cam_extr, j)
                v = transf_point_array_np(cam_extr, v)
                j_list.append(j)
                v_list.append(v)
            j_traj = np.concatenate(j_list, axis=0)
            v_traj = np.concatenate(v_list, axis=0)
            rh_out["j"] = j_traj
            rh_out["v"] = v_traj
        # viz
        viz_step = 60
        for fid in range(ptask_data.frame_range[0], ptask_data.frame_range[1], viz_step):
            extra_mesh = []
            offset = fid - ptask_data.frame_range[0]
            # index obj
            obj_pose_map = {}
            for obj_id in ptask_data.task_obj_list:
                obj_pose_map[obj_id] = cam_extr @ ptask_data.obj_transf[obj_id][offset]
            for obj_id in ptask_data.scene_obj_list:
                if obj_id not in ptask_data.task_obj_list:
                    obj_pose_map[obj_id] = None
            if ptask_data.frame_range_lh is not None:
                offset_lh = fid - ptask_data.frame_range_lh[0]
                v_lh = lh_out["v"][offset_lh]
                extra_mesh.append(trimesh.Trimesh(vertices=v_lh, faces=hand_faces_lh))
            if ptask_data.frame_range_rh is not None:
                offset_rh = fid - ptask_data.frame_range_rh[0]
                v_rh = rh_out["v"][offset_rh]
                extra_mesh.append(trimesh.Trimesh(vertices=v_rh, faces=hand_faces_rh))
            img = renderer(
                obj_pose_map=obj_pose_map,
                extra_mesh=extra_mesh,
                stick=True,
                background=np.ones((VIDEO_SHAPE[1], VIDEO_SHAPE[0], 3), dtype=np.uint8) * 255,
            )
            img = caption_combined_view(img, ptask_data.task_desc)

            while True:
                cv2.imshow("x", img)
                key = cv2.waitKey(1)
                if key == ord("\r"):
                    break


def main():
    # region: program setup
    log_upkeep.log_init()
    log_upkeep.enable_console()

    config_reg = ConfigRegistry(prog=PROG)
    reg_entry(config_reg)

    parser = argparse.ArgumentParser(prog=PROG)
    config_reg.hook(parser)
    config_reg.parse(parser)

    run_cfg = reg_extract(config_reg)

    _logger.info("run_cfg: %s", argdict_to_string(run_cfg))

    suppress_trimesh_logging()
    # endregion

    # region: run
    run(run_cfg)
    # endregion


if __name__ == "__main__":
    main()
