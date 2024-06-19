import torch

from ..transform.rotation import quat_to_rotvec, rotvec_to_quat

ROT_FIELD_LIST = [
    "world_rot",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
]

def cvt_quat_to_rotvec(input_map):
    res = input_map.copy()
    for field in ROT_FIELD_LIST:
        res[field] = quat_to_rotvec(input_map[field])
    return res

def cvt_rotvec_to_quat(input_map):
    res = input_map.copy()
    for field in ROT_FIELD_LIST:
        res[field] = rotvec_to_quat(input_map[field])
    return res