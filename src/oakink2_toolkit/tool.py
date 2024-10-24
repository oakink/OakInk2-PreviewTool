import numpy as np
import torch


def index_param(param, index_list, l_pad=0, r_pad=0):
    res = {}
    index_mask = torch.zeros((l_pad + len(index_list) + r_pad,), dtype=torch.bool, device=param[next(iter(param.keys()))].device)
    index_mask[l_pad:-r_pad] = True
    for k, v in param.items():
        _content = v[index_list, ...]
        _l_pad = torch.zeros((l_pad, *_content.shape[1:]), dtype=_content.dtype, device=_content.device)
        _r_pad = torch.zeros((r_pad, *_content.shape[1:]), dtype=_content.dtype, device=_content.device)
        res[k] = torch.cat([_l_pad, _content, _r_pad], dim=0)
    return res, index_mask


def zero_param(param, length):
    res = {}
    index_mask = torch.zeros((length,), dtype=torch.bool, device=param[next(iter(param.keys()))].device)
    for k, v in param.items():
        res[k] = torch.zeros((length, *v.shape[1:]), dtype=v.dtype, device=v.device)
    return res, index_mask
