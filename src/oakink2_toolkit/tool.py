import numpy as np
import torch


def index_param(param, index_list, l_pad=0, r_pad=0):
    res = {}
    index_mask = torch.zeros(
        (l_pad + len(index_list) + r_pad,), dtype=torch.bool, device=param[next(iter(param.keys()))].device
    )
    index_mask[l_pad : l_pad + len(index_list)] = True
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


def slice_param(param, batch_size=1):
    key_list = list(param.keys())

    total_sample = param[key_list[0]].shape[0]
    num_batch = (total_sample + batch_size - 1) // batch_size

    res = []
    for batch_id in range(num_batch):
        store = {}
        start = batch_id * batch_size
        end = min((batch_id + 1) * batch_size, total_sample)
        for k in key_list:
            store[k] = param[k][start:end]
        res.append(store)
    return res


def merge_param(param_list):
    key_list = list(param_list[0].keys())

    res = {}
    for k in key_list:
        _res = []
        for param in param_list:
            _res.append(param[k])
        _res = torch.cat(_res, dim=0)
        res[k] = _res
    return res
