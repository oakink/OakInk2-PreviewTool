def seg_key_pair_to_frame_range(seg_pair):
    if seg_pair[0] is not None and seg_pair[1] is None:
        return (seg_pair[0][0], seg_pair[0][1])
    elif seg_pair[0] is None and seg_pair[1] is not None:
        return (seg_pair[1][0], seg_pair[1][1])
    elif seg_pair[0] is not None and seg_pair[1] is not None:
        _beg = min(seg_pair[0][0], seg_pair[1][0])
        _end = max(seg_pair[0][1], seg_pair[1][1])
        return (_beg, _end)
    else:
        return None


def suffix_affordance_primitive_segment(attr_store):
    seg_name_map = {}

    counter = {}
    for k, v in attr_store.items():
        _prim = v["primitive"]
        if _prim in ["hold", "rearrange", "swap", "?(unk)"]:
            continue
        if _prim not in counter:
            name = f"{_prim}:{0:0>2}"
            counter[_prim] = 1
        else:
            name = f"{_prim}:{counter[_prim]:0>2}"
            counter[_prim] += 1
        seg_name_map[str(k)] = name
    return seg_name_map
