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


TRANSIENT_LIST = ["hold", "rearrange", "swap"]


def suffix_transient_primitive_segment(attr_store, transient_list=None):
    if transient_list is None:
        transient_list = TRANSIENT_LIST

    seg_name_map = {}

    counter = {}
    for k, v in attr_store.items():
        _prim = v["primitive"]
        if _prim not in transient_list:
            continue
        if _prim not in counter:
            name = f"{_prim}:{0:0>2}"
            counter[_prim] = 1
        else:
            name = f"{_prim}:{counter[_prim]:0>2}"
            counter[_prim] += 1
        seg_name_map[str(k)] = name
    return seg_name_map


def is_transient(primitive):
    return primitive in TRANSIENT_LIST


def frame_range_def_enclose(frame_range_def):
    if frame_range_def[0] is not None and frame_range_def[1] is None:
        return (frame_range_def[0][0], frame_range_def[0][1])
    elif frame_range_def[0] is None and frame_range_def[1] is not None:
        return (frame_range_def[1][0], frame_range_def[1][1])
    elif frame_range_def[0] is not None and frame_range_def[1] is not None:
        _beg = min(frame_range_def[0][0], frame_range_def[1][0])
        _end = max(frame_range_def[0][1], frame_range_def[1][1])
        return (_beg, _end)
    else:
        return None


def determine_hand_involved(frame_range_def):
    if frame_range_def[0] is not None and frame_range_def[1] is not None:
        return "bh"
    elif frame_range_def[0] is not None:
        return "lh"
    elif frame_range_def[1] is not None:
        return "rh"
    else:
        return None
