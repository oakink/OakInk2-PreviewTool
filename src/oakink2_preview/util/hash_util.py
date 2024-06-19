import pickle
import hashlib


def hash_attr_map(attr_map, trunc=8):
    attr_bin = pickle.dumps(attr_map)
    return hashlib.sha256(attr_bin).hexdigest()[:trunc]


def hash_str(attr_str, trunc=20):
    return hashlib.sha256(attr_str.encode()).hexdigest()[:trunc]
