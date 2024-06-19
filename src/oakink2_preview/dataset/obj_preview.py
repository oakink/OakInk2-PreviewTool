import os
import trimesh


def load_obj_map(obj_prefix: str, obj_list: list[str]):
    res = {}
    for obj_id in obj_list:
        obj_filedir = os.path.join(obj_prefix, obj_id)
        candidate_list = [el for el in os.listdir(obj_filedir) if os.path.splitext(el)[-1] in [".obj", ".ply"]]
        assert len(candidate_list) == 1
        obj_filename = candidate_list[0]
        obj_filepath = os.path.join(obj_filedir, obj_filename)
        if os.path.splitext(obj_filename)[-1] == ".obj":
            mesh = trimesh.load_mesh(obj_filepath, process=False, skip_materials=True, force="mesh")
        else:
            mesh = trimesh.load(obj_filepath, process=False)
        res[obj_id] = mesh
    return res
