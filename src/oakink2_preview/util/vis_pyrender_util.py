from typing import Dict, List

import cv2
import numpy as np
import pyglet
import pyrender
import trimesh
from copy import deepcopy


class PyRenderer:
    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), matrix=matrix))

        return nodes

    def __init__(self, width, height, obj, cam_intr, raymond=False) -> None:
        # os.environ["PYOPENGL_PLATFORM"] = "egl"
        self.width = width
        self.height = height
        self.r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
        self.PYRENDER_EXTR = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.cam_intr = cam_intr
        cx, cy, fx, fy = (
            self.cam_intr[0, 2],
            self.cam_intr[1, 2],
            self.cam_intr[0, 0],
            self.cam_intr[1, 1],
        )
        self.camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        self.scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2], bg_color=[0.0, 0.0, 0.0])
        self.scene.add(self.camera)
        if obj is None:
            try:
                from shapely import geometry

                poly = geometry.Polygon([[0.04, -0.03], [-0.03, -0.03], [-0.03, 0.06]])
                obj = trimesh.primitives.Extrusion(polygon=poly, height=0.08)
            except:
                obj = trimesh.primitives.Box(extents=(0.04, 0.06, 0.08))
        object = pyrender.Mesh.from_trimesh(obj)
        self.node_obj = pyrender.Node(mesh=object, matrix=np.eye(4))
        self.scene.add_node(self.node_obj)
        if raymond:
            self.light_list = self._create_raymond_lights()
            for light_node in self.light_list:
                self.scene.add_node(light_node)

    def __call__(
        self,
        obj_pose,
        background=None,
        alpha=False,
        extra_mesh=None,
        stick=False,
        blend=0.6,
    ):
        self.scene.set_pose(self.node_obj, pose=self.PYRENDER_EXTR @ obj_pose)

        if extra_mesh is not None:
            extra_node = []
            for emh in extra_mesh:
                emh_pr = pyrender.Mesh.from_trimesh(emh)
                end_pr = pyrender.Node(mesh=emh_pr, matrix=np.eye(4))
                self.scene.add_node(end_pr)
                extra_node.append(end_pr)

        color, depth = self.r.render(
            self.scene,
            flags=pyrender.RenderFlags.NONE | pyrender.RenderFlags.RGBA if alpha else 0,
        )
        color = color.copy()
        if background is not None:
            background = background[:, :, (2, 1, 0)]
            if alpha:
                mask = np.stack((depth, depth, depth, depth), axis=-1)
                background = np.concatenate([background, np.ones_like(background[:, :, [0]]) * 127], axis=-1)
                color[:, :, 3] = color[:, :, 3] * 127
                np.putmask(color, mask == 0, background)
            else:
                mask = np.stack((depth, depth, depth), axis=-1)
                if not stick:
                    color[:, :, :] = blend * color[:, :, :] + (1 - blend) * background[:, :, :]
                else:
                    np.putmask(color, mask == 0, background)

        if extra_mesh is not None:
            for ee in extra_node:
                self.scene.remove_node(ee)

        return color[:, :, (2, 1, 0, 3)] if alpha else color[:, :, (2, 1, 0)]


class PyMultiObjRenderer:
    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), matrix=matrix))

        return nodes

    def __init__(self, width, height, obj_map, cam_intr, raymond=False) -> None:
        # os.environ["PYOPENGL_PLATFORM"] = "egl"
        self.width = width
        self.height = height
        self.r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
        self.PYRENDER_EXTR = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.cam_intr = cam_intr
        cx, cy, fx, fy = (
            self.cam_intr[0, 2],
            self.cam_intr[1, 2],
            self.cam_intr[0, 0],
            self.cam_intr[1, 1],
        )
        self.camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        self.scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2], bg_color=[0.0, 0.0, 0.0])
        self.scene.add(self.camera)

        # handle obj_map
        self.node_obj_map = {}
        for obj_name, obj in obj_map.items():
            if obj is None:
                try:
                    from shapely import geometry

                    poly = geometry.Polygon([[0.04, -0.03], [-0.03, -0.03], [-0.03, 0.06]])
                    obj = trimesh.primitives.Extrusion(polygon=poly, height=0.08)
                except:
                    obj = trimesh.primitives.Box(extents=(0.04, 0.06, 0.08))
            # create material if has texture
            if obj.visual.kind == "face":
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorTexture=pyrender.Texture(
                        source=pyrender.ImageData(
                            obj.visual.image,
                            "RGB",
                        )
                    )
                )
                object = pyrender.Mesh.from_trimesh(obj, material=material)
            else:
                object = pyrender.Mesh.from_trimesh(obj)
            node_obj = pyrender.Node(mesh=object, matrix=np.eye(4))
            self.node_obj_map[obj_name] = node_obj
            self.scene.add_node(node_obj)

        if raymond:
            self.light_list = self._create_raymond_lights()
            for light_node in self.light_list:
                self.scene.add_node(light_node)

    def __call__(
        self,
        obj_pose_map,
        extra_mesh=None,
        background=None,
        alpha=False,
        stick=False,
        blend=0.6,
        extra_flags=0,
        seg=False,
    ):
        for obj_name, obj_pose in obj_pose_map.items():
            if obj_pose is not None:
                node_obj = self.node_obj_map[obj_name]
                node_obj.mesh.is_visible = True
                self.scene.set_pose(node_obj, pose=self.PYRENDER_EXTR @ obj_pose)
            else:
                node_obj = self.node_obj_map[obj_name]
                node_obj.mesh.is_visible = False

        if extra_mesh is not None:
            extra_node = []
            for emh in extra_mesh:
                emh.vertices = (self.PYRENDER_EXTR[:3, :3] @ emh.vertices.T).T + self.PYRENDER_EXTR[:3, 3]
                emh_pr = pyrender.Mesh.from_trimesh(emh)
                end_pr = pyrender.Node(mesh=emh_pr, matrix=np.eye(4))
                self.scene.add_node(end_pr)
                extra_node.append(end_pr)

        if seg:
            # get a map of node to obj & extra mesh
            nmmap = {}
            for obj_name, node in self.node_obj_map.items():
                nmmap[node] = obj_name
            for _off in range(len(extra_mesh)):
                end_pr = extra_node[_off]
                nmmap[end_pr] = f"_extra:{_off}"

            all_node_list = list(v for k, v in self.node_obj_map.items() if obj_pose_map[k] is not None) + extra_node
            avail_node_list = [el for el in all_node_list if el in self.scene.mesh_nodes]

            nm = {node: 1 * (i + 1) for i, node in enumerate(avail_node_list)}
            color = self.r.render(self.scene, pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.SEG, nm)[0]

            # use name version of nm
            nm_ = {}
            for node, seg_id in nm.items():
                nm_[nmmap[node]] = seg_id

            if extra_mesh is not None:
                for ee in extra_node:
                    self.scene.remove_node(ee)
            return color, nm_

        color, depth = self.r.render(
            self.scene,
            flags=pyrender.RenderFlags.NONE | (pyrender.RenderFlags.RGBA if alpha else 0) | extra_flags,
        )

        color = color.copy()
        if background is not None:
            background = background[:, :, (2, 1, 0)]
            if alpha:
                mask = np.stack((depth, depth, depth, depth), axis=-1)
                background = np.concatenate([background, np.ones_like(background[:, :, [0]]) * 127], axis=-1)
                color[:, :, 3] = color[:, :, 3] * 127
                np.putmask(color, mask == 0, background)
            else:
                mask = np.stack((depth, depth, depth), axis=-1)
                if not stick:
                    color[:, :, :] = blend * color[:, :, :] + (1 - blend) * background[:, :, :]
                else:
                    np.putmask(color, mask == 0, background)

        if extra_mesh is not None:
            for ee in extra_node:
                self.scene.remove_node(ee)

        return color[:, :, (2, 1, 0, 3)] if alpha else color[:, :, (2, 1, 0)]


class SelectViewer(pyrender.Viewer):
    @staticmethod
    def toggle_select_mode(viewer):
        viewer.in_select = not viewer.in_select
        if viewer.in_select:
            print("\nSelection Mode")
            viewer.print_select_prompt()
        else:
            print("\nDragging Mode")

    def print_press_prompt(self):
        print("Press 'S' to toggle selection mode")
        print("Press 'N' to select next marker")
        print("Press 'P' to select previous marker")
        print("Press 'Y' to save and quit")
        print("Press 'Q' to quit without saving")

    def print_select_prompt(self):
        print(f'\nPlease select marker #{self.select_index + 1} "{self.name_list[self.select_index]}"')
        if self.name_list[self.select_index] in self.mkrset_dict:
            print("CAUTION: You have selected this marker before, select now will OVERWRITE")
        print("...")

    @staticmethod
    def next(viewer):
        if viewer.in_select:
            viewer.select_index = (viewer.select_index + 1) % len(viewer.name_list)
            viewer.print_select_prompt()
        else:
            print("Not in selection mode")

    @staticmethod
    def prev(viewer):
        if viewer.in_select:
            viewer.select_index = (viewer.select_index - 1) % len(viewer.name_list)
            viewer.print_select_prompt()
        else:
            print("Not in selection mode")

    @staticmethod
    def save_and_quit(viewer):
        if viewer.ready_to_save:
            print("Saving and quiting...")
            viewer.close()
        else:
            print("Please select all markers first")

    @staticmethod
    def clear_and_quit(viewer):
        print("Quiting...")
        viewer.mkrset_dict.clear()
        viewer.close()

    def __init__(
        self, *arg, viewport_size=None, cam_intr=None, smplx_trimesh=None, name_list=None, mkrset_dict=None, **kwarg
    ):
        self.mkrset_dict = mkrset_dict
        self.select_index = 0
        self.select_count = 0

        # Sphere mesh for marker
        self.sphere_node_dict = {}

        # Read marker names
        self.name_list = []
        if name_list is not None:
            self.name_list = name_list

        # TODO: Display self.name_list[self.select_index] on screen

        # clear already defined key callbacks
        def empty_fn(*_arg, **_kwarg):
            return

        reg_k = {}
        for k in ["a", "c", "f", "h", "i", "l", "m", "n", "o", "r"]:
            reg_k[k] = empty_fn

        # status for selection
        self.in_select = False
        self.ready_to_save = False
        reg_k["s"] = self.toggle_select_mode
        reg_k["n"] = self.next
        reg_k["p"] = self.prev
        reg_k["y"] = self.save_and_quit
        reg_k["q"] = self.clear_and_quit

        # prompt
        self.print_press_prompt()

        # for compute
        self.ori_viewport_size = deepcopy(viewport_size)
        self.cam_intr = cam_intr
        self.PYRENDER_TRANSF = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.smplx_trimesh = smplx_trimesh

        super().__init__(viewport_size=viewport_size, registered_keys=reg_k, *arg, **kwarg)

    def on_mouse_press(self, x, y, buttons, modifiers):
        if self.in_select:
            cam_pose = self._camera_node.matrix.copy() @ self.PYRENDER_TRANSF
            coord = (
                x / self.viewport_size[0] * self.ori_viewport_size[0],
                (self.viewport_size[1] - y) / self.viewport_size[1] * self.ori_viewport_size[1],
            )
            f = self.cam_intr[0, 0]
            cx = self.cam_intr[0, 2]
            cy = self.cam_intr[1, 2]
            diff_vec = np.array((coord[0] - cx, coord[1] - cy, f), dtype=np.float32)
            diff_dirvec = diff_vec / np.linalg.norm(diff_vec)
            diff_dirvec = cam_pose[:3, :3] @ diff_dirvec

            ray_ori = cam_pose[:3, 3]
            ray_dir = diff_dirvec

            loc, _, f_idx = self.smplx_trimesh.ray.intersects_location(
                ray_origins=ray_ori[None],
                ray_directions=ray_dir[None],
            )
            intersected = len(loc) > 0
            if intersected:
                param = (loc - ray_ori) @ ray_dir
                pidx = np.argmin(param)
                target_pos = loc[pidx]
                target_face = f_idx[pidx]

                # Create local coordinate system and solve coefficients
                mkr_dict = {}
                mkr_dict["v_idx"] = self.smplx_trimesh.faces[target_face].tolist()
                origin = self.smplx_trimesh.vertices[mkr_dict["v_idx"][0]]
                x1 = self.smplx_trimesh.vertices[mkr_dict["v_idx"][1]] - origin
                x2 = self.smplx_trimesh.vertices[mkr_dict["v_idx"][2]] - origin
                e0 = np.cross(x1, x2)
                e0 = e0 / np.linalg.norm(e0)
                e1 = x1 / np.linalg.norm(x1)
                e2 = np.cross(e0, e1)
                A = np.array([e1, e2]).T
                b = target_pos - origin
                mkr_dict["coeff"] = (np.linalg.inv(A.T @ A) @ A.T @ b).tolist()

                # Check if the marker is already selected
                if self.name_list[self.select_index] not in self.mkrset_dict:
                    self.select_count += 1
                else:
                    print(f'Overwrite marker #{self.select_index + 1} "{self.name_list[self.select_index]}"')
                    self._scene.remove_node(self.sphere_node_dict[self.name_list[self.select_index]])
                sphere = trimesh.creation.uv_sphere(radius=0.005)
                sphere.visual.vertex_colors = np.array([0.8, 0.0, 0.0, 0.8])
                tf = np.eye(4)
                tf[:3, 3] = target_pos
                sphere_mesh = pyrender.Mesh.from_trimesh(sphere, poses=tf[None])
                sphere_node = pyrender.Node(mesh=sphere_mesh)
                self._scene.add_node(sphere_node)
                self.sphere_node_dict[self.name_list[self.select_index]] = sphere_node

                # Save to dict
                self.mkrset_dict[self.name_list[self.select_index]] = mkr_dict
                print(f"Done! ({self.select_count}/{len(self.name_list)})")
                if self.select_count == len(self.name_list):
                    self.ready_to_save = True
                    self.toggle_select_mode(self)
                    print("\nAll markers are selected")
                    self.print_press_prompt()
                else:
                    self.next(self)
        else:
            super().on_mouse_press(x, y, buttons, modifiers)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.in_select:
            pass
        else:
            super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    def on_mouse_release(self, x, y, button, modifiers):
        if self.in_select:
            pass
        else:
            super().on_mouse_release(x, y, button, modifiers)

    def on_mouse_scroll(self, x, y, dx, dy):
        if self.in_select:
            pass
        else:
            super().on_mouse_scroll(x, y, dx, dy)


def draw_arrows(trans_abs, scene):
    for j in range(trans_abs.shape[0]):
        axis = trimesh.creation.axis(
            transform=trans_abs[j], origin_size=0.002, origin_color=(1.5, 1.8, 4.7), axis_length=0.03
        )
        axis = pyrender.Mesh.from_trimesh(axis, smooth=False)
        scene.add(axis)
