import os
import numpy as np
import torch
import logging
import trimesh
import json
import pickle
from matplotlib import pyplot as plt
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui import QImage, QPixmap

from ..transform.transform_np import inv_transf_np, transf_point_array_np, project_point_array_np, assemble_T_np
from ..transform.rotation_np import rotvec_to_rotmat_np, rotmat_to_rotvec_np
from ..dataset.stream_preview import StreamDataset, CAMERA_LAYOUT_DESC
from ..dataset.obj_preview import load_obj_map
from ..util.vis_pyrender_util import PyMultiObjRenderer
from ..util.vis_cv2_util import combine_view
from ..layer import smplx
from ..util.hash_util import hash_str

SMPLX_ROT_MODE = "quat"
SMPLX_DIM_SHAPE_ALL = 300

# global vars
_logger = logging.getLogger(__name__)


def seg_pair_def_to_seg_def(seg_pair_def):
    if seg_pair_def[0] is not None and seg_pair_def[1] is None:
        return (seg_pair_def[0][0], seg_pair_def[0][1])
    elif seg_pair_def[0] is None and seg_pair_def[1] is not None:
        return (seg_pair_def[1][0], seg_pair_def[1][1])
    elif seg_pair_def[0] is not None and seg_pair_def[1] is not None:
        _beg = min(seg_pair_def[0][0], seg_pair_def[1][0])
        _end = max(seg_pair_def[0][1], seg_pair_def[1][1])
        return (_beg, _end)
    else:
        return None


class MainWindow(QtWidgets.QWidget):
    def __init__(
        self,
        stream_prefix,
        anno_prefix,
        object_prefix,
        program_prefix,
    ) -> None:
        super().__init__()

        # persistent data
        self.stream_prefix = stream_prefix
        self.anno_prefix = anno_prefix
        self.object_prefix = object_prefix
        self.program_prefix = program_prefix

        self.dtype = torch.float32
        self.device = torch.device(f"cuda:0")
        self.smplx_layer = smplx.SMPLXLayer(
            "asset/smplx_v1_1/models",
            dtype=self.dtype,
            rot_mode=SMPLX_ROT_MODE,
            num_betas=SMPLX_DIM_SHAPE_ALL,
            gender="neutral",
            use_body_upper_asset="asset/smplx_extra/body_upper_idx.pt",
        ).to(self.device)
        self.smplx_faces_np = self.smplx_layer.body_upper_faces.detach().clone().cpu().numpy()
        self.smplx_body_upper_idx = self.smplx_layer.body_upper_vert_idx.detach().clone().cpu().numpy()

        # mocap_dev data
        self.process_def_str = None
        self.stream_filedir = None
        self.anno_filepath = None
        self.stream_data = None
        self.obj_desc = None
        self.target = None
        self.obj_map = None
        self.renderer_map = None

        # obj
        self.n_obj = 0
        self.obj_id_list = []
        self.obj_render_set = set()

        # body mesh
        self.body_mesh_list = []
        self.body_mesh_draw_set = set()

        # total
        self.n_model = self.n_obj

        self.frame_list = []
        self.n_frame = 0

        # runtime
        self.curr_frame_offset = None
        self.curr_frame_data = None
        self.curr_frame_id = None
        self.curr_frame_img = None

        # set window title
        self.setWindowTitle("OakInk2-PreviewTool")

        # set top bar (file switch dialog) & reload button
        # file dialog
        self.widget_button_opennew = QtWidgets.QPushButton("switch sequence")
        self.widget_button_opennew.clicked.connect(self.widget_button_opennew_click)

        # line edit
        self.widget_stream_root_line_edit = QtWidgets.QLineEdit()

        # reload button
        self.widget_button_reload = QtWidgets.QPushButton("reload")
        self.widget_button_reload.clicked.connect(self.widget_button_reload_click)

        self.layout_top_bar_layout = QtWidgets.QHBoxLayout()
        self.layout_top_bar_layout.addWidget(self.widget_button_opennew)
        self.layout_top_bar_layout.addWidget(self.widget_stream_root_line_edit)
        self.layout_top_bar_layout.addWidget(self.widget_button_reload)

        # set progress bar
        self.widget_button_front = QtWidgets.QPushButton("|<")
        self.widget_button_prev = QtWidgets.QPushButton("<")
        self.widget_button_next = QtWidgets.QPushButton(">")
        self.widget_button_rear = QtWidgets.QPushButton(">|")

        self.widget_button_front.clicked.connect(self.widget_button_front_click)
        self.widget_button_prev.clicked.connect(self.widget_button_prev_click)
        self.widget_button_next.clicked.connect(self.widget_button_next_click)
        self.widget_button_rear.clicked.connect(self.widget_button_rear_click)

        self.widget_frame_ind = QtWidgets.QLineEdit()
        self.widget_frame_ind.setFixedSize(200, 60)
        font = self.widget_frame_ind.font()
        font.setPointSize(30)
        self.widget_frame_ind.setFont(font)
        self.widget_frame_ind.setStyleSheet("border: 1px solid black;")
        self.widget_frame_ind.returnPressed.connect(self.widget_frame_ind_return_pressed)

        self.widget_slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        self.widget_slider.setStyleSheet(
            """
.QSlider {
    min-height: 68px;
    max-height: 68px;
}

.QSlider::groove:horizontal {
    border: 1px solid #262626;
    height: 5px;
    margin: 0 12px;
}

.QSlider::handle:horizontal {
    background: #F0F0F0;
    border: 5px solid #101010;
    width: 23px;
    height: 100px;
    margin: -24px -12px;
}
"""
        )
        self.widget_slider.setMinimum(0)
        self.widget_slider.setMaximum(max(0, self.n_frame - 1))
        self.widget_slider.setValue(0)
        self.widget_slider.valueChanged[int].connect(self.slider_value_change)

        self.layout_progress_layout = QtWidgets.QHBoxLayout()
        self.layout_progress_layout.addWidget(self.widget_frame_ind)
        self.layout_progress_layout.addWidget(self.widget_button_front)
        self.layout_progress_layout.addWidget(self.widget_button_prev)
        self.layout_progress_layout.addWidget(self.widget_slider)
        self.layout_progress_layout.addWidget(self.widget_button_next)
        self.layout_progress_layout.addWidget(self.widget_button_rear)

        # set main content
        # table widget with each row a checker box and descriptions
        self.widget_model_table = QtWidgets.QTableWidget()
        self.widget_model_table.setRowCount(self.n_model)
        self.widget_model_table.setColumnCount(3)
        self.widget_model_table.setHorizontalHeaderLabels(["visible", "obj_id", "name"])
        header = self.widget_model_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)

        self.widget_frame_show = QtWidgets.QLabel()
        self.widget_frame_show.setBackgroundRole(QtGui.QPalette.Base)
        self.widget_frame_show.setMinimumWidth(848)

        self.widget_label_scenario = QtWidgets.QLabel()
        self.widget_label_scenario.setText("scenario: ")
        self.widget_label_scenario.setMinimumWidth(100)
        self.widget_label_scenario.setWordWrap(True)
        self.widget_label_target = QtWidgets.QLabel()
        self.widget_label_target.setText("target: ")
        self.widget_label_target.setMinimumWidth(100)
        self.widget_label_target.setWordWrap(True)
        self.widget_label_primitive_id = QtWidgets.QLabel("primitive id: ")
        self.widget_label_primitive_id.setMinimumWidth(100)
        self.widget_label_primitive_id.setWordWrap(True)
        self.widget_label_primitive_id_lh = QtWidgets.QLabel()
        self.widget_label_primitive_id_lh.setMinimumWidth(100)
        self.widget_label_primitive_id_lh.setWordWrap(True)
        self.widget_label_primitive_id_rh = QtWidgets.QLabel()
        self.widget_label_primitive_id_rh.setMinimumWidth(100)
        self.widget_label_primitive_id_rh.setWordWrap(True)
        self.widget_label_primitive_text = QtWidgets.QLabel("primtive text: ")
        self.widget_label_primitive_text.setMinimumWidth(100)
        self.widget_label_primitive_text.setWordWrap(True)
        self.widget_label_primitive_range_lh = QtWidgets.QLabel()
        self.widget_label_primitive_range_lh.setMinimumWidth(100)
        self.widget_label_primitive_range_lh.setWordWrap(True)
        self.widget_label_primitive_range_rh = QtWidgets.QLabel()
        self.widget_label_primitive_range_rh.setMinimumWidth(100)
        self.widget_label_primitive_range_rh.setWordWrap(True)

        self.widget_label_primitive_id_alt = QtWidgets.QLabel()
        self.widget_label_primitive_id_alt.setMinimumWidth(100)
        self.widget_label_primitive_id_alt.setWordWrap(True)
        self.widget_label_primitive_id_alt_lh = QtWidgets.QLabel()
        self.widget_label_primitive_id_alt_lh.setMinimumWidth(100)
        self.widget_label_primitive_id_alt_lh.setWordWrap(True)
        self.widget_label_primitive_id_alt_rh = QtWidgets.QLabel()
        self.widget_label_primitive_id_alt_rh.setMinimumWidth(100)
        self.widget_label_primitive_id_alt_rh.setWordWrap(True)
        self.widget_label_primitive_text_alt = QtWidgets.QLabel()
        self.widget_label_primitive_text_alt.setMinimumWidth(100)
        self.widget_label_primitive_text_alt.setWordWrap(True)
        self.widget_label_primitive_range_alt_lh = QtWidgets.QLabel()
        self.widget_label_primitive_range_alt_lh.setMinimumWidth(100)
        self.widget_label_primitive_range_alt_lh.setWordWrap(True)
        self.widget_label_primitive_range_alt_rh = QtWidgets.QLabel()
        self.widget_label_primitive_range_alt_rh.setMinimumWidth(100)
        self.widget_label_primitive_range_alt_rh.setWordWrap(True)

        self.widget_panel_right = QtWidgets.QWidget()
        self.layout_panel_right = QtWidgets.QVBoxLayout()
        self.layout_panel_right.addWidget(self.widget_label_scenario)
        self.layout_panel_right.addWidget(self.widget_label_target)
        _line = QtWidgets.QFrame()
        _line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        _line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.layout_panel_right.addWidget(_line)
        self.layout_panel_right.addWidget(self.widget_label_primitive_id)
        self.layout_panel_right.addWidget(self.widget_label_primitive_text)
        self.layout_panel_right.addWidget(self.widget_label_primitive_range_lh)
        self.layout_panel_right.addWidget(self.widget_label_primitive_id_lh)
        self.layout_panel_right.addWidget(self.widget_label_primitive_range_rh)
        self.layout_panel_right.addWidget(self.widget_label_primitive_id_rh)
        _line = QtWidgets.QFrame()
        _line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        _line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.layout_panel_right.addWidget(_line)
        self.layout_panel_right.addWidget(self.widget_label_primitive_id_alt)
        self.layout_panel_right.addWidget(self.widget_label_primitive_text_alt)
        self.layout_panel_right.addWidget(self.widget_label_primitive_range_alt_lh)
        self.layout_panel_right.addWidget(self.widget_label_primitive_id_alt_lh)
        self.layout_panel_right.addWidget(self.widget_label_primitive_range_alt_rh)
        self.layout_panel_right.addWidget(self.widget_label_primitive_id_alt_rh)
        _line = QtWidgets.QFrame()
        _line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        _line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.layout_panel_right.addWidget(_line)

        self.layout_panel_right.addStretch()
        self.widget_panel_right.setLayout(self.layout_panel_right)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.widget_model_table)
        splitter.addWidget(self.widget_frame_show)
        splitter.addWidget(self.widget_panel_right)
        splitter.splitterMoved.connect(self.widget_splitter_moved)
        splitter.setHandleWidth(20)

        self.layout_main_layout = QtWidgets.QHBoxLayout()
        self.layout_main_layout.addWidget(splitter)

        # set combined layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addLayout(self.layout_top_bar_layout)
        self.layout.addLayout(self.layout_progress_layout)
        self.layout.addLayout(self.layout_main_layout)

    def set_stream(self, stream_root):
        # clear old gui
        self.widget_slider.setMaximum(0)
        self.widget_slider.setValue(0)
        self.widget_frame_ind.setText("")
        self.widget_model_table.setRowCount(0)

        # region: construct everything!
        self.process_def_str = os.path.relpath(stream_root, self.stream_prefix)
        self.stream_filedir = stream_root
        process_key = self.process_def_str.replace("++", "/")
        # load data
        self.anno_filepath = os.path.join(self.anno_prefix, f"{self.process_def_str}.pkl")
        stream_data = StreamDataset(stream_filedir=stream_root, anno_filepath=self.anno_filepath)
        obj_list = stream_data.object_list
        obj_map = load_obj_map(os.path.join(self.object_prefix, "align_ds"), obj_list)

        # load program and desc
        _program_filepath = os.path.join(self.program_prefix, "program_info", f"{self.process_def_str}.json")
        program_info = {}
        with open(_program_filepath, "r") as ifs:
            _program_info = json.load(ifs)
            for k, v in _program_info.items():
                seg_pair_def = eval(k)
                program_info[seg_pair_def] = v
        self.program_info = program_info

        desc_info = {}
        _desc_filepath = os.path.join(self.program_prefix, "desc_info", f"{self.process_def_str}.json")
        with open(_desc_filepath, "r") as ifs:
            _desc_text = json.load(ifs)
            for k, v in _desc_text.items():
                seg_pair_def = eval(k)
                desc_info[seg_pair_def] = v
        self.desc_info = desc_info

        _obj_desc_filepath = os.path.join(self.object_prefix, "obj_desc.json")
        with open(_obj_desc_filepath, "r") as ifs:
            self.obj_desc = json.load(ifs)

        _target_desc_filepath = os.path.join(self.program_prefix, "task_target.json")
        with open(_target_desc_filepath, "r") as ifs:
            self.target = json.load(ifs)[process_key]

        if process_key.startswith("scene_01"):
            cur_scenario = "kitchen table"
        elif process_key.startswith("scene_02"):
            cur_scenario = "study room table"
        elif process_key.startswith("scene_03"):
            cur_scenario = "demo chem lab"
        elif process_key.startswith("scene_04"):
            cur_scenario = "bathroom table"
        self.scenario = cur_scenario

        # init renderer
        frame_shape = stream_data.frame_shape()
        frame_height, frame_width = frame_shape[:2]
        cm = plt.get_cmap("turbo")
        _obj_map = {}
        for _obj_id in obj_map:
            _obj_desc = self.obj_desc[_obj_id]
            _color = (np.array(cm(int(hash_str(_obj_id), 16) % 256))[0:3] ** 1.2) * 0.8 + 0.2
            _obj_model = obj_map[_obj_id]
            _v = _obj_model.vertices
            _f = _obj_model.faces
            _vc = _color.reshape((1, 3)).repeat(_v.shape[0], axis=0)
            _new_model = trimesh.Trimesh(vertices=_v, faces=_f, vertex_colors=_vc, process=False)
            _obj_map[_obj_id] = _new_model
        renderer_map = {}
        for cam_desc in CAMERA_LAYOUT_DESC:
            frame_data_init = stream_data[0]
            renderer_map[cam_desc] = PyMultiObjRenderer(
                width=frame_width,
                height=frame_height,
                obj_map=_obj_map,
                cam_intr=frame_data_init[f"cam_intr_{cam_desc}"],
                raymond=True,
            )
        # endregion

        # mocap_dev data
        self.stream_data = stream_data
        self.obj_map = obj_map
        self.renderer_map = renderer_map

        # obj
        self.n_obj = len(self.obj_map)
        self.obj_id_list = list(self.obj_map.keys())
        self.obj_render_set = set(self.obj_id_list)

        # body_mesh
        self.body_mesh_list = ["SMPLX"]
        self.body_mesh_draw_set = set(self.body_mesh_list)

        # table
        self.n_model = self.n_obj + len(self.body_mesh_draw_set)

        # frame navigation
        self.frame_list = self.stream_data.frame_id_list
        self.n_frame = len(self.stream_data)

        self.curr_frame_offset = 0
        self.update_stream_data()

        # update gui
        self.widget_stream_root_line_edit.setText(stream_root)
        self.widget_slider.setMaximum(max(0, self.n_frame - 1))
        self.widget_slider.setValue(0)

        # render table
        self.widget_model_table.setRowCount(self.n_model)
        for _offset in range(self.n_obj):
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.construct_table_row_check_obj(self.obj_id_list[_offset]))
            self.widget_model_table.setCellWidget(_offset, 0, checkbox)

            _obj_id = self.obj_id_list[_offset]
            _obj_desc = self.obj_desc[_obj_id]["obj_name"]
            item = QtWidgets.QTableWidgetItem(_obj_id)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.widget_model_table.setItem(_offset, 1, item)
            item = QtWidgets.QTableWidgetItem(_obj_desc)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.widget_model_table.setItem(_offset, 2, item)

        for _offset in range(len(self.body_mesh_draw_set)):
            base_offset = self.n_obj
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.construct_table_row_check_body_mesh(self.body_mesh_list[_offset]))
            self.widget_model_table.setCellWidget(_offset + base_offset, 0, checkbox)

            item = QtWidgets.QTableWidgetItem(self.body_mesh_list[_offset])
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.widget_model_table.setItem(_offset + base_offset, 1, item)

        # update frame
        self.update_frame()

        # update program, text desc
        self.widget_label_scenario.setText(f"scenario: {self.scenario}")
        self.widget_label_target.setText(f"target: {self.target}")

    @QtCore.Slot()
    def widget_button_opennew_click(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        file_dialog.setOptions(QtWidgets.QFileDialog.ShowDirsOnly)
        file_dialog.setDirectory(self.stream_prefix)

        if file_dialog.exec():
            _qdir = file_dialog.directory()
            stream_root = _qdir.absolutePath()
            self.set_stream(stream_root)

    @QtCore.Slot()
    def widget_button_reload_click(self):
        if self.stream_filedir is not None:
            self.set_stream(self.stream_filedir)

    @QtCore.Slot()
    def widget_frame_ind_return_pressed(self):
        text = self.widget_frame_ind.text()
        try:
            value = int(text)
        except ValueError:
            value = self.curr_frame_offset

        if value == self.curr_frame_offset:
            return
        else:
            # find the nearest frame
            value_offset = np.argmin(np.abs(np.array(self.frame_list) - value))
            self.curr_frame_offset = value_offset
            self.update_stream_data()

    @QtCore.Slot()
    def slider_value_change(self, value):
        value = int(value)
        value = max(0, value)
        value = min(self.n_frame - 1, value)
        self.curr_frame_offset = value
        self.update_stream_data()

    @QtCore.Slot()
    def widget_button_front_click(self):
        self.curr_frame_offset = 0
        self.update_stream_data()

    @QtCore.Slot()
    def widget_button_prev_click(self):
        self.curr_frame_offset = max(self.curr_frame_offset - 1, 0)
        self.update_stream_data()

    @QtCore.Slot()
    def widget_button_next_click(self):
        self.curr_frame_offset = min(self.curr_frame_offset + 1, self.n_frame - 1)
        self.update_stream_data()

    @QtCore.Slot()
    def widget_button_rear_click(self):
        self.curr_frame_offset = self.n_frame - 1
        self.update_stream_data()

    def construct_table_row_check_obj(self, obj_id):
        @QtCore.Slot()
        def model_table_item_check(state):
            if state == 0:
                self.obj_render_set.remove(obj_id)
            else:
                self.obj_render_set.add(obj_id)
            # print(self.obj_render_set)
            self.update_frame()

        return model_table_item_check

    def construct_table_row_check_body_mesh(self, body_mesh_name):
        @QtCore.Slot()
        def model_table_item_check(state):
            if state == 0:
                self.body_mesh_draw_set.remove(body_mesh_name)
            else:
                self.body_mesh_draw_set.add(body_mesh_name)
            self.update_frame()

        return model_table_item_check

    def update_stream_data(self):
        self.curr_frame_data = self.stream_data[self.curr_frame_offset]
        self.curr_frame_id = self.curr_frame_data.frame_id

        # load res
        _opt_val = self.curr_frame_data["smplx_result"]
        for k, v in _opt_val.items():
            _opt_val[k] = v.to(device=self.device, dtype=self.dtype)
        _opt_model = self.smplx_layer(**_opt_val)
        _v = _opt_model.vertices.detach().clone().cpu().numpy().squeeze(0)
        _v = _v[self.smplx_body_upper_idx]
        self.curr_frame_body_vertices = _v

        self.widget_frame_ind.setText(str(self.curr_frame_id))
        self.widget_slider.setValue(self.curr_frame_offset)

        self.update_frame()
        self.update_text_desc()

    def render_frame(self):
        obj_opti_pose_map = self.curr_frame_data.optitrack_obj_transf

        img_list = []
        for cam_desc in CAMERA_LAYOUT_DESC:
            frame_color = self.curr_frame_data[f"color_{cam_desc}"].copy()
            cam_extr = self.curr_frame_data[f"cam_extr_{cam_desc}"]

            obj_pose_map = {}
            for obj_raw_id in self.obj_map:
                if obj_raw_id not in self.obj_render_set:
                    obj_pose_map[obj_raw_id] = None
                    continue

                obj_pose_map[obj_raw_id] = cam_extr @ obj_opti_pose_map[obj_raw_id]

            if "SMPLX" in self.body_mesh_draw_set and self.curr_frame_body_vertices is not None:
                body_vertices_cam = transf_point_array_np(cam_extr, self.curr_frame_body_vertices)
                body_mesh = trimesh.Trimesh(vertices=body_vertices_cam, faces=self.smplx_faces_np, process=False)
                extra_mesh = [body_mesh]
            else:
                extra_mesh = None

            img = self.renderer_map[cam_desc](
                obj_pose_map=obj_pose_map,
                background=frame_color[:, :, (2, 1, 0)],
                stick=True,
                blend=0.6,
                extra_mesh=extra_mesh,
            )
            img = img[:, :, (2, 1, 0)]  # cv2 convention
            img_list.append(img.copy())

        self.curr_render_img = img_list

    def resizeEvent(self, event):
        if self.curr_frame_img is not None:
            image = self.curr_frame_img
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            w = self.widget_frame_show.width()
            h = self.widget_frame_show.height()
            self.widget_frame_show.setPixmap(pixmap.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def widget_splitter_moved(self, event):
        if self.curr_frame_img is not None:
            image = self.curr_frame_img
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            w = self.widget_frame_show.width()
            h = self.widget_frame_show.height()
            self.widget_frame_show.setPixmap(pixmap.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def draw(self):
        curr_frame_img = []
        for _offset, cam_desc in enumerate(CAMERA_LAYOUT_DESC):
            render_img = self.curr_render_img[_offset].copy()
            curr_frame_img.append(render_img)

        self.curr_frame_img = combine_view(curr_frame_img)[:, :, (2, 1, 0)]
        self.curr_frame_img = self.curr_frame_img.copy()

    def update_frame(self, redraw=True):
        if redraw:
            self.render_frame()

        self.draw()

        # show image
        image = self.curr_frame_img
        qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        w = self.widget_frame_show.width()
        h = self.widget_frame_show.height()
        self.widget_frame_show.setPixmap(pixmap.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def update_text_desc(self):
        lh_cand = None
        rh_cand = None
        for seg_pair_def in self.program_info:
            lh_range, rh_range = seg_pair_def
            if lh_range is not None and lh_range[0] <= self.curr_frame_id < lh_range[1]:
                assert lh_cand is None
                lh_cand = seg_pair_def
            if rh_range is not None and rh_range[0] <= self.curr_frame_id < rh_range[1]:
                assert rh_cand is None
                rh_cand = seg_pair_def

        if lh_cand is None and rh_cand is None:
            cand = None
            cand_alt = None
        elif lh_cand is not None and rh_cand is None:
            cand = lh_cand
            cand_alt = None
        elif lh_cand is None and rh_cand is not None:
            cand = rh_cand
            cand_alt = None
        else:
            if lh_cand == rh_cand:
                cand = lh_cand
                cand_alt = None
            else:
                lh_beg = seg_pair_def_to_seg_def(lh_cand)[0]
                rh_beg = seg_pair_def_to_seg_def(rh_cand)[0]
                if lh_beg <= rh_beg:
                    cand = lh_cand
                    cand_alt = rh_cand
                else:
                    cand = rh_cand
                    cand_alt = lh_cand

        if cand is not None:
            seg_program = self.program_info[cand]
            seg_desc = self.desc_info[cand]
            self.widget_label_primitive_id.setText(f"primitive id: {seg_program['primitive']}")
            self.widget_label_primitive_text.setText(f"primtive text: {seg_desc['seg_desc']}")
            if seg_program["primitive_lh"] is not None:
                self.widget_label_primitive_id_lh.setText(f"primitive id (lh): {seg_program['primitive_lh']}")
            if seg_program["primitive_rh"] is not None:
                self.widget_label_primitive_id_rh.setText(f"primitive id (rh): {seg_program['primitive_rh']}")
            if cand[0] is not None:
                self.widget_label_primitive_range_lh.setText(f"primitive range (lh): {cand[0]}")
            else:
                self.widget_label_primitive_range_lh.setText("")
            if cand[1] is not None:
                self.widget_label_primitive_range_rh.setText(f"primitive range (rh): {cand[1]}")
            else:
                self.widget_label_primitive_range_rh.setText("")
        else:
            self.widget_label_primitive_id.setText("primitive id: ")
            self.widget_label_primitive_text.setText("primtive text: ")
            self.widget_label_primitive_id_lh.setText("")
            self.widget_label_primitive_id_rh.setText("")
            self.widget_label_primitive_range_lh.setText("")
            self.widget_label_primitive_range_rh.setText("")

        if cand_alt is not None:
            seg_program = self.program_info[cand_alt]
            seg_desc = self.desc_info[cand_alt]
            self.widget_label_primitive_id_alt.setText(f"primitive id: {seg_program['primitive']}")
            self.widget_label_primitive_text_alt.setText(f"primtive text: {seg_desc['seg_desc']}")
            if seg_program["primitive_lh"] is not None:
                self.widget_label_primitive_id_alt_lh.setText(f"primitive id (lh): {seg_program['primitive_lh']}")
            if seg_program["primitive_rh"] is not None:
                self.widget_label_primitive_id_alt_rh.setText(f"primitive id (rh): {seg_program['primitive_rh']}")
            if cand_alt[0] is not None:
                self.widget_label_primitive_range_alt_lh.setText(f"primitive range (lh): {cand_alt[0]}")
            else:
                self.widget_label_primitive_range_alt_lh.setText("")
            if cand_alt[1] is not None:
                self.widget_label_primitive_range_alt_rh.setText(f"primitive range (rh): {cand_alt[1]}")
            else:
                self.widget_label_primitive_range_alt_rh.setText("")
        else:
            self.widget_label_primitive_id_alt.setText("")
            self.widget_label_primitive_text_alt.setText("")
            self.widget_label_primitive_id_alt_lh.setText("")
            self.widget_label_primitive_id_alt_rh.setText("")
            self.widget_label_primitive_range_alt_lh.setText("")
            self.widget_label_primitive_range_alt_rh.setText("")
