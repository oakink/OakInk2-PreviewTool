import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
from typing import Sequence, Tuple


DEFAULT_FONT = "FreeMonoBold.ttf"
ZH_FONT = "wqy-microhei.ttc"


def caption_combined_view(combine_image, caption="This is a caption", font=DEFAULT_FONT):
    ncol = combine_image.shape[1]
    canvas = np.ones((30, ncol, 3), dtype=np.uint8) * 255
    font = ImageFont.truetype(font, size=20)
    canvas_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_pil)
    draw.text((20, 5), caption, font=font, fill=(0, 0, 0))
    canvas = np.array(canvas_pil)
    res = np.concatenate([canvas, combine_image], axis=0)
    return res


def combine_view(view_list, ncol=None):
    if ncol is None:
        ncol = int(math.sqrt(len(view_list)))

    img_shape = view_list[0].shape
    img_width = img_shape[1]

    res = []
    for row_offset in range(0, len(view_list), ncol):
        row_img = view_list[row_offset : row_offset + ncol]
        row_img = np.concatenate(row_img, axis=1)
        res.append(row_img)

    res = np.concatenate(res, axis=0)
    return res


def _out_of_frame(pos, shape):
    h, w = shape
    x, y = pos
    if x < 0 or x >= w or y < 0 or y >= h:
        return True
    return False


def draw_wireframe(
    img,
    vert_list,
    edge_list,
    vert_color,
    edge_color,
    vert_size=3,
    edge_size=1,
    vert_type=None,
    vert_thickness=1,
    vert_mask=None,
):
    img_h, img_w = img.shape[:2]

    vert_list = np.asarray(vert_list)
    n_vert = len(vert_list)
    n_edge = len(edge_list)
    vert_color = np.asarray(vert_color)
    edge_color = np.asarray(edge_color)

    # expand edge color
    if edge_color.ndim == 1:
        edge_color = np.tile(edge_color, (n_edge, 1))

    # expand edge size
    if isinstance(edge_size, (int, float)):
        edge_size = [edge_size] * n_edge

    # # expand vert color
    if vert_color.ndim == 1:
        vert_color = np.tile(vert_color, (n_vert, 1))

    # expand vert size
    if isinstance(vert_size, (int, float)):
        vert_size = [vert_size] * n_vert

    # set default vert type
    if vert_type is None:
        vert_type = ["circle"] * n_vert

    # expand vert thickness
    if isinstance(vert_thickness, (int, float)):
        vert_thickness = [vert_thickness] * n_vert

    # draw edge
    for edge_id, connection in enumerate(edge_list):
        if vert_mask is not None:
            if not vert_mask[int(connection[1])] or not vert_mask[int(connection[0])]:
                continue
        coord1 = vert_list[int(connection[1])]
        coord2 = vert_list[int(connection[0])]
        if _out_of_frame(coord1, (img_h, img_w)) and _out_of_frame(coord2, (img_h, img_w)):
            continue

        cv2.line(
            img,
            coord1.astype(np.int32),
            coord2.astype(np.int32),
            color=edge_color[edge_id] * 255,
            thickness=edge_size[edge_id],
        )

    for vert_id in range(vert_list.shape[0]):
        if vert_mask is not None:
            if not vert_mask[vert_id]:
                continue
        if _out_of_frame(vert_list[vert_id], (img_h, img_w)):
            continue

        draw_type = vert_type[vert_id]
        # if vert_id in [1, 5, 9, 13, 17]:  # mcp joint
        if draw_type == "circle":
            cv2.circle(
                img,
                (int(vert_list[vert_id, 0]), int(vert_list[vert_id, 1])),
                radius=vert_size[vert_id],
                color=vert_color[vert_id] * 255,
                thickness=cv2.FILLED,
            )
        # elif vert_id in [2, 6, 10, 14, 18]:  # proximal joints
        elif draw_type == "square":
            cv2.drawMarker(
                img,
                (int(vert_list[vert_id, 0]), int(vert_list[vert_id, 1])),
                color=vert_color[vert_id] * 255,
                markerType=cv2.MARKER_SQUARE,
                markerSize=vert_size[vert_id] * 2,
                thickness=vert_thickness[vert_id],
            )
        # elif vert_id in [3, 7, 11, 15, 19]:  # distal joints:
        elif draw_type == "triangle_up":
            cv2.drawMarker(
                img,
                (int(vert_list[vert_id, 0]), int(vert_list[vert_id, 1])),
                color=vert_color[vert_id] * 255,
                markerType=cv2.MARKER_TRIANGLE_UP,
                markerSize=vert_size[vert_id] * 2,
                thickness=vert_thickness[vert_id],
            )
        # elif vert_id in [4, 8, 12, 16, 20]:
        elif draw_type == "diamond":
            cv2.drawMarker(
                img,
                (int(vert_list[vert_id, 0]), int(vert_list[vert_id, 1])),
                color=vert_color[vert_id] * 255,
                markerType=cv2.MARKER_DIAMOND,
                markerSize=vert_size[vert_id] * 2,
                thickness=vert_thickness[vert_id],
            )
        elif draw_type == "star":
            cv2.drawMarker(
                img,
                (int(vert_list[vert_id, 0]), int(vert_list[vert_id, 1])),
                color=vert_color[vert_id] * 255,
                markerType=cv2.MARKER_STAR,
                markerSize=vert_size[vert_id] * 2,
                thickness=vert_thickness[vert_id],
            )
        else:
            # fallback
            cv2.circle(
                img,
                (int(vert_list[vert_id, 0]), int(vert_list[vert_id, 1])),
                radius=vert_size[vert_id],
                color=vert_color[vert_id] * 255,
                thickness=cv2.FILLED,
            )


edge_list_hand = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
vert_color_hand = np.array(
    [
        [1.0, 0.0, 0.0],
        #
        [0.0, 0.4, 0.2],
        [0.0, 0.6, 0.3],
        [0.0, 0.8, 0.4],
        [0.0, 1.0, 0.5],
        #
        [0.0, 0.0, 0.4],
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 0.8],
        [0.0, 0.0, 1.0],
        #
        [0.0, 0.4, 0.4],
        [0.0, 0.6, 0.6],
        [0.0, 0.8, 0.8],
        [0.0, 1.0, 1.0],
        #
        [0.4, 0.4, 0.0],
        [0.6, 0.6, 0.0],
        [0.8, 0.8, 0.0],
        [1.0, 1.0, 0.0],
        #
        [0.4, 0.0, 0.4],
        [0.6, 0.0, 0.6],
        [0.7, 0.0, 0.8],
        [1.0, 0.0, 1.0],
    ]
)
vert_color_hand = vert_color_hand[:, ::-1]
edge_color_hand = np.array(
    [
        vert_color_hand[1, :],
        vert_color_hand[2, :],
        vert_color_hand[3, :],
        vert_color_hand[4, :],
        vert_color_hand[5, :],
        vert_color_hand[6, :],
        vert_color_hand[7, :],
        vert_color_hand[8, :],
        vert_color_hand[9, :],
        vert_color_hand[10, :],
        vert_color_hand[11, :],
        vert_color_hand[12, :],
        vert_color_hand[13, :],
        vert_color_hand[14, :],
        vert_color_hand[15, :],
        vert_color_hand[16, :],
        vert_color_hand[17, :],
        vert_color_hand[18, :],
        vert_color_hand[19, :],
        vert_color_hand[20, :],
    ]
)
vert_type_hand = [
    "star",
    "circle",
    "square",
    "triangle_up",
    "diamond",
    "circle",
    "square",
    "triangle_up",
    "diamond",
    "circle",
    "square",
    "triangle_up",
    "diamond",
    "circle",
    "square",
    "triangle_up",
    "diamond",
    "circle",
    "square",
    "triangle_up",
    "diamond",
]


def draw_wireframe_hand(img, hand_joint_arr, hand_joint_mask):
    draw_wireframe(
        img,
        hand_joint_arr,
        edge_list=edge_list_hand,
        vert_color=vert_color_hand,
        edge_color=edge_color_hand,
        vert_type=vert_type_hand,
        vert_mask=hand_joint_mask,
    )


def draw_wireframe_hand_large(img, hand_joint_arr, hand_joint_mask):
    draw_wireframe(
        img,
        hand_joint_arr,
        edge_list=edge_list_hand,
        vert_color=vert_color_hand,
        edge_color=edge_color_hand,
        vert_type=vert_type_hand,
        vert_mask=hand_joint_mask,
        vert_size=8,
        edge_size=2,
        vert_thickness=3,
    )


edge_list_hand_kp = [
    (0, 2),
    (2, 4),
    (0, 5),
    (5, 8),
    (0, 9),
    (9, 12),
    (0, 13),
    (13, 16),
    (0, 17),
    (17, 20),
]


def draw_wireframe_hand_kp(img, hand_joint_arr, hand_joint_mask):
    draw_wireframe(
        img,
        hand_joint_arr,
        edge_list=edge_list_hand_kp,
        vert_color=vert_color_hand,
        edge_color=edge_color_hand,
        vert_type=vert_type_hand,
        vert_mask=hand_joint_mask,
    )


def draw_wireframe_hand_kp_large(img, hand_joint_arr, hand_joint_mask):
    draw_wireframe(
        img,
        hand_joint_arr,
        edge_list=edge_list_hand_kp,
        vert_color=vert_color_hand,
        edge_color=edge_color_hand,
        vert_type=vert_type_hand,
        vert_mask=hand_joint_mask,
        vert_size=6,
        edge_size=2,
        vert_thickness=3,
    )


def get_combined_image_offset(position, img_shape, len_img_list, ncol=None):
    if ncol is None:
        ncol = int(math.sqrt(len_img_list))
    nrow = len_img_list // ncol + 1

    img_width = int(img_shape[1])
    img_height = int(img_shape[0])

    pos_x, pos_y = position
    col_offset = pos_x // img_width
    row_offset = pos_y // img_height
    offset = ncol * row_offset + col_offset
    return int(offset)


def get_combined_image_pos(position, img_shape):
    img_width = int(img_shape[1])
    img_height = int(img_shape[0])

    pos_x, pos_y = position
    pos_x = pos_x % img_width
    pos_y = pos_y % img_height
    return (pos_x, pos_y)


def get_combined_image_pos_fix_offset(position, img_shape, offset, len_img_list, ncol=None):
    # compute offset base coord
    img_width = int(img_shape[1])
    img_height = int(img_shape[0])

    if ncol is None:
        ncol = int(math.sqrt(len_img_list))
    offset_col_id = offset % ncol
    offset_row_id = offset // ncol

    base_coord_x = offset_col_id * img_width
    base_coord_y = offset_row_id * img_height

    pos_x, pos_y = position
    pos_x = pos_x - base_coord_x
    pos_y = pos_y - base_coord_y
    return (pos_x, pos_y)


def decaption_pos(position):
    pos_x, pos_y = position
    pos_y -= 30
    return (pos_x, pos_y)


def offset_combined_image_pos(position_local, img_shape, offset, len_img_list, ncol=None):
    img_width = int(img_shape[1])
    img_height = int(img_shape[0])

    if ncol is None:
        ncol = int(math.sqrt(len_img_list))
    offset_col_id = offset % ncol
    offset_row_id = offset // ncol

    base_coord_x = offset_col_id * img_width
    base_coord_y = offset_row_id * img_height

    pos_x, pos_y = position_local
    pos_x = pos_x + base_coord_x
    pos_y = pos_y + base_coord_y
    return (pos_x, pos_y)


edge_list_markerset_body = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (3, 4),
    (3, 5),
    (4, 7),
    (5, 11),
    (6, 8),
    (6, 12),
    (7, 8),
    (7, 9),
    (8, 9),
    (9, 10),
    (11, 12),
    (11, 13),
    (12, 13),
    (13, 14),
    (15, 16),
    (15, 17),
    (16, 17),
    (16, 18),
    (17, 18),
]
vert_color_markerset_body = np.array(
    [
        [234.0 / 255.0, 128.0 / 255.0, 255.0 / 255.0],
        [234.0 / 255.0, 128.0 / 255.0, 255.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [234.0 / 255.0, 128.0 / 255.0, 255.0 / 255.0],
        [202.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [234.0 / 255.0, 128.0 / 255.0, 255.0 / 255.0],
        [255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [202.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [59.0 / 255.0, 102.0 / 255.0, 0.0 / 255.0],
        [202.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [59.0 / 255.0, 102.0 / 255.0, 0.0 / 255.0],
        [202.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [202.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
    ]
)
vert_color_markerset_body = vert_color_markerset_body[:, ::-1]
edge_color_markerset_body = np.array(
    [
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [127.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [127.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [127.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],
        [127.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],
        [127.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],
        [127.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],
    ]
)
edge_color_markerset_body = edge_color_markerset_body[:, ::-1]

edge_list_markerset_hand = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 8), (2, 9), (3, 6), (3, 7), (4, 5)]
vert_color_markerset_hand = np.array(
    [
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [75.0 / 255.0, 225.0 / 255.0, 255.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [75.0 / 255.0, 225.0 / 255.0, 255.0 / 255.0],
        [255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0],
        [255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0],
        [127.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [127.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
    ]
)
vert_color_markerset_hand = vert_color_markerset_hand[:, ::-1]
edge_color_markerset_hand = np.array(
    [
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [222.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [127.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [127.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
        [0.0 / 255.0, 235.0 / 255.0, 255.0 / 255.0],
    ]
)
edge_color_markerset_hand = edge_color_markerset_hand[:, ::-1]


def draw_wireframe_markerset_body(img, body_markerset_arr, body_markerset_mask):
    draw_wireframe(
        img,
        body_markerset_arr,
        edge_list=edge_list_markerset_body,
        vert_color=vert_color_markerset_body,
        edge_color=edge_color_markerset_body,
        # vert_type=vert_type_hand,
        vert_mask=body_markerset_mask,
    )


def draw_wireframe_markerset_hand(img, hand_markerset_arr, hand_markerset_mask):
    draw_wireframe(
        img,
        hand_markerset_arr,
        edge_list=edge_list_markerset_hand,
        vert_color=vert_color_markerset_hand,
        edge_color=edge_color_markerset_hand,
        vert_mask=hand_markerset_mask,
    )


edge_list_bbox = [
    [0, 1],
    [1, 3],
    [3, 2],
    [2, 0],
    #
    [4, 5],
    [5, 7],
    [7, 6],
    [6, 4],
    #
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]


def blend_mask(image, mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 144 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape((1, 1, -1))
    # alpha blend image and mark
    image = image.copy()
    image = image.astype(np.float32) / 255
    image = image * (1 - mask_image[:, :, 3:]) + mask_image[:, :, :3] * mask_image[:, :, 3:]
    image = (image * 255).astype(np.uint8)
    return image
