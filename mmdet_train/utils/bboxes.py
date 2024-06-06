# -*- coding: utf-8 -*-
"""
Time ： 2024/2/19 10:43
Auth ： xiazhichao
File ：boxes.py
IDE ：PyCharm
Description:
"""

import numpy as np
import cv2
from pycocotools import mask as mask_util
from sympy import N
import torch
from typing import Optional, List, Union
from .typing_u import ListListT, OptListT


DataType = Union[torch.Tensor, np.ndarray]


def poly2mask(points: ListListT, width: int, height: int, fill_value: int=1):
    # if isinstance(mask, type(None)):
    mask = np.zeros((height, width), dtype=np.uint8)
    obj = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, obj, 1)
    mask = mask.astype(np.int32)
    mask[mask == 1] = fill_value
    return mask


def poly2mask_huggingface(points: ListListT, width: int, height: int, label: OptListT=None,
                          instance_index: int=None, vaild_index: Optional[Union[int, List]]=None):
    """
    label: start with 1.
    """
    if vaild_index:
        vaild_index = [vaild_index] if isinstance(vaild_index, int) else vaild_index
    else:
        vaild_index = range(len(points))

    if instance_index:
        instance_index = 1
        mask = np.zeros((height, width, 3), dtype=np.int32)
    else:
        mask = np.zeros((height, width), dtype=np.int32)

    for index in vaild_index:
        if instance_index:
            label_img = poly2mask(np.array(points[index], dtype=np.int32).reshape(-1, 2), width, height, fill_value=label[index])
            mask[:,:,0] = np.where(mask[:,:,0] !=0, mask[:,:,0], label_img)
            
            instance_img = poly2mask(np.array(points[index], dtype=np.int32).reshape(-1, 2),
                                    width, height, fill_value=instance_index)
            mask[:,:,1] = np.where(mask[:,:,1] !=0, mask[:,:,1], instance_img)
            instance_index += 1
        else:
            label_img = poly2mask(np.array(points[index], dtype=np.int32).reshape(-1, 2), width, height, fill_value=label[index])
            mask = np.where(mask !=0, mask, label_img)
    return mask



def single_mask2rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def xyn2xy(x: DataType, w: int=640, h: int=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y


def xywh2xyxy(x: DataType, w=640, h=640, padw=0, padh=0, norm=False):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    if not norm:
        w, h = 1, 1
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def x1y1wh2xyxy(x: DataType, w=640, h=640, padw=0, padh=0, norm=False):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    if not norm:
        w, h = 1, 1

    y[..., 0] = w * (x[..., 0]) + padw  # bottom right x
    y[..., 1] = h * (x[..., 1]) + padh  # bottom right y
    y[..., 2] = w * (x[..., 0] + x[..., 2]) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3]) + padh  # bottom right y
    return y

def clip_boxes(boxes: DataType, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def xyxy2xywh(x, w=640, h=640, clip=False, eps=0.0, norm=False):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    if not norm:
        w, h = 1, 1
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y

def to_boxes(point_array:np.ndarray):
    """

    :param point_array: [[1,2,[3,4]]
    :return: [x1,y1,x2,y2]
    """
    x1, y1 = np.min(point_array, axis=0)
    x2, y2 = np.max(point_array, axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def yolo_letter_box(img, new_shape=(640, 640), scaleup=False, center=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if center:
        dw /= 2  # divide padding into 2 sides
        dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    return img, r, left, top