# -*- coding: utf-8 -*-
"""
Time ： 2024/2/19 9:35
Auth ： xiazhichao
File ：format_trans.py
IDE ：PyCharm
Description:
"""
import json
from types import SimpleNamespace
import base64
import numpy as np
import cv2


def dict_args(**_dict):
    """key is not digital"""
    return SimpleNamespace(**_dict)

def args_dict(args):
    return vars(args)


def file_base64_str(image_path):
    """file to base64 string"""
    with open(image_path, 'rb') as f:
        image = f.read()
        image_base64 = str(base64.b64encode(image), encoding='utf-8')
    return image_base64


def base64_str_cv(base64_str):
    """base64 string to cv"""
    base64_byte = bytes(base64_str, encoding='utf-8')
    img_string = base64.b64decode(base64_byte)
    np_arr = np.fromstring(img_string, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    return image

def cv2_to_base64(img):
    img = cv2.imencode('.jpg', img)[1]
    image_code = str(base64.b64encode(img))[2:-1]
    return image_code


def to_numpy(inputs):
    import torch
    if isinstance(inputs, torch.Tensor):
        return inputs.cpu().detach().numpy()

    if isinstance(inputs, list):
        return np.stack(inputs, axis=0)

    assert isinstance(inputs, np.ndarray), 'input must be np.ndarray.'
    return inputs
