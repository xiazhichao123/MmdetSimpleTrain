# -*- coding: utf-8 -*-
"""
Time ： 2024/2/22 16:50
Auth ： xiazhichao
File ：check.py
IDE ：PyCharm
Description:
"""


import contextlib
from PIL import Image, ExifTags, ImageOps
from .logger import logger, Constant

import platform
from pathlib import Path

def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s

def verify_image(file:str):
    im = Image.open(file)
    im.verify()
    shape = exif_size(im)
    assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels.--> {file}'
    assert im.format.lower() in Constant.IMG_FORMATS, f'invalid image format {im.format}.-->{file}'
    if im.format.lower() in ('jpg', 'jpeg'):
        with open(file, 'rb') as f:
            f.seek(-2, 2)
            if f.read() != b'\xff\xd9':  # corrupt JPEG
                ImageOps.exif_transpose(Image.open(file)).save(file, 'JPEG', subsampling=0, quality=100)
                logger.warning(f'WARNING ⚠️ {file}: corrupt JPEG restored and saved')


def get_os_type():
    os_type = platform.system()
    return os_type


def check_add_prefiex_path(path, prefix, force=False):
    if not isinstance(path, (str, Path)):
        return path
    if not ('/' in path or '\\' in path):
        return path
    path = Path(path)

    if force or (not path.exists() and prefix):
        return (Path(prefix) /path.__str__()).__str__()
    return path.__str__()


def is_chinese(string1):
    for char in string1:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False