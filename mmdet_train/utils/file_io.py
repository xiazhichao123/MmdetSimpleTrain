# -*- coding: utf-8 -*-
"""
Time ： 2024/2/5 17:11
Auth ： xiazhichao
File ：file_io.py
IDE ：PyCharm
Description:
    
"""

import yaml, re
from pathlib import Path
import json
import urllib.request
import cv2
import numpy as np

def yaml_read(file, append_filename=False,  replace=True, **kwargs):
    with open(file, 'r', encoding='utf-8') as f:
        s = f.read()  # string

        if kwargs.get('replace', True):
            if s.startswith('&'):
                s = '\n' + s
        
            matches = re.findall(r'(\n&__.*?: .*?\n)', s)
            if matches:
                for match_i in matches:
                    s = re.sub(match_i, '\n', s)
                    match_ii = match_i.strip().split(':')
                    _key = r'{{' + match_ii[0].strip()[1:] + '}}'
                    s = re.sub(_key, match_ii[1].strip(), s)

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data['yaml_file'] = str(file)
        return data


def yaml_save(file='data.yaml', data={}):
    # Single-line safe yaml saving
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def json_read(file):
    with open(file, 'r', encoding='utf-8') as f:
        content = json.load(f)
    return content

def json_save(file, data):
    with open(file, 'w') as fw:
        json.dump(data, fw)

def read_file(file, **kwargs):
    '''
        kwargs:
            type: yaml
    :param file:
    :return:
    '''
    if not Path(file).exists():
        return None
    suffix = Path(file).suffix.split('.')[-1]
    try:
        return eval(f'{suffix}_read')(str(file, **kwargs))
    except NameError:
        raise NameError(f'donot support file format:{suffix}')


def save_file(file, data, **kwargs):
    """

    :param file:
    :param data:
    :param kwargs:
                type: yaml
    :return:
    """
    suffix = Path(file).suffix.split('.')[-1]
    try:
        type1 = kwargs.get('type')
        if type1:
            eval(f'{type1}_save')(str(file), data)
        else:
            eval(f'{suffix}_save')(str(file), data)
    except NameError:
        raise NameError((f'donot support file format:{suffix or kwargs.get("type")}'))


def huggingface_download_file(repo_id="bert-base-chinese", cache_dir=r'',resume_download=True):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, cache_dir=cache_dir,resume_download=resume_download)

def url_img_to_cv(img_url):
    with urllib.request.urlopen(img_url) as url_response:
        s = url_response.read()

    arr = np.asarray(bytearray(s), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img

if __name__ == '__main__':
    huggingface_download_file("THUDM/chatglm-6b", cache_dir=r'H:\xc_coding\github\ChatGLM-6B\weights')
