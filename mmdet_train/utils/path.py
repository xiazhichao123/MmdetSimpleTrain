# -*- coding: utf-8 -*-
"""
Time ： 2024/2/21 8:39
Auth ： xiazhichao
File ：path.py
IDE ：PyCharm
Description:
"""
import site, os
from pathlib import Path
import os, glob
from typing import Union, List
import sys
from pathlib import Path
from .logger import logger


def finde_site_packages(package_name: str, parent_dir=None) -> tuple[str, bool]:
    if parent_dir:
        result = os.path.join(parent_dir, package_name)
        if os.path.exists(result):
            logger.info(f'find {package_name} in {parent_dir}')
            return result, False

    for package_i in site.getsitepackages():
        if package_i.endswith('site-packages'):
            result = Path(package_i) / package_name
            assert result.exists(), f'cant find {package_name} in {package_i}'
            logger.info(f'find {package_name} in {package_i}')
            return str(result), True

    return '', False



def get_latest_run(folder, file_patten):
    """
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    """
    last_list = glob.glob(f"{folder}/**/{file_patten}", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


def scandir(folder:str, file_patten: Union[List, str], recursive=True, mid_patten=None, case_sensitive=None):
    if isinstance(file_patten, str):
        file_patten = [file_patten]
        
    for patten_i in file_patten:
        if mid_patten:
            yield from glob.iglob(f"{folder}/**/{mid_patten}/**/{patten_i}", recursive=recursive)
        else:
            yield from glob.iglob(f"{folder}/**/{patten_i}", recursive=recursive)


def get_path_sys_path(path1):
    for path_i in sys.path:
        if Path(path_i).exists() and path1 == Path(path_i).stem:
            return path_i