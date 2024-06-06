# -*- coding: utf-8 -*-
"""
Time ： 2024/2/21 8:27
Auth ： xiazhichao
File ：reload.py
IDE ：PyCharm
Description:
"""

from .utils.logger import logger
from .utils.path import finde_site_packages

import os, sys
from pathlib import Path


from mmengine.config import config
from mmengine.config.config import Config

_get_cfg_path_copy = Config._get_cfg_path
def _get_cfg_path(cfg_path: str, filename: str=''):
    if not cfg_path.startswith('mmdet'):
        logger.info(f'use default config file: mmengine.config.config._get_cfg_path')
        return _get_cfg_path_copy(cfg_path, filename)
    model_type, cfg_path = cfg_path.split('.', maxsplit=1)
    mmdet_path, flag = finde_site_packages(model_type,
                                           parent_dir=str(Path(__file__).parent.parent.parent.parent.parent.parent.parent / 'github/mmdetection'))   # flag: True, site-package; False, dir
    if not flag:
        cfg_path = os.path.abspath(os.path.join(str(Path(mmdet_path).parent / 'configs'), cfg_path))
    else:
        cfg_path = os.path.abspath(os.path.join(mmdet_path, '.mim', 'configs', cfg_path))

    logger.warning(f'use custom config file {cfg_path}')
    logger.info(filename)
    return cfg_path, None
config.Config._get_cfg_path = _get_cfg_path


def dump(self, file):
    """Dump config to file or return config text.

    Args:
        file (str or Path, optional): If not specified, then the object
        is dumped to a str, otherwise to a file specified by the filename.
        Defaults to None.

    Returns:
        str or None: Config text.
    """
    if Path(self.filename).name == Path(file).name:    # save faster-rcnn_r50_fpn_1x_coco,py
        file = str(Path(self.filename).parent / f'{Path(self.filename).stem}_new.py')
    file = str(file) if isinstance(file, Path) else file
    cfg_dict = self.to_dict()
    if file is None:
        if self.filename is None or self.filename.endswith('.py'):
            return self.pretty_text
        else:
            file_format = self.filename.split('.')[-1]
            return dump(cfg_dict, file_format=file_format)
    elif file.endswith('.py'):
        with open(file, 'w', encoding='utf-8') as f:
            f.write(self.pretty_text)
    else:
        file_format = file.split('.')[-1]
        return dump(cfg_dict, file=file, file_format=file_format)

config.Config.dump = dump