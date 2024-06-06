# -*- coding: utf-8 -*-
"""
Time ： 2024/6/6 14:25
Auth ： xiazhichao
File ：trainner.py
IDE ：PyCharm
Description:
"""

from pathlib import Path
import sys, os

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from mmdet_train.utils.file_io import read_file
from mmdet_train.utils.check import check_add_prefiex_path
from mmdet_train.utils import logger
from mmdet_train.utils.reload_package import Reload

root_dir = Path(__file__).parent
config_file = root_dir / 'config.yaml'
config = read_file(config_file)


config['log_format']['log_filename'] = str(root_dir / f'project/{config["project_name"]}/log.log')

logger_instant = logger.init_log(root_dir, **config['log_format'])
logger.logger = logger_instant.get_log('main_train')
Reload('mmdet_train', 'mmdet_train.utils.logger', logger)
logger = logger.logger
logger.info(f'{__file__}. config path: {config_file}')

def main():
    project_name = config['project_name']
    # read config file
    project_config = read_file(str(root_dir / f'project/{project_name}/config.yaml'))
    datasets = project_config.pop('datasets')
    for k, v in datasets.items():
        if isinstance(v, (list, tuple)):
            for i in range(len(v)):
                datasets[k][i] = check_add_prefiex_path(v[i], project_config['root_dir'], force=True)
                continue
        datasets[k] = check_add_prefiex_path(v, project_config['root_dir'], force=True)
    datasets['project_name'] = project_name
    project_config['datasets'] = datasets

    cache_path = check_add_prefiex_path(project_config['model_args'].get('torch_cache', ''), project_config['root_dir'], force=True)
    os.environ['TORCH_HOME'] = cache_path   # set torch cache before load torch

    from mmdet_train.img_det import Img_det

    model = Img_det(**project_config)
    model.do()


if __name__ == '__main__':
    main()