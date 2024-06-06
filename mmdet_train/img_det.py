# -*- coding: utf-8 -*-
"""
Time ： 2024/6/3 14:27
Auth ： xiazhichao
File ：img_det.py
IDE ：PyCharm
Description:
"""

from pathlib import Path
alg_type = Path(__file__).stem

from . import reload
from .utils.check import check_add_prefiex_path


from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.runner import Runner
from mmengine.registry import RUNNERS
from mmengine.config import Config
import os.path as osp
import os
from .utils.format_trans import dict_args
from . import datasets
from copy import deepcopy
from .utils.file_io import read_file
from .utils.dict_child_update import find_all_child_keys, auto_update_dict_value
from collections import OrderedDict


class Img_det():
    def __init__(self, **kwargs):
        use_datasets = kwargs.pop('datasets')
        # self.train_package_path = kwargs['model_args'].pop('train_package_path', '')
        root_dir = kwargs.pop('root_dir')

        self.conf = kwargs['model_args'].pop('conf',0)
        args = kwargs.pop('model_args')

        for k, v in use_datasets.items():
            new_v = check_add_prefiex_path(v, root_dir, force=True)
            if new_v:
                use_datasets[k] = new_v

        for k, v in args.items():
            new_v = check_add_prefiex_path(v, root_dir, force=True)
            if new_v:
                args[k] = new_v

        load_from = args.pop('load_from', None)
        args = dict_args(**args)
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
        # Reduce the number of repeated compilations and improve
        # training speed.
        setup_cache_size_limit_of_dynamo()

        # load config
        cfg = Config.fromfile(args.config)
        cfg.launcher = args.launcher


        cfg.train_dataloader.dataset.use_dataset = use_datasets
        cfg.train_dataloader.dataset.type = 'MyCOCO'
        cfg.train_dataloader.dataset.flag = 'train'

        cfg.val_dataloader.dataset.use_dataset = deepcopy(use_datasets)
        cfg.val_dataloader.dataset.type = 'MyCOCO'
        cfg.val_dataloader.dataset.flag = 'val'
        cfg.val_evaluator.use_dataset = deepcopy(use_datasets)
        cfg.val_evaluator.type = 'MyCOCOMetric'
        cfg.val_evaluator.flag = 'val'

        # replace defalut num_classes
        child_keys = OrderedDict()
        find_all_child_keys(child_keys, cfg['model'], force=False)
        auto_update_dict_value(child_keys, cfg['model'], {'num_classes': len(read_file(use_datasets['meta_data'])['names']),
                                                          'init_cfg': None})

        if args.cfg_options:
            cfg.merge_from_dict(args.cfg_options)

        # work_dir is determined in this priority: CLI > segment in file > filename
        if args.work_dir:
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = args.work_dir
        elif cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])


        # enable automatic-mixed-precision training
        if args.amp is True:
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

        # enable automatically scaling LR
        if args.auto_scale_lr:
            if 'auto_scale_lr' in cfg and \
                    'enable' in cfg.auto_scale_lr and \
                    'base_batch_size' in cfg.auto_scale_lr:
                cfg.auto_scale_lr.enable = True
            else:
                raise RuntimeError('Can not find "auto_scale_lr" or '
                                   '"auto_scale_lr.enable" or '
                                   '"auto_scale_lr.base_batch_size" in your'
                                   ' configuration file.')

        # resume is determined in this priority: resume from > auto_resume
        cfg.load_from = load_from
        if args.resume == 'auto':
            cfg.resume = True
            cfg.load_from = None
        elif args.resume:
            cfg.resume = True
            cfg.load_from = args.resume

        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            self.runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            self.runner = RUNNERS.build(cfg)


    def do(self):
        # start training
        self.runner.train()
