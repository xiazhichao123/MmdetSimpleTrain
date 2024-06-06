# -*- coding: utf-8 -*-
"""
Time ： 2024/6/5 8:36
Auth ： xiazhichao
File ：coco_metric.py
IDE ：PyCharm
Description:
"""

from mmdet.evaluation.metrics.coco_metric import CocoMetric, METRICS
from .utils import MMdet_coco
from ..utils.annotation.coco2coco import Coco2coco
from ..utils.logger import logger


@METRICS.register_module()
class MyCOCOMetric(CocoMetric):
    def __init__(self, **cfg):
        self.use_dataset = cfg.pop('use_dataset')
        cfg['ann_file'] = None
        self.flag = cfg.pop('flag', 'val')  # train or val or test
        super(MyCOCOMetric, self).__init__(**cfg)

        logger.info(f'load {self.flag} metric datasets')
        if self.flag == 'val':
            coco = Coco2coco(self.use_dataset['val_datasets'], yaml_file=self.use_dataset['meta_data'])
        self._coco_api = MMdet_coco().from_json_data(coco())
        if cfg.get('sort_categories'):
            # 'categories' list in objects365_train.json and
            # objects365_val.json is inconsistent, need sort
            # list(or dict) before get cat_ids.
            cats = self._coco_api.cats
            sorted_cats = {i: cats[i] for i in sorted(cats)}
            self._coco_api.cats = sorted_cats
            categories = self._coco_api.dataset['categories']
            sorted_categories = sorted(
                categories, key=lambda i: i['id'])
            self._coco_api.dataset['categories'] = sorted_categories