# -*- coding: utf-8 -*-
"""
Time ： 2024/6/3 15:16
Auth ： xiazhichao
File ：coco.py
IDE ：PyCharm
Description:
"""

from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset
from ..utils.typing_u import ListDict

from ..utils.annotation.coco2coco import Coco2coco
from .utils import MMdet_coco
from ..utils.file_io import read_file
from ..utils.visual import _ncolors
import copy
import os.path as osp
from ..utils.logger import logger




@DATASETS.register_module(name='MyCOCO')
class MyCOCO(CocoDataset):
    ANN_ID_UNIQUE = True

    def __init__(self, **cfg):
        self.use_dataset = cfg.pop('use_dataset')
        self.flag = cfg.pop('flag')    # train or val or test
        metainfo = self._matainfo()
        super(MyCOCO, self).__init__(metainfo=metainfo, **cfg)

    def _matainfo(self):
        id2labels = read_file(self.use_dataset['meta_data'])['names']
        label_list = list(zip(*sorted(id2labels.items(), key=lambda x: x[0])))[1]
        return {
            'classes': tuple(label_list),
            'palette': _ncolors(len(label_list)),
        }

    def load_data_list(self) -> ListDict:
        logger.info(f'load {self.flag} datasets......')
        if self.flag == 'train':
            coco = Coco2coco(self.use_dataset['train_datasets'], yaml_file=self.use_dataset['meta_data'])
        elif self.flag == 'val':
            coco = Coco2coco(self.use_dataset['val_datasets'], yaml_file=self.use_dataset['meta_data'])

        coco = MMdet_coco().from_json_data(coco())
        self.img_ann_map = coco.imgToAnns
        self.cat_img_map = coco.catToImgs

        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = coco.getCatIds(catNms=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(coco.catToImgs)

        img_ids = coco.getImgIds()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = coco.loadImgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = coco.getAnnIds(imgIds=[img_id])
            raw_ann_info = coco.loadAnns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                    raw_ann_info,
                'raw_img_info':
                    raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        return data_list


    def parse_data_info(self, raw_data_info: dict):
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        data_info['img_path'] = img_info['file_name']
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = None
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info