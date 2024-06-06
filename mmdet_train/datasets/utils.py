# -*- coding: utf-8 -*-
"""
Time ： 2024/6/5 8:47
Auth ： xiazhichao
File ：utils.py
IDE ：PyCharm
Description:
"""

from ..utils.annotation.coco2coco import COCOapi

class MMdet_coco(COCOapi):
    def __init__(self, coco_json=None):
        super(MMdet_coco, self).__init__(coco_json)

    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids):
        return self.loadAnns(ids)

    def load_cats(self, ids):
        return self.loadCats(ids)

    def load_imgs(self, ids):
        return self.loadImgs(ids)