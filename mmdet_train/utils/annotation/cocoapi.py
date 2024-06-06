# -*- coding: utf-8 -*-
"""
Time ： 2023/7/3 15:11
Auth ： xiazhichao
File ：cocoapi.py
IDE ：PyCharm
Description:
    
"""

from pycocotools import coco
from pathlib import Path
from typing import List, Union, Optional
import random
from copy import deepcopy




class COCOapi(coco.COCO):
    def __init__(self, coco_json=None):
        if coco_json:
            super(COCOapi, self).__init__(coco_json)
            self.root = Path(coco_json).parent

    def from_json_data(self, json_data, is_copy=False):
        copy_self = deepcopy(self) if is_copy else self
        copy_self.dataset = json_data
        copy_self.createIndex()
        return copy_self
            

    @property
    def json_data(self):
        return self.dataset

    def get_img_allid(self):
        """return images all id"""
        return list(self.imgs.keys())

    def get_img_info(self, id:Union[int, list]) -> list:
        """:return list
        [{'file_name':'images/train2017/0001.jpg',
           'height': 500,
           'width': 473,
           'id': 107   # imgid
        """
        return self.loadImgs(id)

    def filter_imgid(self, imgids:Union[int, list]=None, category_id: Union[int, list]=None):
        ''''''
        return self.getImgIds(imgids, category_id)


    def get_id_by_class_name(self, classes_name:[str, list]=None):
        """input label name,return index."""
        return self.getCatIds(catIds=classes_name)

    def get_class_name_by_id(self, category_id:[int, list]) -> list:
        """[person]"""
        category_info = self.loadCats(category_id)
        class_name = [name_i['name'] for name_i in category_info]
        return class_name

    def get_all_class_names(self):
        categories = self.loadCats(self.getCatIds())
        class_names = [category['name'] for category in categories]
        return class_names

    def get_all_class_ids(self):
        return self.getCatIds()


    def get_class_id_name(self):
        ''':return dict'''
        return dict(zip(self.get_all_class_ids(), self.get_all_class_names()))

    def get_class_name_id(self):
        ''':return dict'''
        return dict(zip(self.get_all_class_names(), self.get_all_class_ids()))


    def get_gt_id_by_img_id(self, id:[int, list]):
        """return per images contain gt id"""
        return self.getAnnIds(id)


    def get_anns_by_gt_id(self, gt_id:[int, list]):
        return self.loadAnns(gt_id)

    def get_anns_by_img_id(self, id: [int, list]):
        """
        [{'area': 100,
           'bbox': [],
           'category_id': 16,
           'id': 500,
           'image_id': 107,
           'iscrowd: 0,
           'segmentation': [[1,2,34]]
        :param id:
        :return:
        """
        return self.get_anns_by_gt_id(self.get_gt_id_by_img_id(id))

    def get_contain_gt_images_id(self):
        """:return images id"""
        return self.imgToAnns.keys()


    def show_img_by_anns(self, img_file, anns):
        img = cv2.imread(img_file)
        filename = Path(img_file).name
        for label_i in anns:
            bbox = label_i['bbox']
            label_id = label_i['category_id']
            _draw_box_label(img, bbox, label=self.get_class_name_by_id(label_id)[0], color=color[label_id])

        # img = resize(img)
        # cv2.imshow(filename, img)
        # cv2.waitKey(0)
        show_img(img)

    def show_img_by_id(self, img_id):
        img_file = self.get_img_info([img_id])[0]['file_name']
        anns = self.get_anns_by_img_id(img_id)
        self.show_img_by_anns(img_file, anns)


    def train_test_split(self, test_size=0.2):
        """"""
        random.seed(42)
        imgids = self.get_img_allid()
        random.shuffle(imgids)
        random.shuffle(imgids)
        split_idx = int(len(imgids) * test_size)

        test_img_ids = imgids[:split_idx]
        train_img_ids = imgids[split_idx:]

        def gen_coco(img_ids):
            train_coco = {}
            train_coco['images'] = []
            train_coco['annotations'] = []
            train_coco['categories'] = self.json_data['categories']

            bbox_id = 0
            for index, train_img_id in enumerate(img_ids):
                image_item = self.get_img_info(train_img_id)[0]
                image_item['id'] = index
                train_coco['images'].append(image_item)

                for _, anno_index in enumerate(self.get_gt_id_by_img_id(train_img_id)):
                    annota_item = self.get_anns_by_gt_id(anno_index)[0]
                    annota_item['id'] = bbox_id
                    annota_item['image_id'] = index
                    train_coco['annotations'].append(annota_item)
                    bbox_id += 1
            return train_coco
            
        return self.from_json_data(gen_coco(train_img_ids), is_copy=True), self.from_json_data(gen_coco(test_img_ids), is_copy=True)