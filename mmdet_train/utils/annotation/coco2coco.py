# -*- coding: utf-8 -*-
"""
Time ： 2023/7/12 9:26
Auth ： xiazhichao
File ：coco2coco.py
IDE ：PyCharm
Description:
    merge mult coco json to one. label.yaml must be train label.
"""


from pathlib import Path
from ..logger import logger
from ..file_io import read_file, save_file
from .cocoapi import COCOapi




class Coco2coco():
    def __init__(self, cocolist=None, file_save=None, yaml_file=None):
        '''
        class_name: yaml, list
        :param cocolist: json file, list
        '''
        self.cocolist = cocolist
        self.file_save = file_save
        self.yaml_file = yaml_file
        id_name:dict = self._read_classes()
        self.name_id = {name: _id for _id, name in id_name.items()}
        self.init_config()
        self.img_index = 0
        self.bbox_id = 0


    def init_config(self):
        self.coco = {}
        self.coco['images'] = []
        self.coco['annotations'] = []
        self.coco['categories'] = []


    def _read_classes(self):
        if isinstance(self.yaml_file, (str, Path)):
            if Path(str(self.yaml_file)).exists():
                categories = read_file(str(self.yaml_file))['names']
                if isinstance(categories, dict):
                    return categories
                if isinstance(categories, list):  # old version
                    data = {i: category for i, category in enumerate(categories)}
                    logger.warning(f'⚠️ overwrite {self.yaml_file} to new version.')
                    save_file(str(self.yaml_file), {'name': data})
                    return data
                raise Exception("classes is wrong ❌")
        raise Exception(f"{self.yaml_file} non exists.")


    def __call__(self):
        self._add_categories()

        for json_i in self.cocolist:
            logger.info(f'{json_i} start ......')
            coco_data = COCOapi(json_i)
            id_name = coco_data.get_class_id_name()
            self._add_img_anno(coco_data, id_name)

        if self.file_save:
            save_file(str(self.file_save), self.coco)
            logger.info(f'done✅.json file save in {self.file_save}')
        logger.info(f'datasets coco2coco end.')
        return self.coco


    def _add_categories(self):
        for name, index in self.name_id.items():
            category_item = dict()
            category_item['supercategor'] = None
            category_item['id'] = int(index)
            category_item['name'] = name
            self.coco['categories'].append(category_item)


    def _add_img_anno(self, coco: 'COCOapi', id_name):
        images = coco.json_data['images']

        for index, image in enumerate(images):
            old_img_id = image['id']
            images[index]['id'] = self.img_index

            annotation:list = coco.get_anns_by_img_id(old_img_id)
            for i, annotation_i in enumerate(annotation):
                annotation[i]['id'] = self.bbox_id
                annotation[i]['image_id'] = self.img_index
                annotation[i]['category_id'] = self.name_id[id_name[annotation_i['category_id']]]
                self.bbox_id += 1
            self.coco['annotations'].extend(annotation)
            self.img_index += 1

        self.coco['images'].extend(images)
