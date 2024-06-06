# -*- coding: utf-8 -*-
"""
Time ： 2024/4/8 8:31
Auth ： xiazhichao
File ：yolo2coco.py
IDE ：PyCharm
Description:
"""
from pathlib import Path
import os, cv2
import numpy as np
from copy import deepcopy

from ..logger import logger, Constant
from ..check import verify_image
from ..path import scandir
from ..file_io import read_file, save_file
from ..bboxes import poly2mask, single_mask2rle

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format

class Yolo2coco():
    def __init__(self, folder=None, dataset_name=None, yaml_file=None, save_file=None, iscrowd=0, is_verify_img=False, **kwargs):
        """iscrowd: 0 for poly; 1 for rle(run-length encoding)
        dataset_name: train , val, test
        """

        self.dataset_name = dataset_name
        self.folder = Path(folder)
        self.classes = yaml_file
        dataset_name = dataset_name if dataset_name else 'train'
        assert dataset_name in ['train', 'val', 'test'], 'dataset_name must be one of ["train", "val", "test"]'
        self.save_path = Path(save_file).parent / f'{Path(save_file).stem}_{dataset_name}.json'
        self.iscrowd = iscrowd


        self.is_verify_img = is_verify_img
        if self.is_verify_img:
            logger.warning(f'verify every image ⚠️')
        self.init_config()

        self.img_index = 0
        self.bbox_id = 0
        self.vaild = 0
        self.miss_label = 0
        self.error_img = 0
        self.total = 0


    def init_config(self):
        self.coco = {}
        self.coco['images'] = []
        self.coco['annotations'] = []
        self.coco['categories'] = []


    def __call__(self, suffix=None, recursive=True, case_sensitive=False, **kwargs):
        self._add_category()
        img_formats = [f'*{i}' for i in Constant.IMG_FORMATS]
        suffix = suffix if suffix else img_formats
        img_path_inter = scandir(str(self.folder), suffix, recursive, mid_patten=f'**{self.dataset_name}**' if self.dataset_name else None)
        with ThreadPoolExecutor(max_workers=100) as excutor:
            pbar = tqdm(excutor.map(self._convert, img_path_inter), desc='yolo2coco', bar_format=TQDM_BAR_FORMAT)
            for image, label_path, shape, total, miss_label, error_img in pbar:
                self.total += 1
                if total:
                    self._add_image(str(image), shape)
                    self._add_anno(label_path, shape)
                    self.img_index += 1
                elif miss_label:
                    self.miss_label += 1
                elif error_img:
                    self.error_img += 1

        logger.info(f'total: {self.total}, missing label: {self.miss_label}, error img: {self.error_img}')
        save_file(self.save_path, self.coco)
        logger.info(f'json file saved: {self.save_path}')


    def _add_category(self):
        for index, classes in self._read_classes().items():
            category_item = dict()
            category_item['supercategor'] = None
            category_item['id'] = int(index)
            category_item['name'] = classes
            self.coco['categories'].append(category_item)

    def _read_classes(self):
        if isinstance(self.classes, (str, Path)):
            if Path(str(self.classes)).exists():
                categories = read_file(str(self.classes))['names']
                if isinstance(categories, dict):
                    return categories
                if isinstance(categories, list):  # old version
                    return {i:category for i, category in enumerate(categories)}
                raise Exception("classes is wrong ❌")
        if isinstance(self.classes, (list, tuple)):
            return {i: category for i, category in enumerate(self.classes)}
        raise Exception("classes is wrong ❌")

    def _add_image(self, img_path, shape):
        image_item = {}
        image_item['id'] = self.img_index
        image_item['file_name'] = img_path
        image_item['width'] = shape[1]
        image_item['height'] = shape[0]
        self.coco['images'].append(image_item)

    def _convert(self, image_path):
        total, miss_label, error_img, shape = 0, 0, 0, (0, 0, 0)
        if self.is_verify_img:
            verify_image(image_path)

        rel_path = image_path.replace(os.path.commonprefix([image_path, str(self.folder)]), "")
        rel_path = rel_path.replace('images', 'labels', 1)

        image = Path(image_path)
        name = image.stem

        label_path = (self.folder / rel_path[1:]).parent / f'{name}.txt'
        if label_path.exists() and label_path.stat().st_size:
            try:
                img = cv2.imread(str(image))
                shape = img.shape
                total = 1
            except:
                error_img = 1
                logger.error(f'read error: {image}')
        else:
            logger.error(f'{label_path} non_exists ❌')
            miss_label = 1
        return image, label_path, shape, total, miss_label, error_img


    def _add_anno(self, file, shape):
        with open(file, 'r', encoding='utf-8') as fr:
            for line_i in fr.readlines():
                annota_item = {}
                content = line_i.strip().split()
                lable = int(float(content[0]))
                bbox, segment, area = self.convert(np.array(content[1:]).astype(np.float32), shape)
                annota_item['id'] = self.bbox_id
                annota_item['image_id'] = self.img_index
                annota_item['category_id'] = lable
                annota_item['segmentation'] = segment
                annota_item['area'] = area
                annota_item['bbox'] = list(bbox)
                annota_item['iscrowd'] = self.iscrowd

                self.bbox_id += 1
                self.coco['annotations'].append(annota_item)

    def convert(self, bbox, shape):
        if len(bbox) > 4:
            seg = np.array(bbox).reshape(-1, 2)
            seg = seg * np.array([shape[1], shape[0]]).reshape(-1, 2)

            seg_copy = deepcopy(seg)
            seg_copy = seg_copy.astype('int')
            area = cv2.contourArea(seg_copy)
            segmentation = seg.reshape(1, -1).tolist()

            if self.iscrowd:
                mask = poly2mask(seg, shape[1], shape[0])
                segmentation = single_mask2rle(mask)

            x1, y1, x2, y2 = *np.min(seg, 0), *np.max(seg, 0)
            return [x1,y1,x2-x1,y2-y1], segmentation, area

        x1 = max(0,(bbox[0] - bbox[2] / 2.0) * shape[1])
        y1 = max(0, (bbox[1] - bbox[3] / 2.0) * shape[0])
        x2 = min((bbox[0] + bbox[2] / 2.0) * shape[1], shape[1])
        y2 = min((bbox[1] + bbox[3] / 2.0) * shape[0], shape[0])

        area = (x2 - x1) * (y2 - y1)
        return [x1,y1,x2-x1,y2-y1], [[x1,y1,x2,y2]], area