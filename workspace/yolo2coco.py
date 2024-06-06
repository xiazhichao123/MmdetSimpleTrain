
from pathlib import Path
import sys, os

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from mmdet_train.utils.annotation.yolo2coco import Yolo2coco

import os


if __name__ == '__main__':
    folder = r'H:\xc_coding\pycode\train\workspace\project\coco128\datasets'
    yaml_file = os.path.join(r'H:\open_sources\MmdetSimpleTrain\workspace\project\coco128\datasets', 'xc-meta-data.yaml') # cannot change `xc-meta-data.yaml`
    save_file = os.path.join(r'H:\open_sources\MmdetSimpleTrain\workspace\project\coco128\datasets', 'coco128.json')

    iscrowd = 0  # 1 for rle 0; 0 for poly
    is_verify_img = False
    suffix = None  # or *.jpg or [*.jpg

    kwargs = {
        'folder': folder,
        'dataset_name':'',
        'yaml_file': yaml_file,
        'save_file': save_file,
        'iscrowd': iscrowd,
        'is_verify_img': is_verify_img,
        'suffix': suffix
    }

    Yolo2coco(**kwargs)(**kwargs)