
import cv2
import random, colorsys
import platform

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
from .check import is_ascii
from pathlib import Path
import matplotlib.pyplot as plt

from .bboxes import xywh2xyxy, x1y1wh2xyxy

# random.seed(100)




def _ncolors(num, cv_flag=False):
  rgb_colors = []
  if num < 1:
    return rgb_colors

  def get_n_hls_colors(num):
      hls_colors = []
      i = 0
      step = 360.0 / num
      while i < 360:
          h = i
          s = 90 + random.random() * 10
          l = 50 + random.random() * 10
          _hlsc = [h / 360.0, l / 100.0, s / 100.0]
          hls_colors.append(_hlsc)
          i += step
      return hls_colors

  hls_colors = get_n_hls_colors(max(num, 50))
  for hlsc in hls_colors:
    _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
    r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
    if cv_flag:
        rgb_colors.append((b, g, r))
    else:
        rgb_colors.append((r, g, b))
    random.shuffle(rgb_colors)
  return rgb_colors

color = _ncolors(1000)
font_path = str(Path(__file__).absolute().parent / 'resert/Arial.Unicode.ttf')

def draw_box_label(img0, box=None, line_width=None, label='', color=(128, 128, 128), text_color=(255, 255, 255), font_size=0, **kwargs):
    '''box:(x1,y1,x2,y2)'''
    if not box:
        cv2.putText(img0, label, (132, 132), cv2.FONT_HERSHEY_SIMPLEX, line_width, text_color, thickness=tf, lineType=cv2.LINE_AA)
        return img0

    color = tuple(color)
    line_width = line_width or max(round(sum(img0.shape) / 2 * 0.003), 2)  # line width
    tf = max(line_width - 1, 1)  # font thickness
    non_ascii = not is_ascii(label)
    pil = non_ascii
    if pil:  # use PIL
        img0 = img0 if isinstance(img0, Image.Image) else Image.fromarray(cv2.cvtColor(img0,cv2.COLOR_BGR2RGB))
        font = ImageFont.truetype(font_path, font_size or max(round(sum(img0.size) / 2 * 0.035), 12))
        draw = ImageDraw.Draw(img0)
        draw.rectangle(box, width=line_width, outline=color)  # box
        if label:
            w, h = font.getsize(label)
            outside = box[1] - h >= 0  # label fits outside box
            draw.rectangle((box[0], box[1] - h if outside else box[1],
                            box[0] + w + 1,box[1] + 1 if outside else box[1] + h + 1),
                           fill=color)
            draw.text((box[0], box[1] - h if outside else box[1]), label, fill=text_color, font=font)
            img0 = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
        return img0

    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img0, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
    if label:
        tf = max(line_width - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=line_width / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img0, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img0, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, line_width / 3,
                    text_color, thickness=tf, lineType=cv2.LINE_AA)
    return img0


def _draw_region(img, region:list, color=[0, 0, 255], thickness=5, **kwargs):
    """

    :param img:
    :param region: [np[[1,2],],]
    :return:
    """
    img = cv2.polylines(img, region, isClosed=True, color=color, thickness=thickness, **kwargs)
    return img



def show_det_result(img, result, wtime=0, **kwargs):
    """

    :param img:
    :param result: [{'bbox': [175, 457, 223, 572], 'cls': 24, 'conf': 0.26, 'monitorAreaId': 1, 'targetClass': 'backpack'},
    :param wtime:
    :return:
    """

    if result:
        bbox = [result_i['bbox'] for result_i in result]
        conf = [result_i['conf'] for result_i in result]
        targetClass = [result_i['targetClass'] for result_i in result]
        img = show_img(img, bbox, labels=targetClass, label_other=conf, **kwargs)
        return img
    if isinstance(img, np.ndarray):   # cv
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        img.show()
    return img



def get_unique_mask_pix(mask):
    
    if isinstance(mask, type(None)):
        return None, None
    if isinstance(mask, (str, Path)):
        mask = Image.open(mask)
    elif isinstance(mask, np.ndarray):   # cv
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=2).repeat(3, axis=2)

    mask = np.array(mask)
    unique_pix = np.unique(mask.reshape(mask.shape[0]*mask.shape[1], mask.shape[2]), axis=0).tolist()
    return unique_pix, mask


def append_mask_color(mask=None, img=None, alpha=0.5, mask_index:list=None):
    unique_pix, mask = get_unique_mask_pix(mask)
    
    if not unique_pix:
        return None, None
    if mask_index:
        unique_pix = [unique_pix[i] for i in mask_index]
    color_s = _ncolors(len(unique_pix))
  
    mask_img = np.zeros_like(mask)
    for i, c_i in enumerate(unique_pix):
        pss = list(map(lambda x: x.tolist(), np.where(np.all(mask == c_i, axis=-1))))
        mask_img[pss[0], pss[1]] = color_s[i]

    img = np.array(img)
    img = (img* alpha + mask_img *(1-alpha)).astype(np.uint8)
    return img, mask_img


def show_img(img, bboxs=None,mask=None, reverse_rgb=False, labels:list='', label_other='', format='xyxy', is_jupyter=False, **kwargs):
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):   # cv
        img = Image.fromarray(img.astype(np.uint8))

    
    merge_img, mask_img = append_mask_color(mask, img, kwargs.get('alpha', 0.5), kwargs.get('mask_index', None))
    if not isinstance(merge_img, type(None)):
        return show_img(merge_img, is_jupyter=is_jupyter), show_img(mask_img, is_jupyter=is_jupyter)
    
    draw = ImageDraw.Draw(img)
    position = kwargs.get('position', (10, 10))
    font = ImageFont.truetype(kwargs.get('font', font_path), kwargs.get('font_size', 20))
    if isinstance(bboxs, type(None)):
        if labels:
            draw.text(position, labels, fill=kwargs.get('text_color', (255,255,255)), font=font)
        if reverse_rgb:
            img = Image.fromarray(np.array(img)[:, :, ::-1])
        if not is_jupyter:
            img.show()
        return img
    
    norm = False
    if 'n' in format:
        norm = True
        format = format.replace('n', '')

    w, h = img.size
    bboxs = np.array(bboxs) if isinstance(bboxs, list) else bboxs
    bboxs = bboxs.reshape(-1, 4) if len(bboxs.shape) == 1 else bboxs

    if format == 'x1y1wh':
        bboxs = x1y1wh2xyxy(bboxs, w=w, h=h, norm=norm)

    if isinstance(labels, str):
        labels = [labels] * len(bboxs)
    if isinstance(label_other, str):
        label_other = [label_other] * len(bboxs)
    
    list_label = list(set(labels))
    n_color = [kwargs['color']]*len(list_label) if 'color' in kwargs else _ncolors(len(list_label))
    
    for index, (bbox, label) in enumerate(zip(bboxs, labels)):
        bbox = bbox.tolist()
        draw.rectangle(bbox, width=kwargs.get('line_width', 3), outline=n_color[list_label.index(label)])  # box
        if label:
            w1, h1 = 2,2
            outside = bbox[1] - h1 >= 0  # label fits outside box
            draw.rectangle((bbox[0], bbox[1] - h1 if outside else bbox[1],
                            bbox[0] + w1 + 1,bbox[1] + 1 if outside else bbox[1] + h1 + 1), fill=kwargs.get('text_color', (255,255,255)))
            draw.text((bbox[0], bbox[1] - kwargs.get('font_size', 20)-h1 if outside else bbox[1]), str(label)+' '+str(label_other[index]), fill=kwargs.get('text_color', (255,255,255)), font=font)
    if reverse_rgb:
        img = Image.fromarray(np.array(img)[:, :, ::-1])
    if not is_jupyter:
        img.show()
    return img
