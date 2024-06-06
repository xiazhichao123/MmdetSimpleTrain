# mmdetection框架训练简化
## 1. 动机
[mmdetection](https://github.com/open-mmlab/mmdetection) 框架是一个很流行的目标检测框架，包括了很多目标检测模型。然而训练一个mmdetection模型对新手小白而言并不是件容易的事。虽然相关教程文档很多，但总觉的混乱不清晰。除此之外，要训练一个模型，需要在源码的特定的目录下运行[train.py](https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#train-with-customized-datasets)，并且添加相应的配置文件，这势必会造成对源码文件的污染。

## 特点

- 与mmdetection源码完全解耦，只需在python环境中安装mmdetection包
- 项目隔离。每个项目都是单独的一个文件夹

## 安装

见[mmdetection](https://mmdetection.readthedocs.io/en/latest/get_started.html).后续程序缺少包报错时，再缺啥pip啥。

## 使用

### 新建项目

在workspace/project目录下新建项目文件夹，比如需要对coco128数据集进行训练，则新建coco128文件夹，即workspace/project/coco128

### 数据集准备

必须使用coco格式的数据集。coco格式中的图片路径必须为绝对路径

[coco128数据集下载](https://ultralytics.com/assets/coco128.zip)

下载解压后的为yolo格式，可在workspace/yolo2coco.py中将其转换成coco格式。

### 新建datasets文件夹

新建workspace/project/coco128/datasets。该文件夹用于存放coco格式的json文件以及数据集的标签yaml文件。数据集的标签yaml文件名必须为xc-meta-data.yaml。

### 新建config.yaml文件

新建workspace/project/coco128/config.yaml。可以将已经存在的workspace/project/coco128/config.yaml复制到自己的项目中，然后更改里面一些必要的内容：

```
&__project_name: coco128    # 项目的名称
root_dir: <your workspace path>      # workspace的绝对路径
datasets:             # # 相对于workspace的路径，当然也可以使用绝对路径
  train_datasets: [project/coco128/datasets/coco128_train.json]  
  val_datasets: [project/coco128/datasets/coco128_train.json]   # 测试集
  
  meta_data: project/coco128/datasets/xc-meta-data.yaml
  
  
model_args:    相对于workspace的路径
  config: weights/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py    
  load_from: weights/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
  
  
  cfg_options:   # 更改默认模型文件中的参数。即config中的参数
      train_dataloader.batch_size: 2
      default_hooks.logger.interval: 1
      default_hooks.checkpoint.interval: 1
      val_dataloader.batch_size: 2
      train_cfg.max_epochs: 10
```

### 下载模型配置和文件

[模型列表](https://mmdetection.readthedocs.io/en/latest/model_zoo.html)。比如下载faster_rcnn下的faster-rcnn_r50_fpn_1x_coco.py和预训练模型权重faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth。将其放入workspace/weights/faster_rcnn目录下。

### 更改py文件中内容

下载的模型.py文件中含有\_base_字段，其中的内容是相对路径。要将相对路径更改为以mmdet..开始，后面接configs目录下的相对路径。比如[faster-rcnn_r101_fpn_1x_coco.py](https://github.com/open-mmlab/mmdetection/blob/main/configs/faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py)中_base_ = './faster-rcnn_r50_fpn_1x_coco.py'。而faster-rcnn_r50_fpn_1x_coco.py相对于configs的路径为faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py，因此需要将base_ = './faster-rcnn_r50_fpn_1x_coco.py'更改为base_ = 'mmdet../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

### 更改config.yaml

更改/workspace/config.yaml中的内容

```
log_format:
  dir_out: False    # True for file。为True时则会输出至worspace/project/<project_name>/log.log中
  
project_name: <project_name>     # 比如coco128
```

### 运行trainner.py

在workspace目录下运行trainner.py。

**说明**：初次运行时，会在权重存放的目录中生成一个以_new.py为后缀的新的配置文件，这个配置文件包含了所有的配置信息。可以通过这个信息来更改workspace/project/coco128/config.yaml中cfg_options下的内容。

## 最终的模型文件

训练结束后，权重文件会保存在项目文件夹的output中，而模型配置文件可选用新生成的以_new.py的文件。