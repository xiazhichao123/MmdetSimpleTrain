# -*- coding: utf-8 -*-
"""
Time ： 2024/2/5 10:39
Auth ： xiazhichao
File ：logger.py
IDE ：PyCharm
Description:
    
"""

from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


class Constant():
    # const
    IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm', \
                  'BMP', 'DNG', 'JPEG', 'JPG', 'MPO', 'PNG', 'TIF', 'TIFF', 'WEBP', 'PFM'



@dataclass
class Global_val():
    train_project_config_path: Optional[str] = field(default=None, metadata={'help': 'config: /home/train/workspace/project/coco128/huggingface/config.yaml'})
    hf_cache_dir: Optional[str] = field(default=None, metadata={'help': ''})
    torch_cache_dir: Optional[str] = field(default=None, metadata={'help': ''})

    datasets_cfg: Optional[dict] = field(default=None, metadata={'help': ''})



time_format = {
    # W0,W1
    'when': 'midnight',  # S M H D midnight - roll over at midnight W{0-6} - roll over on a certain day; 0 - Monday
    'interval': 1,
    'backup_nums': 2,
    'encoding': 'utf-8'
}



size_format = {
    'max_M': 2,
    'backup_nums': 2,
    'encoding': 'utf-8'
}



LOG_INFO = {
    'log_filename': Path.cwd() / 'log.log',
    'dir_out': False,
    'log_level': 'DEBUG',
    'time_format': time_format,
    'size_format': size_format,
    'size_format_flag': True,
    'verbose': True      # False for only print error log
}



class ColoredFormatter(logging.Formatter):
    GREEN = "\033[32m"
    RED = "\033[31m"
    PINK = "\033[35m"
    RESET = "\033[0m"
    def format(self, record):
        if record.levelno == logging.INFO:
            record.msg = f"{self.GREEN}{record.msg}{self.RESET}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{self.RED}{record.msg}{self.RESET}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{self.PINK}{record.msg}{self.RESET}"
        return super().format(record)


class BaseLogger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def init(self):
        rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
        self.log_level = self.log_level.upper() if self.log_level else self.kwargs['log_level']
        level = self.log_level if self.kwargs.get('verbose') and rank in {-1, 0} else 'ERROR'

        self.filename = Path(self.kwargs['log_filename']).with_suffix('.'+level.lower())
        # self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.name)
        # self.logger = logging.basicConfig()
        self.logger.setLevel(level)
        self.LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'  # 格式

        '''定义handler的输出格式'''
        formatter = ColoredFormatter(self.LOG_FORMAT)

        if self.kwargs['dir_out']:
            '''按大小切分'''
            if self.kwargs['size_format_flag']:
                size_handler = RotatingFileHandler(self.filename,
                                                   maxBytes=1024 * self.kwargs['size_format']['max_M'],
                                                   backupCount=self.kwargs['size_format']['backup_nums'],
                                                   encoding=self.kwargs['size_format']['encoding'])
                size_handler.setLevel(level)
                size_handler.setFormatter(formatter)
                self.logger.addHandler(size_handler)
            else:
                time_handler = TimedRotatingFileHandler(self.filename,
                                                        when=self.kwargs['time_format']['when'],
                                                        interval=self.kwargs['time_format']['interval'],
                                                        backupCount=self.kwargs['time_format']['backup_nums'],
                                                        encoding=self.kwargs['time_format']['encoding'])
                time_handler.setLevel(level)
                time_handler.setFormatter(formatter)
                self.logger.addHandler(time_handler)
        else:
            '''输出至控制台'''
            console = logging.StreamHandler()
            console.setLevel(level)
            console.setFormatter(formatter)
            self.logger.addHandler(console)

    def get_log(self, name: str, level: str=None) -> logging.Logger:
        '''log_level: DEBUG,INFO,WARNING,ERROR'''
        self.name = name
        self.log_level = level
        self.init()
        return self.logger


def init_log(root_path: Path=None, **config) -> 'BaseLogger':
    if not os.sep in config['log_filename']:  # filename
        config['log_filename'] = str(Path(root_path) / 'log' / config['log_filename'])
    Path(config['log_filename']).parent.mkdir(parents=True, exist_ok=True)

    logger_instant = BaseLogger(**config)
    return logger_instant


def set_logging(name: str='train', verbose=True) -> logging.Logger:
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR

    logger = logging.getLogger(name)
    # self.logger = logging.basicConfig()
    logger.setLevel(level)
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'  # 格式

    '''定义handler的输出格式'''
    formatter = ColoredFormatter(log_format)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

logger: logging.Logger = set_logging()
# error_logger: logging.Logger = set_logging('error')

if __name__ == "__main__":
    logger_instant = BaseLogger(**LOG_INFO)
    logger = logger_instant.get_log(__name__)
    logger.info('ddddE')