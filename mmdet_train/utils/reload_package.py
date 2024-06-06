# -*- coding: utf-8 -*-
"""
Time ： 2024/2/5 15:10
Auth ： xiazhichao
File ：reload_package.py
IDE ：PyCharm
Description:
    
"""


from typing import Any
import sys


def Reload(root_package: str, package: str, func: Any):
    """
    from deploy.reload_package import Reload
    from deploy.utils import logger

    logger.logger = logger.set_logging('tttt')
    Reload('deploy', 'deploy.utils.logger', logger)
    from deploy.utils import Registry, a

    a()
    """
    # sys.modules = dict(filter(lambda x: not x[0].startswith(root_package), sys.modules.items()))
    remove_key = [key for key in sys.modules if key.startswith(root_package)]
    for key in remove_key:
        sys.modules.pop(key)

    sys.modules[package] = func