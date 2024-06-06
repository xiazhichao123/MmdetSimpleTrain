# -*- coding: utf-8 -*-
"""
Time ： 2024/5/28 9:37
Auth ： xiazhichao
File ：typing_u.py
IDE ：PyCharm
Description:
"""


from typing import Optional, List, TypeVar, Callable, Any, Tuple, Dict


T = TypeVar('T')

ListT = List[T]
OptListT = Optional[ListT]

ListStr = List[str]
ListDict = List[Dict]

OptListInt = OptListT[int]
OptListFloat = OptListT[float]
OptListStr = OptListT[str]

OptionStr = Optional[str]
OptionInt = Optional[int]
OptinonFloat = Optional[float]


ListListT = List[ListT]
OptionListListT = Optional[List[ListT]]

