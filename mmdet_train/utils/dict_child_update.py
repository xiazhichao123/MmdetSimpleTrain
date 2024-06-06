# -*- coding: utf-8 -*-
"""
Time ： 2024/2/18 16:49
Auth ： xiazhichao
File ：dict_child_update.py
IDE ：PyCharm
Description:
"""


def find_all_child_keys(child_keys: dict, dict1: dict, parent='', temp_list=[], force=True):
    '''child_keys: defalut dict()'''
    for key, value in dict1.items():
        if isinstance(value, dict):
            child_keys[key] = parent
            # find_all_child_keys(child_keys, value, parent, force)
            temp_list.append((value,  parent + f' {key}'))
        else:
            if force:
                assert key not in child_keys, f"key {key} already"
            child_keys[key] = parent

    while temp_list:
        dict1, parent = temp_list.pop()
        find_all_child_keys(child_keys, dict1, parent, temp_list, force)


# find_all_child_keys(child_keys, dict1, '')
def auto_update_dict_value(child_keys, dict1: dict, dict2:dict=None, is_all=False):
    dict2 = dict2 if dict2 else {}
    def find_child(dict2):
        for key, value in dict2.items():
            if is_all:
                dict1[key] = value
            if isinstance(value, dict):
                find_child(value)
            else:
                if key in child_keys:
                    if child_keys[key]:
                        temp = 'dict1'
                        for value_i in child_keys[key].strip().split(' '):
                            temp += f'["{value_i}"]'
                        temp += f'["{key}"]={"value"}'
                        exec(temp)
                    else:
                        dict1[key] = value
    find_child(dict2)



if __name__ == '__main__':
    a = {
        'header': {
            'X-APP-ID': '8cc20c0154638547c362ece5919f741b',
            'ff': '4304ca85b6b320e43076c580b6da04d3',

        },
        'as': {
            'X-APP-ID': '8cc20c0154638547c362ece5919f741b',
            'ff': '4304ca85b6b320e43076c580b6da04d3'
        }

    }
    from collections import OrderedDict
    b = {'X-APP-ID': 'gg'}
    child_keys = OrderedDict()
    find_all_child_keys(child_keys, a, '', force=False)
    auto_update_dict_value(child_keys, a, b)
    print(a)