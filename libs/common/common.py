import yaml
import os


def load_config(cfg_path):
    return yaml.full_load(open(cfg_path, 'r', encoding='utf-8-sig'))


def find_all_files(root_dir, ret_list=[]):
    for it in os.scandir(root_dir):
        if it.is_file():
            ret_list.append(it.path)

        if it.is_dir():
            ret_list = find_all_files(it, ret_list)
    return ret_list


def get_type(path_list, type_list=[]):
    ret_list = []
    if len(type_list) > 0:
        for path_row in path_list:
            ext_split = path_row.split(".")[-1]

            if ext_split in type_list:
                ret_list.append(path_row)
    else:
        ret_list = path_list

    return ret_list
