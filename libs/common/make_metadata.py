import os

import pandas as pd

from libs.common.common import find_all_files, get_type


def df_spliter(df):
    train = df.sample(frac=0.6, random_state=200)  # random state is a seed value
    tmp = df.drop(train.index)
    valid = tmp.sample(frac=0.5, random_state=200)  # random state is a seed value
    test = tmp.drop(valid.index)

    return train, valid, test


def make_dataframe(target_dir, platform_sys='Windows'):
    path_files = []
    path_files = find_all_files(target_dir, path_files)
    path_files = get_type(path_files, type_list=['jpg', 'png'])
    if platform_sys == 'Windows':
        target_keyword = '\\mask'
        replace_keyword = '\\ori'
    else:
        target_keyword = '/mask'
        replace_keyword = '/ori'

    total = []
    error_list = []
    for f in path_files:
        ff = f.replace(target_keyword, replace_keyword).replace('.png', '.jpg')
        total.append([ff, f])

    os.makedirs('data', exist_ok=True)

    if len(total) > 0:
        df = pd.DataFrame(total, columns=['img_path', 'mask_path'])

        df_cls1 = df[df['img_path'].str.contains("garibi")]
        df_cls2 = df[df['img_path'].str.contains("oyster")]

        cls1_train, cls1_valid, cls1_test = df_spliter(df_cls1)
        cls2_train, cls2_valid, cls2_test = df_spliter(df_cls2)

        print("cls1 Train / cls1 Vaild / cls1 Test", len(cls1_train), len(cls1_valid), len(cls1_test))
        print("cls2 Train / cls2 Vaild / cls2 Test", len(cls2_train), len(cls2_valid), len(cls2_test))

        train = pd.concat([cls1_train, cls2_train])
        vaild = pd.concat([cls1_valid, cls2_valid])
        test = pd.concat([cls1_test, cls2_test])

        print("total: ", len(train), len(vaild), len(test))

        train.to_csv('./data/train_data.csv', index=False)
        vaild.to_csv('./data/valid_data.csv', index=False)
        test.to_csv('./data/test_data.csv', index=False)

    if len(error_list) != 0:
        with open("errors.txt", 'w') as w:
            for line in error_list:
                w.write(str(line) + "\n")


def make_dataframe_v2(target_dir, platform_sys):
    path_files = find_all_files(target_dir, [])
    path_files = get_type(path_files, type_list=['jpg', 'png'])
    if platform_sys == 'Windows':
        target_keyword = '_aug'
        replace_keyword = '_mask'
    else:
        target_keyword = '_aug'
        replace_keyword = '_mask'

    total = []
    error_list = []
    for target in path_files:
        mask = target.replace(target_keyword, replace_keyword)
        total.append([target, mask])

    os.makedirs('meta', exist_ok=True)

    if len(total) > 0:
        df = pd.DataFrame(total, columns=['img_path', 'mask_path'])

        train, vaild, test = df_spliter(df)

        print("total: ", len(train), len(vaild), len(test))

        train.to_csv('./meta/train_set.csv', index=False)
        vaild.to_csv('./meta/valid_set.csv', index=False)
        test.to_csv('./meta/test_set.csv', index=False)

    if len(error_list) != 0:
        with open("errors.txt", 'w') as w:
            for line in error_list:
                w.write(str(line) + "\n")