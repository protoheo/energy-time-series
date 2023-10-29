import pandas as pd
import os
import joblib
from torch import FloatTensor

from configs.setting import global_setting
from libs.train_utils import get_accuracy
from model.model_load import load_scaler, load_inference, inference_ensemble
from libs import train_utils

import torch


def ensemble_data_test(dir_path, df, config, num):
    x_cols = config['DATA']['X_COLS']
    y_cols = config['DATA']['Y_TARGET']

    test_x = df[x_cols]
    test_y = df[y_cols]

    # Set Scaler
    scaler_x = 'ckpt/{}/SCALER/x_scaler.pkl'.format(dir_path)
    scaler_y = 'ckpt/{}/SCALER/y_scaler.pkl'.format(dir_path)
    sc_x = joblib.load(scaler_x)
    sc_y = joblib.load(scaler_y)

    # Scale transform
    ensemble_cols = config['DATA']['ENSEMBLE_{}'.format(str(num))]

    tes_x = sc_x.transform(test_x)[:, ensemble_cols]
    tes_y = sc_y.transform(test_y)
    X_tensor = FloatTensor(tes_x)
    Y_tensor = FloatTensor(tes_y)

    return X_tensor, Y_tensor, [sc_x, sc_y]


def scaler_data_test(dir_path, df, config, full=False):
    import joblib
    from torch import FloatTensor

    x_cols = config['DATA']['X_COLS']
    y_cols = config['DATA']['Y_TARGET']

    if full:
        test = df
    else:
        _, _, test = train_utils.split_dataset(df, config)

    test_x = test[x_cols]
    test_y = test[y_cols]

    # Set Scaler
    scaler_x = 'ckpt/{}/SCALER/x_scaler.pkl'.format(dir_path)
    scaler_y = 'ckpt/{}/SCALER/y_scaler.pkl'.format(dir_path)
    sc_x = joblib.load(scaler_x)
    sc_y = joblib.load(scaler_y)

    tes_x = sc_x.transform(test_x)
    tes_y = sc_y.transform(test_y)
    X_tensor = FloatTensor(tes_x)
    Y_tensor = FloatTensor(tes_y)

    return X_tensor, Y_tensor, [sc_x, sc_y]


def baseline_test(make_predict=False):
    # 모델 및 토크나이저
    st1_dir = 'ws_t'
    config, device = global_setting('cfg.yaml', 'ckpt/{}'.format(st1_dir))
    model = inference_ensemble(st1_dir, device, config)

    # dataframes
    data_path = os.path.join(config['DATA']['DATA_PATH'], config['DATA']['FILE_NAME'])
    df = pd.read_excel(data_path)

    test_x, test_y, sc = scaler_data_test(st1_dir, df, config)

    outputs = model(test_x)

    print(get_accuracy(outputs, test_y))


def stage_test():
    from libs.core.ensemble import Ensemble

    st1_dir = 't2_2s_ws'
    st2_dir = '2s_st'

    # 모델 및 토크나이저
    config1, device = global_setting('cfg.yaml', 'ckpt/{}'.format(st1_dir))
    config2, _ = global_setting('cfg.yaml', 'ckpt/{}'.format(st2_dir))

    model1 = inference_ensemble(st1_dir, device, config1)
    model2 = inference_ensemble(st2_dir, device, config2)

    # dataframes
    data_path = os.path.join(config1['DATA']['DATA_PATH'], config1['DATA']['FILE_NAME'])
    df = pd.read_excel(data_path)

    test_x, _, _ = scaler_data_test(st1_dir, df, config1)
    _, test_y, sc = scaler_data_test(st2_dir, df, config2)

    mid = model1(test_x)
    outputs = model2(mid)

    print(get_accuracy(outputs, test_y))


def stage_test_v2(make_predict=False):
    from libs.core.ensemble import Ensemble

    st1_dir = '2s_ws'
    st2_dir = 'ws_t'

    # 모델 및 토크나이저
    config1, device = global_setting('cfg.yaml', 'ckpt/{}'.format(st1_dir))
    config2, _ = global_setting('cfg.yaml', 'ckpt/{}'.format(st2_dir))

    model1 = inference_ensemble(st1_dir, device, config1)
    model2 = inference_ensemble(st2_dir, device, config2)

    # dataframes
    data_path = os.path.join(config1['DATA']['DATA_PATH'], config1['DATA']['FILE_NAME'])
    df = pd.read_excel(data_path)

    test_x, _, _ = scaler_data_test(st1_dir, df, config1)
    mid_x, test_y, sc = scaler_data_test(st2_dir, df, config2)

    mid = model1(test_x)

    mid2 = torch.cat((test_x, mid), 1)
    outputs = model2(mid2)

    print(get_accuracy(outputs, test_y))
    if make_predict:
        testX, _, _ = scaler_data_test(st1_dir, df, config1, full=True)
        mid = model1(testX)
        mid2 = torch.cat((testX, mid), 1)
        outputs = model2(mid2).detach().numpy()
        inv_pred = sc[1].inverse_transform(outputs)

        testY = df[config2['DATA']['Y_TARGET']]

        # save_df = testY
        # save_df['pred'] = inv_pred
        # save_df.to_excel('2stage_lstm_ws_t2.xlsx', index=False)

        pass


def ensemble_test_total():
    from libs.core.ensemble import Ensemble

    st1_dir = 'ens_t2_w1'
    st2_dir = 'ens_t2_w2'

    st3_dir = 'ens_s1'
    st4_dir = 'ens_s2'

    # 모델 및 토크나이저
    config1, device = global_setting('cfg.yaml', 'ckpt/{}'.format(st1_dir))
    config2, _ = global_setting('cfg.yaml', 'ckpt/{}'.format(st2_dir))
    config3, device = global_setting('cfg.yaml', 'ckpt/{}'.format(st3_dir))
    config4, _ = global_setting('cfg.yaml', 'ckpt/{}'.format(st4_dir))

    model1 = inference_ensemble(st1_dir, device, config1, True)
    model2 = inference_ensemble(st2_dir, device, config2, True)
    model3 = inference_ensemble(st3_dir, device, config3, True)
    model4 = inference_ensemble(st4_dir, device, config4, True)
    # dataframes
    data_path = os.path.join(config1['DATA']['DATA_PATH'], config1['DATA']['FILE_NAME'])
    df = pd.read_excel(data_path)

    test_x1, _, sc1 = ensemble_data_test(st1_dir, df, config1, 1)
    test_x2, _, sc2 = ensemble_data_test(st2_dir, df, config2, 2)
    _, test_y, sc3 = ensemble_data_test(st3_dir, df, config3, 1)
    _, test_y, sc4 = ensemble_data_test(st4_dir, df, config4, 2)

    pred_1 = model1(test_x1)
    pred_2 = model2(test_x2)
    pred_mid = 0.5 * pred_1 + 0.5 * pred_2

    ens_col1 = config3['DATA']['ENSEMBLE_1']
    ens_col2 = config4['DATA']['ENSEMBLE_2']
    pred_3 = model3(pred_mid[:, ens_col1])
    pred_4 = model4(pred_mid[:, ens_col2])

    outputs = 0.5 * pred_3 + 0.5 * pred_4

    print(get_accuracy(outputs, test_y))


if __name__ == '__main__':
    # baseline_test()
    # stage_test()
    # ensemble_test_total()

    stage_test_v2(make_predict=True)
