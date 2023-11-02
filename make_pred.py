import pandas as pd
import os
import joblib
from torch import FloatTensor

from libs.core.ensemble import load_scaler_ensemble
from libs.core.tester import Tester
from configs.setting import global_setting
from model.model_load import load_scaler, load_inference, inference_ensemble
from libs import train_utils

from dataload.dataloader import build_testloader


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
    X_tensor = FloatTensor(tes_x)

    return X_tensor, test_y, [sc_x, sc_y]


def scaler_data_test(dir_path, df, config):
    x_cols = config['DATA']['X_COLS']
    y_cols = config['DATA']['Y_TARGET']

    test_x = df[x_cols]
    test_y = df[y_cols]

    # Set Scaler
    scaler_x = 'ckpt/{}/SCALER/x_scaler.pkl'.format(dir_path)
    scaler_y = 'ckpt/{}/SCALER/y_scaler.pkl'.format(dir_path)
    sc_x = joblib.load(scaler_x)
    sc_y = joblib.load(scaler_y)

    tes_x = sc_x.transform(test_x)
    X_tensor = FloatTensor(tes_x)

    return X_tensor, test_y, [sc_x, sc_y]


def baseline_test():
    # 모델 및 토크나이저
    st1_dir = 'bilstm_baseline'
    config, device = global_setting('cfg.yaml', 'ckpt/{}'.format(st1_dir))
    model = inference_ensemble(st1_dir, device, config).to('cpu')

    # dataframes
    data_path = os.path.join(config['DATA']['DATA_PATH'], config['DATA']['FILE_NAME'])
    df = pd.read_excel(data_path)

    test_x, test_y, sc = scaler_data_test(st1_dir, df, config)

    outputs = model(test_x)
    pred = sc[1].inverse_transform(outputs.float().detach().numpy())

    save_df = test_y
    save_df['pred'] = pred
    save_df.to_excel('{}.xlsx'.format(st1_dir), index=False)


def stage_test():
    from libs.core.ensemble import Ensemble

    st1_dir = 'bilstm_ws'
    st2_dir = 'bilstm_st'

    # 모델 및 토크나이저
    config1, device = global_setting('cfg.yaml', 'ckpt/{}'.format(st1_dir))
    config2, _ = global_setting('cfg.yaml', 'ckpt/{}'.format(st2_dir))

    model1 = inference_ensemble(st1_dir, device, config1).to('cpu')
    model2 = inference_ensemble(st2_dir, device, config2).to('cpu')

    # dataframes
    data_path = os.path.join(config1['DATA']['DATA_PATH'], config1['DATA']['FILE_NAME'])
    df = pd.read_excel(data_path)

    test_x, _, _ = scaler_data_test(st1_dir, df, config1)
    _, test_y, sc = scaler_data_test(st2_dir, df, config2)

    mid = model1(test_x)
    outputs = model2(mid)

    pred = sc[1].inverse_transform(outputs.float().detach().numpy())

    save_df = test_y
    save_df['pred'] = pred
    save_df.to_excel('{}_{}.xlsx'.format(st1_dir, st2_dir), index=False)


def ensemble_test_total():
    from libs.core.ensemble import Ensemble

    st1_dir = 'bilstm_ws_ens1'
    st2_dir = 'bilstm_ws_ens2'

    st3_dir = 'bilstm_st_ens1'
    st4_dir = 'bilstm_st_ens2'

    # 모델 및 토크나이저
    config1, device = global_setting('cfg.yaml', 'ckpt/{}'.format(st1_dir))
    config2, _ = global_setting('cfg.yaml', 'ckpt/{}'.format(st2_dir))
    config3, device = global_setting('cfg.yaml', 'ckpt/{}'.format(st3_dir))
    config4, _ = global_setting('cfg.yaml', 'ckpt/{}'.format(st4_dir))

    model1 = inference_ensemble(st1_dir, device, config1, True).to('cpu')
    model2 = inference_ensemble(st2_dir, device, config2, True).to('cpu')
    model3 = inference_ensemble(st3_dir, device, config3, True).to('cpu')
    model4 = inference_ensemble(st4_dir, device, config4, True).to('cpu')
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

    pred = sc4[1].inverse_transform(outputs.float().detach().numpy())

    save_df = test_y
    save_df['pred'] = pred
    save_df.to_excel('{}_{}.xlsx'.format(st1_dir, st3_dir), index=False)


if __name__ == '__main__':
    # baseline_test()
    # stage_test()
    ensemble_test_total()