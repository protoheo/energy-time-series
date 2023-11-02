import pandas as pd
import os

from libs.core.ensemble import load_scaler_ensemble
from libs.core.tester import Tester
from configs.setting import global_setting
from libs.train_utils import get_accuracy
from model.model_load import load_scaler, load_inference, inference_ensemble
from libs import train_utils

from dataload.dataloader import build_testloader


def scaler_data_test(dir_path, df, config):
    import joblib
    from torch import FloatTensor

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
    tes_y = sc_y.transform(test_y)
    X_tensor = FloatTensor(tes_x)
    Y_tensor = FloatTensor(tes_y)

    return X_tensor, Y_tensor, [sc_x, sc_y]


def main_test():
    # 모델 및 토크나이저
    st1_dir = 'bilstm_st'
    config, device = global_setting('cfg.yaml', 'ckpt/{}'.format(st1_dir))
    model = inference_ensemble(st1_dir, device, config)

    # dataframes
    data_path = os.path.join(config['DATA']['DATA_PATH'], config['DATA']['FILE_NAME'])
    df = pd.read_excel(data_path)

    test_x, test_y, sc = scaler_data_test(st1_dir, df, config)

    outputs = model(test_x.to(device))

    print(get_accuracy(outputs, test_y))


def stage_test():
    from libs.core.ensemble import Ensemble

    st1_dir = 'bilstm_ws'
    st2_dir = 'bilstm_st'

    # 모델 및 토크나이저
    config1, device = global_setting('cfg.yaml', 'ckpt/{}'.format(st1_dir))
    config2, _ = global_setting('cfg.yaml', 'ckpt/{}'.format(st2_dir))

    model1 = inference_ensemble(st1_dir, device, config1)
    model2 = inference_ensemble(st2_dir, device, config2)

    # dataframes
    data_path = os.path.join(config1['DATA']['DATA_PATH'], config1['DATA']['FILE_NAME'])
    df = pd.read_excel(data_path)

    train, valid, test = train_utils.split_dataset(df, config1)

    # Define Dataset
    weather_cols = config1['DATA']['X_COLS']
    target_cols = config2['DATA']['Y_TARGET']

    test_weather_x = test[weather_cols]
    test_weather_y = test[target_cols]

    sc_x, sc_y = load_scaler_ensemble(st1_dir, st2_dir)
    test_x = sc_x.transform(test_weather_x.values)
    test_y = sc_y.transform(test_weather_y.values)

    test_loader = build_testloader([test_x, test_y])

    test = Ensemble(config=[config1, config2],
                    model=[model1, model2],
                    test_loader=test_loader,
                    device=device,
                    scaler=sc_y)

    test.test()


def ensemble_test_v1():
    from libs.core.ensemble import Ensemble

    st1_dir = 'bilstm_ws'
    st2_dir = 'bilstm_st'

    # 모델 및 토크나이저
    config1, device = global_setting('cfg.yaml', 'ckpt/{}'.format(st1_dir))
    config2, _ = global_setting('cfg.yaml', 'ckpt/{}'.format(st2_dir))

    model1 = inference_ensemble(st1_dir, device, config1, False)
    model2 = inference_ensemble(st2_dir, device, config2, False)

    # dataframes
    data_path = os.path.join(config1['DATA']['DATA_PATH'], config1['DATA']['FILE_NAME'])
    df = pd.read_excel(data_path)

    train, valid, test = train_utils.split_dataset(df, config1)

    # Define Dataset
    weather_cols = config1['DATA']['X_COLS']
    target_cols = config2['DATA']['Y_TARGET']

    test_weather_x = test[weather_cols]
    test_weather_y = test[target_cols]

    sc_x, sc_y = load_scaler_ensemble(st1_dir, st2_dir)
    test_x = sc_x.transform(test_weather_x.values)
    test_y = sc_y.transform(test_weather_y.values)

    test_loader = build_testloader([test_x, test_y])

    test = Ensemble(config=[config1, config2],
                    model=[model1, model2],
                    test_loader=test_loader,
                    device=device,
                    scaler=sc_y)

    test.merge_test()


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

    train, valid, test = train_utils.split_dataset(df, config1)

    # Define Dataset
    weather_cols = config1['DATA']['X_COLS']
    target_cols = config3['DATA']['Y_TARGET']

    test_weather_x = test[weather_cols]
    test_weather_y = test[target_cols]

    sc_x, sc_y = load_scaler_ensemble(st1_dir, st3_dir)
    test_x = sc_x.transform(test_weather_x.values)
    test_y = sc_y.transform(test_weather_y.values)

    test_loader = build_testloader([test_x, test_y])

    test = Ensemble(config=[config1, config2, config3, config4],
                    model=[model1, model2],
                    test_loader=test_loader,
                    device=device,
                    scaler=sc_y)

    test.total_test(stage_model=[model3, model4])


if __name__ == '__main__':
    # main_test()
    # stage_test()
    # ensemble_test_v1()

    ensemble_test_total()
