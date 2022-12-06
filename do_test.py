import pandas as pd
import os

from libs.core.tester import Tester
from configs.setting import global_setting
from model.model_load import load_scaler, load_inference
from libs import train_utils

from dataload.dataloader import build_dataloader


def main_test():
    # 모델 및 토크나이저
    config, device = global_setting('cfg.yaml')
    model = load_inference(device, config)

    # dataframes
    data_path = os.path.join(config['DATA']['DATA_PATH'], config['DATA']['FILE_NAME'])
    df = pd.read_excel(data_path)

    train, valid, test = train_utils.split_dataset(df, config)

    # Define Dataset
    weather_cols = config['DATA']['X_COLS']
    target_cols = config['DATA']['Y_TARGET']

    test_weather_x = test[weather_cols]
    test_weather_y = test[target_cols]

    sc_x, sc_y = load_scaler()
    test_x = sc_x.transform(test_weather_x.values)
    test_y = sc_y.transform(test_weather_y.values)

    test_loader = build_dataloader([test_x, test_y], config=config, mode='test')

    test = Tester(config=config,
                  model=model,
                  test_loader=test_loader,
                  device=device,
                  scaler=sc_y)

    test.test()


def ensemble_test():
    from libs.core.ensemble import Ensemble

    # 모델 및 토크나이저
    config1, device = global_setting('cfg1.yaml')
    config2, _ = global_setting('cfg2.yaml')

    model1 = load_inference(device, config1)
    model2 = load_inference(device, config2)

    # dataframes
    data_path = os.path.join(config1['DATA']['DATA_PATH'], config1['DATA']['FILE_NAME'])
    df = pd.read_excel(data_path)

    train, valid, test = train_utils.split_dataset(df, config1)

    # Define Dataset
    weather_cols = config1['DATA']['X_COLS']
    target_cols = config2['DATA']['Y_TARGET']

    test_weather_x = test[weather_cols]
    test_weather_y = test[target_cols]

    sc_x, sc_y = load_scaler()
    test_x = sc_x.transform(test_weather_x.values)
    test_y = sc_y.transform(test_weather_y.values)

    test_loader = build_dataloader([test_x, test_y], config=config1, mode='test')

    test = Ensemble(config=[config1, config2],
                    model=[model1, model2],
                    test_loader=test_loader,
                    device=device,
                    scaler=sc_y)

    test.test()


if __name__ == '__main__':
    main_test()
