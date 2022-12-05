import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

from libs.common.common import scaler_save
from libs.core.trainer import Trainer
from configs.setting import global_setting
from model.model_load import load_model
from model.opt_load import opt_load
from model.loss_load import loss_load
from libs.logger.csv_logger import CSVLogger
from libs.callbacks.early_stopping import EarlyStopping
from libs.callbacks.save_checkpoint import SaveCheckPoint
from libs import train_utils
from dataload.dataloader import build_dataloader


def main():
    # 모델 및 토크나이저
    config, device = global_setting('cfg.yaml')

    # dataframes
    data_path = os.path.join(config['DATA']['DATA_PATH'], config['DATA']['FILE_NAME'])
    df = pd.read_excel(data_path)

    train, valid, test = train_utils.split_dataset(df, config)

    # Define Dataset
    weather_cols = config['DATA']['X_COLS']
    target_cols = config['DATA']['Y_TARGET']

    train_weather_x = train[weather_cols]
    valid_weather_x = valid[weather_cols]
    test_weather_x = test[weather_cols]

    train_weather_y = train[target_cols]
    valid_weather_y = valid[target_cols]
    test_weather_y = test[target_cols]

    # Set Scaler
    sc_x = MinMaxScaler()
    sc_x.fit(train_weather_x)
    sc_y = MinMaxScaler()
    sc_y.fit(train_weather_y)
    scaler_save(sc_x, name='x_scaler')
    scaler_save(sc_y, name='y_scaler')

    train_x = sc_x.transform(train_weather_x)
    valid_x = sc_x.transform(valid_weather_x)
    test_x = sc_x.transform(test_weather_x)

    # train_y = train_weather_y[target_cols].values
    # valid_y = valid_weather_y[target_cols].values
    # test_y = test_weather_y[target_cols].values

    train_y = sc_y.transform(train_weather_y[target_cols].values)
    valid_y = sc_y.transform(valid_weather_y[target_cols].values)
    test_y = sc_y.transform(test_weather_y[target_cols].values)

    train_loader = build_dataloader([train_x, train_y], config=config, mode="train")
    valid_loader = build_dataloader([valid_x, valid_y], config=config, mode="valid")
    test_loader = build_dataloader([test_x, test_y], config=config, mode='test')

    # Load Model
    model = load_model(device, config)

    # loss, optimizer
    criterion = loss_load(config=config, device=device)
    optimizer = opt_load(config=config, model=model)

    # callbacks
    logger = CSVLogger(
        path=config['TRAIN']['LOGGING_SAVE_PATH'], sep=config['TRAIN']['LOGGING_SEP']
    )
    checkpoint = SaveCheckPoint(path=config['TRAIN']['MODEL_SAVE_PATH'],
                                model_name=config['MODEL']['NAME'],
                                opt_name=config['MODEL']['OPTIMIZER'],
                                lr=config['MODEL']['LR'],
                                )

    early_stopping = EarlyStopping(
        patience=config['TRAIN']['EARLYSTOP_PATIENT'], verbose=True
    )

    train = Trainer(config=config,
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    test_loader=test_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    logger=logger,
                    checkpoint=checkpoint,
                    early_stopping=early_stopping)
    train.train()


if __name__ == '__main__':
    main()



