import pandas as pd


from libs.core.trainer import Trainer
from configs.setting import global_setting
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
    model = load_model(device, config, name='LSTM')

    # dataframes
    data_path = './data/train_data.csv'
    train_df, valid_df = train_utils.split_dataset(data_path)
    test_df = pd.read_csv('./data/test_data.csv')

    # transfor

    # 데이터 로더
    train_loader = build_dataloader(config=config, df=train_df, mode="train")
    valid_loader = build_dataloader(config=config, df=valid_df, mode="valid")
    test_loader = build_dataloader(config=config, df=test_df, mode='test')

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
                    transform=transform,
                    logger=logger,
                    checkpoint=checkpoint,
                    early_stopping=early_stopping)
    train.train()


if __name__ == '__main__':
    # target_dir = 'imgs'
    # make_dataframe(target_dir, platform_sys='Windows')
    main()



