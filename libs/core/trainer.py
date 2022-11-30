import torch
import pandas as pd
from libs import train_utils
from libs.common.project_paths import GetPaths


class Trainer:
    def __init__(self,
                 config=None,
                 model=None,
                 train_loader=None,
                 valid_loader=None,
                 test_loader=None,
                 criterion=None,
                 optimizer=None,
                 device=None,
                 logger=None,
                 checkpoint=None,
                 early_stopping=None):

        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping

    def train(self):
        cfg = self.config

        for epoch in range(cfg['TRAIN']['EPOCHS']):
            # train

            train_avg_loss, train_total_loss, train_avg_acc, train_total_acc = train_utils.share_loop(
                epoch,
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                mode="train"
            )

            # validation
            valid_avg_loss, valid_total_loss, valid_avg_acc, valid_total_acc = train_utils.share_loop(
                epoch,
                self.model,
                self.valid_loader,
                self.criterion,
                self.optimizer,
                mode="valid"
            )

            # TBD: list에 담기
            # list_train_loss.extend(train_total_loss)
            # list_valid_loss.extend(valid_total_loss)
            # list_avg_train_loss.append(train_avg_loss)
            # list_avg_valid_loss.append(valid_avg_loss)

            results = [epoch, train_avg_loss, valid_avg_loss, train_avg_acc, valid_avg_acc]
            train_utils.print_result(result=results)  # 결과출력
            self.logger.logging(results)  # 로깅
            self.checkpoint(valid_avg_loss, self.model)  # 체크포인트 저장
            self.early_stopping(valid_avg_loss)  # 얼리스탑
            if self.early_stopping.early_stop:
                break
