import pandas as pd
import os
from pathlib import Path

class CSVLogger:
    def __init__(self, path=None, sep=','):
        self.df = self.define_df()
        self.sep = sep
        self.path = path
        self.set_path()

    @staticmethod
    def define_df():
        columns = ["epoch", "train_loss", "valid_loss", "train_acc", "valid_acc"]
        df = pd.DataFrame(columns=columns)
        return df

    def logging(self, results):
        self.df.loc[len(self.df)] = results  # 로깅
        self.df.to_csv(self.path, index=False, sep=self.sep)

    def set_path(self):
        if self.path is None:
            self.path = Path('./log.txt')
        else:
            self.path = Path(self.path)

        if not os.path.isdir(self.path.parent):
            os.makedirs(self.path.parent, exist_ok=True)

        if not len(self.path.suffixes):
            self.path.replace(self.path.with_suffix('.txt'))  # 파일 확장자가 없을 경우, txt를 기본 확장자로 지정