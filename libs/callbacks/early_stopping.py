import numpy as np


class EarlyStopping:
    """주어진 patience 이후로 score가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=5, verbose=True, delta=0, mode='min'):
        """
        Args:
            patience (int): score가 개선된 후 기다리는 기간
                            Default: 5
            verbose (bool): True일 경우 각 score의 개선 사항 메세지 출력
                            Default: True
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            mode (str): 'min', 'max'를 선택하여 score의 개선 방향을 선택
                            Default: min
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.mode = mode.lower()
        self.delta = delta
        self.best_score = np.Inf
        self.early_stop = False

    def __call__(self, score):
        score_ = abs(score)
        if self.mode == 'min':
            improve_condition = self.best_score - score_ > self.delta
        elif self.mode == 'max':
            improve_condition = score_ - self.best_score > self.delta
        else:
            raise Exception('mode의 변수는 min 또는 max여야 합니다.')

        if not improve_condition:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.verbose_print()
        else:
            self.counter = 0
            self.best_score = score_

    def verbose_print(self):
        print(f"score가 {self.counter}회 동안 개선되지 않았습니다. 학습을 중지합니다.")