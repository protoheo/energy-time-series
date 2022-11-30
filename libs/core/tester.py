from libs import train_utils
import pandas as pd


class Tester:
    def __init__(self,
                 config=None,
                 model=None,
                 test_loader=None,
                 device=None,
                 scaler=None):

        self.config = config
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.scaler = scaler

    def test(self):
        cfg = self.config
        ret_list = []

        ret_list = train_utils.share_loop(
            epoch=0,
            model=self.model,
            data_loader=self.test_loader,
            mode="test"
        )

        # TBD: list에 담기
        # list_train_loss.extend(train_total_loss)
        # list_valid_loss.extend(valid_total_loss)
        # list_avg_train_loss.append(train_avg_loss)
        # list_avg_valid_loss.append(valid_avg_loss)

        # train_utils.print_result(result=results)  # 결과출력

        df = pd.DataFrame(ret_list, columns=['mae', 'mse', 'rmse', 'mape'])
        print(df.describe().loc['mean'])
