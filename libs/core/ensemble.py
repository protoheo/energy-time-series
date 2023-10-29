from libs import train_utils
import pandas as pd


class Ensemble:
    def __init__(self,
                 config=None,
                 model=None,
                 test_loader=None,
                 device=None,
                 scaler=None):

        self.config1 = config[0]
        self.config2 = config[1]
        try:
            self.config3 = config[2]
            self.config4 = config[3]
        except:
            pass

        self.model1 = model[0]
        self.model2 = model[1]

        self.test_loader = test_loader
        self.device = device

    def test(self):

        ret_list = train_utils.ensemble_loop(
            epoch=0,
            model=[self.model1, self.model2],
            data_loader=self.test_loader,
            mode="ensemble"
        )
        print(ret_list)

    def merge_test(self):
        ret_list = train_utils.ensemble_merge(
            epoch=0,
            config=[self.config1, self.config2],
            model=[self.model1, self.model2],
            data_loader=self.test_loader,
            mode="ensemble"
        )
        print(ret_list)

    def total_test(self, stage_model):

        st_model1 = stage_model[0]
        st_model2 = stage_model[1]
        ret_list = train_utils.ensemble_total(
            epoch=0,
            config=[self.config1, self.config2, self.config3, self.config4],
            model=[self.model1, self.model2, st_model1, st_model2],
            data_loader=self.test_loader,
            mode="ensemble"
        )
        print(ret_list)


def load_scaler_ensemble(st1_dir, st2_dir):
    import joblib
    scaler_x = 'ckpt/{}/SCALER/x_scaler.pkl'.format(st1_dir)
    scaler_y = 'ckpt/{}/SCALER/y_scaler.pkl'.format(st2_dir)
    sc_x = joblib.load(scaler_x)
    sc_y = joblib.load(scaler_y)

    return sc_x, sc_y
