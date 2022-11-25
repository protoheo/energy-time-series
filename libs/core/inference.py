import torch
import pandas as pd
from libs import train_utils
from libs.common.project_paths import GetPaths


class Inference:
    def __init__(self,
                 config=None,
                 model_list=None,
                 test_loader=None,
                 device=None,
                 tokenizer=None):
        self.config = config
        self.model_list = model_list
        self.test_loader = test_loader
        self.device = device
        self.tokenizer = tokenizer

    def inference(self, cls_label_map):
        all_preds = train_utils.share_loop(model=self.model_list,
                                           data_loader=self.test_loader,
                                           mode='ensemble')
        submit_df = pd.read_csv(GetPaths.get_data_folder('sample_submission.csv'))
        label = torch.cat(all_preds)
        submit_df['label'] = label.cpu().numpy()
        submit_df['label'] = submit_df['label'].apply(lambda x: cls_label_map[x])
        submit_df.to_csv('./submit.csv', index=False)

    @staticmethod
    def cls_name_maps(cls_name):
        cls_name_maps = {}
        for k, v in enumerate(cls_name):
            cls_name_maps[k] = v
        return cls_name_maps

    def __legacy(self):
        pass
