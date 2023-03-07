from libs.common.common import load_config
from libs.common.project_paths import GetPaths
import torch
import numpy as np
import random


def global_setting(config_file='cfg.yaml', config_dir=None):
    if config_dir is None:
        config_path = GetPaths().get_configs_folder(config_file)
    else:
        config_path = config_dir+'/'+config_file

    config = load_config(config_path)
    torch.manual_seed(config["SEED"])
    np.random.seed(config["SEED"])
    random.seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])  # if use multi-GPU

    # 나중에 재현할 때 쓰는 코드
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return config, device
