from torch.utils.data import DataLoader
# from dataload.dataset import CustomDataset
from torch.utils.data import TensorDataset
from torch import FloatTensor


def build_dataloader(data_list, config, mode):
    """
    Augmentation을 불러오고 Dataset을 불러와 적용하여 Dataloader로 wrapping하는 함수입니다.

    :args
        df: 데이터 프레임
        cfg: config. 본 함수에서는 batch size를 받습니다.
        mode: 'train, valid, test'의 값들 중 하나를 받습니다.
    :return dataloader
    """

    mode = mode.lower()
    assert mode in ['train', 'valid', 'test'], 'mode의 입력값은 train, valid, test 중 하나여야 합니다.'
    param = True if mode in ['train', 'valid'] else False

    X_tensor = FloatTensor(data_list[0])
    Y_tensor = FloatTensor(data_list[1])
    dataset = TensorDataset(X_tensor, Y_tensor)

    dataloader = DataLoader(dataset=dataset, batch_size=config[f'{mode}'.upper()]['BATCH_SIZE'],
                            shuffle=param, drop_last=False)

    return dataloader


def build_testloader(data_list):

    param = False

    X_tensor = FloatTensor(data_list[0])
    Y_tensor = FloatTensor(data_list[1])
    dataset = TensorDataset(X_tensor, Y_tensor)

    dataloader = DataLoader(dataset=dataset, batch_size=len(X_tensor),
                            shuffle=param, drop_last=False)

    return dataloader