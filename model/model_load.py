import torch
import torch.nn as nn


def apply_ckpt(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt)
    # print(f'모델을 성공적으로 불러왔습니다.')
    return model


def apply_device(model, device):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 10:
            print("Multi-Device")
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)
    else:
        model = model.to(device)
    return model


class LSTMBase(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(LSTMBase, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = layers

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        # 예측을 위한 함수

    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x)
        return x


class RNNBase(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(RNNBase, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = layers

        self.lstm = nn.RNN(input_dim, hidden_dim, num_layers=layers)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        # 예측을 위한 함수

    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x)
        return x