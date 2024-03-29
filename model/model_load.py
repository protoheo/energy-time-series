import torch
import torch.nn as nn
import os
import torch.nn.functional as F

from model.autoencoder import AutoEncForecast
from model.dula_stage import DualStage


class LSTMBase(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_size, output_dim, layers):
        super(LSTMBase, self).__init__()
        self.output_dim = output_dim
        self.layers = layers
        self.seq_len = input_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_dim,
                            hidden_size=hidden_size,
                            num_layers=layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # 예측을 위한 함수
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_size),
            torch.zeros(self.layers, self.seq_len, self.hidden_size)
        )

    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = self.relu(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, layers):
        super(BiLSTM, self).__init__()
        self.output_dim = output_dim
        self.layers = layers
        self.seq_len = input_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_dim,
                            hidden_size=hidden_size,
                            num_layers=layers,
                            batch_first=True,
                            bidirectional=True
                            )

        self.fc = nn.Linear(hidden_size, output_dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # 예측을 위한 함수
    def init_hidden(self):
        return (
            torch.zeros(self.layers, self.seq_len, self.hidden_size),
            torch.zeros(self.layers, self.seq_len, self.hidden_size)
        )

    def forward(self, x):
        x, state = self.lstm(x)
        x = self.attention_net(x, state)
        # x = self.attention_net(x, x.transpose(0, 1)[-1])
        x = self.fc(x)

        x = self.sigmoid(x)
        x = self.relu(x)
        return x


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


def load_model(device, config, ensemble=False):
    name = config['MODEL']['NAME']
    if ensemble:
        input_dim = len(config['DATA']['ENSEMBLE_1'])
    else:
        input_dim = len(config['DATA']['X_COLS'])
    hidden_dim = input_dim    # config['MODEL_PARAM']['HIDDEN_DIM']
    output_dim = len(config['DATA']['Y_TARGET'])
    layers = config['MODEL_PARAM']['LAYERS']

    if name == 'LSTM':
        model = LSTMBase(input_dim, hidden_dim, output_dim, layers)
    elif name == 'LSTM-Attention':
        model = BiLSTM(input_dim, hidden_dim, output_dim, layers)
    elif name == 'LSTM-Attention':
        model = AutoEncForecast(input_dim, hidden_dim, output_dim, layers)
    else:
        model = LSTMBase(input_dim, hidden_dim, output_dim, layers)

    return apply_device(model, device)


def load_scaler():
    import joblib
    scaler_x = 'ckpt/SCALER/x_scaler.pkl'
    scaler_y = 'ckpt/SCALER/y_scaler.pkl'
    sc_x = joblib.load(scaler_x)
    sc_y = joblib.load(scaler_y)

    return sc_x, sc_y


def load_inference(device, config):
    model = load_model(device, config)

    checkpoint_dir = 'ckpt/{}'.format(config['MODEL']['NAME'])
    file_name = os.listdir(checkpoint_dir)[-1]
    print(file_name)
    checkpoint_path = '{}/{}'.format(checkpoint_dir, file_name)

    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt)
    model.eval()

    return model


def inference_ensemble(dir_path, device, config, ensemble=False):
    if ensemble:
        model = load_model(device, config, ensemble=ensemble)
    else:
        model = load_model(device, config)

    checkpoint_dir = 'ckpt/{}/{}'.format(dir_path, config['MODEL']['NAME'])
    file_name = os.listdir(checkpoint_dir)[-1]
    print(file_name)
    checkpoint_path = '{}/{}'.format(checkpoint_dir, file_name)

    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt)
    model.eval()

    return model
