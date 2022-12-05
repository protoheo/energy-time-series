import torch
import torch.nn as nn
import os


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

        self.relu = nn.ReLU()

        # 예측을 위한 함수

    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x)
        x = self.relu(x)
        return x


class RNNBase(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(RNNBase, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = layers

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)
        self.relu = nn.ReLU()

        # 예측을 위한 함수

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
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


def load_model(device, config):
    name = config['MODEL']['NAME']
    input_dim = len(config['DATA']['X_COLS'])
    hidden_dim = input_dim * 2    # config['MODEL_PARAM']['HIDDEN_DIM']
    output_dim = len(config['DATA']['Y_TARGET'])
    layers = config['MODEL_PARAM']['LAYERS']

    if name == 'RNN':
        model = RNNBase(input_dim, hidden_dim, output_dim, layers)
    elif name == 'LSTM':
        model = LSTMBase(input_dim, hidden_dim, output_dim, layers)
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


