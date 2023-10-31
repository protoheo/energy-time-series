import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, device='cuda'):
        super(EncoderLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)

    def forward(self, inputs, hidden):
        # embedded = self.embedding(inputs)
        # Pass the embedded word vectors into LSTM and return all outputs
        output, hidden = self.lstm(inputs, hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, self.hidden_size, device=self.device),
                torch.zeros(self.n_layers, self.hidden_size, device=self.device))


class BahdanauDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, drop_prob=0.1):
        super(BahdanauDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.squeeze()
        # Embed input words
        embedded = self.embedding(inputs.long()).view(1, -1)
        embedded = self.dropout(embedded)

        # Calculating Alignment Scores
        x = torch.tanh(self.fc_hidden(hidden[0]) + self.fc_encoder(encoder_outputs))
        alignment_scores = x.bmm(self.weight.unsqueeze(2))

        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores.view(1, -1), dim=1)

        # Multiplying the Attention weights with encoder outputs to get the context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(0),
                                   encoder_outputs.unsqueeze(0))

        # Concatenating context vector with embedded input word
        output = torch.cat((embedded, context_vector[0]), 1).unsqueeze(0)
        # Passing the concatenated vector as input to the LSTM cell
        output, hidden = self.lstm(output, hidden)
        # Passing the LSTM output through a Linear layer acting as a classifier

        return output, hidden, attn_weights


class DualStage(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, layers):
        super(DualStage, self).__init__()

        self.encoder = EncoderLSTM(input_dim, hidden_size, layers)
        self.decoder = BahdanauDecoder(hidden_size, output_dim, layers)

    def forward(self, x, h):
        print(x.size())
        encoder_outputs, h = self.encoder(x, h)
        x = self.decoder(x, h, encoder_outputs)
        print(x.size())
        x = self.sigmoid(x)
        x = self.relu(x)

        return x

    def init_hidden(self, batch_size):
        return self.encoder.init_hidden(batch_size)

