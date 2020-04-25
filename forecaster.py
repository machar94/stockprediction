import torch
from torch import nn

class Forecaster(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers=2, dropout=0.5):
        super(Forecaster, self).__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=dropout,
        )

        # FC Layer
        self.FC = nn.Linear(self.n_hidden, 1)

    def forward(self, x):
        # (batch, seq, features) -> (seq, batch, features)
        lstm_out, self.hidden = self.lstm(torch.transpose(x, 0, 1))

        # lstm_out[-1] is (batch, hidden)
        y_pred = self.FC(lstm_out[-1]).sigmoid()

        return y_pred