import os
import torch
from torch import nn

from models.BaseForecastModel import BaseForecastModel
from utils.ExperimentArgs import ExperimentArgs
from utils.functions import calc_mse


class Model(BaseForecastModel):
    def __init__(self, exp_args:ExperimentArgs):
        super().__init__()
        self.model = _LSTM(exp_args)

    def evaluate(self, batch, training):
        x = batch['observed_data']
        y = batch['predict_data']
        predict = self.model.forward(x)
        return calc_mse(y, predict)
    
    def forecast(self, batch):
        x = batch['observed_data']
        predict = self.model.forward(x)
        return predict

class _LSTM(nn.Module):
    def __init__(self, exp_args:ExperimentArgs):
        super(_LSTM, self).__init__()
        self.device = self._get_device(exp_args)
        self.dropout = exp_args['dropout']
        self.num_layers = exp_args['num_layers']
        self.hidden_size = exp_args['hidden_size']
        self.predict_len = exp_args['predict_length']
        self.bidirectional = exp_args['bidirectional']
        self.dimension = exp_args['dimension']

        self.D = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=self.dimension,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=bool(self.bidirectional),
        )
        self.linear = nn.Linear(self.hidden_size, self.predict_len * self.dimension)

    def _get_device(self, exp_args:ExperimentArgs) -> torch.device:
        if exp_args['use_gpu']:
            GPU_id = exp_args['gpu_id']
            os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id
            device = torch.device(f"cuda:{GPU_id}")
        else:
            device = torch.device('cpu')
        return device


    def init_hidden_state(self, batch_size):
        h_0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(self.device)
        return h_0, c_0

    def forward(self, x, *args):
        # [batch_size, lookback_len, D * hidden_size] for last_layer_l_hidden_state
        # [D * num_layers, batch_size, hidden_size] for h_n
        b, l, d = x.shape
        init_hidden_state = self.init_hidden_state(b)
        last_layer_l_hidden_state, (h_n, c_n) = self.lstm(x, init_hidden_state)
        outputs = self.linear(last_layer_l_hidden_state[:, -1, :])
        outputs = outputs.view(b, self.predict_len, d)
        return outputs


class Cell(nn.Module):
    def __init__(self, args, dimension, bidirectional=False):
        super(Cell, self).__init__()

        self.device = args.device
        self.dimension = dimension
        self.num_layers = args.num_layer
        self.hidden_size = args.d_model
        self.dropout = args.dropout
        self.D = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=self.dimension,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=bidirectional,
        )

    def init_hidden_state(self, batch_size):
        h_0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(self.device)
        return h_0, c_0

    def forward(self, x):
        b = x.shape[0]
        init_hidden_state = self.init_hidden_state(b)
        # output [batch_size, lookback_len, D * hidden_size]
        output, (h_n, c_n) = self.lstm(x, init_hidden_state)
        return output
