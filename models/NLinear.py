import torch
import torch.nn as nn

from models.BaseForecastModel import BaseForecastModel
from utils.ExperimentArgs import ExperimentArgs
from utils.functions import calc_mse


class Model(BaseForecastModel):
    def __init__(self, exp_args:ExperimentArgs):
        super().__init__()
        self.model = _NLinear(exp_args)
    
    def evaluate(self, batch, training):
        x = batch['observed_data']
        y = self.model.forward(x)
        return calc_mse(x, y)
    
    def forecast(self, batch):
        x = batch['observed_data']
        y = self.model.forward(x)
        return y
    
class _NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, exp_args:ExperimentArgs):
        super(_NLinear, self).__init__()
        self.seq_len = exp_args['lookback_length']
        self.pred_len = exp_args['predict_length']
        self.individual = exp_args['individual']
        self.channels = len(exp_args['targets'])

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, *args):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]