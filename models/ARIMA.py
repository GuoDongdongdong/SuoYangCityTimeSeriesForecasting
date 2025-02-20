import torch
import torch.nn as nn
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import DataLoader

from utils.datasets import SuoYangCityDataset
from models.BaseForecastModel import BaseForecastModel
from utils.ExperimentArgs import ExperimentArgs


class Model(BaseForecastModel):
    def __init__(self, exp_args:ExperimentArgs):
        super().__init__()
        self.predict_len = exp_args['predict_length']

    def evaluate(self, batch:dict, training:bool) -> torch.Tensor:
        raise NotImplementedError('Statistical Model do not need to trian!')


    def forecast(self, test_dataset:SuoYangCityDataset, test_dataloader:DataLoader) -> None:
        x = test_dataset.unnorm_observed_data.copy()
        L, D = x.shape
        output = []
        for dim in range(D):
            # TODO : ARIMA parameters are set to (1, 0, 1) for now, maybe we can find better parameters.
            model = ARIMA(x[:, dim], order=(1, 0, 1)).fit()
            output.append(model.forecast(self.predict_len))
            
        
        output = np.stack(output)
        output = torch.tensor(output)
        output = output.unsqueeze(dim=2)
        test_dataset.save_result(output)
