import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils.ExperimentArgs import ExperimentArgs
from utils.logger import logger
from utils.functions import metric
from utils.config import DEFAULT_DATE_COLUMN_NAME, DEFAULT_RESULT_FILE_NAME, NAN_SYMBOL
from utils.TimeFeature import time_features

'''
    Suo Yang City csv file column should be like `date` `targets` `other_column`
    data in csv file will be split into `train` `validate` `test` in order

'''
FLAG_DICT = {
    'train' : 0,
    'validate' : 1,
    'test' : 2
}
class SuoYangCityDataset(Dataset):
    '''
        self.raw_data is orignal data read from csv file.
        self.features is date + targets
        self.date is date column
        self.observed_data is normalized data by train dataset data and set Nan to zero.
        self.observed_mask is observed matrix.
        self.deltas is time gap matrix.
        note that we normalized data before set Nan to zero.
    '''
    def __init__(self, exp_args:ExperimentArgs, flag:str) -> None:
        super().__init__()
        assert flag in ['train', 'validate', 'test'], 'Dataset flag should be [train validate test]'
        self.exp_args = exp_args
        dataset_file_path = os.path.join(self.exp_args['dataset_file_dir'], self.exp_args['dataset_file_name'])
        self.raw_data = pd.read_csv(dataset_file_path)
        self.features = [DEFAULT_DATE_COLUMN_NAME] + self.exp_args['targets']
        self.raw_data = self.raw_data[self.features]
        data = self.raw_data[self.exp_args['targets']]
        self.record_length = len(self.raw_data)
        self.train_dataset_length = int(self.exp_args['train_ratio'] * self.record_length)
        self.validate_dataset_length = int(self.exp_args['vali_ratio'] * self.record_length)
        self.test_dataset_legth = self.record_length - self.train_dataset_length - self.validate_dataset_length
        self.board_l = [0, self.train_dataset_length, self.train_dataset_length + self.validate_dataset_length]
        self.board_r = [self.train_dataset_length, self.train_dataset_length + self.validate_dataset_length, self.record_length]
        self.date = self.raw_data[DEFAULT_DATE_COLUMN_NAME][self.board_l[FLAG_DICT[flag]] : 
        self.board_r[FLAG_DICT[flag]]]
        train_dataset = data[self.board_l[FLAG_DICT['train']] : self.board_r[FLAG_DICT['train']]]
        self.scaler = StandardScaler()
        self.scaler.fit(train_dataset)
        self.unnorm_observed_data:np.ndarray = data[self.board_l[FLAG_DICT[flag]] : self.board_r[FLAG_DICT[flag]]].values
        self.observed_data = self.scaler.transform(self.unnorm_observed_data)
        self.timestamp = pd.to_datetime(self.date.values)
        self.timestamp = time_features(self.timestamp, exp_args['date_frequence'])

    def __getitem__(self, index:int) -> dict:
        x_l, x_r = index, index + self.exp_args['lookback_length']
        y_l, y_r = x_r, x_r + self.exp_args['predict_length']
        res = {
            'observed_data' : self.observed_data[x_l : x_r],
            'predict_data'  : self.observed_data[y_l : y_r],
            'observed_date' : self.timestamp[x_l : x_r],
            'predict_date'  : self.timestamp[y_l : y_r],
        }
        return res

    def __len__(self) -> int:
        return len(self.observed_data) - self.exp_args['lookback_length'] - self.exp_args['predict_length'] + 1

    def _inverse_ndarry(self, data:np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)

    def _inverse_tensor(self, data:torch.Tensor) -> torch.Tensor:
        res = self.scaler.inverse_transform(data)
        return torch.from_numpy(res)

    def inverse(self, data:torch.Tensor|np.ndarray) -> torch.Tensor | np.ndarray:
        data_shape = data.shape
        D = data_shape[-1]
        data = data.reshape(-1, D)
        if isinstance(data, torch.Tensor):
            data = self._inverse_tensor(data)
        elif isinstance(data, np.ndarray):
            data = self._inverse_ndarry(data)
        data = data.reshape(data_shape)
        return data

    def save_result(self, preidct_data:np.ndarray|torch.Tensor) -> None:
        # TODO: we have not impute all test dataset, need to fix it.
        assert preidct_data.ndim == 3, f'shape should be like [B, L, D], but got{preidct_data.shape}!'
        if isinstance(preidct_data, torch.Tensor):
            preidct_data = preidct_data.numpy()
        B, L, D = preidct_data.shape
        predict_data_length = B * L
        preidct_data = preidct_data.reshape(predict_data_length, D)
        observed_data = self.unnorm_observed_data.copy()
        observed_data = observed_data[: predict_data_length]
        mae, mse, rmse, mape, mspe = metric(preidct_data, observed_data)
        logger.info(f"mae: {mae}")
        logger.info(f"mse : {mse}")
        logger.info(f"rmse : {rmse}")
        logger.info(f"mape : {mape}")
        logger.info(f"mspe : {mspe}")
        df = pd.DataFrame()
        padding_length = self.test_dataset_legth - predict_data_length
        padding_data = np.full((padding_length, D), float('nan'))
        preidct_data = np.concatenate((preidct_data, padding_data))
        observed_data = self.unnorm_observed_data.copy()
        df['date'] = self.date
        df[self.exp_args['targets']] = observed_data
        df[[target + '_forecast' for target in self.exp_args['targets']]] = preidct_data
        save_path = os.path.join(self.exp_args.get_save_path(), DEFAULT_RESULT_FILE_NAME)
        df.to_csv(save_path, index=False, float_format='%.2f', na_rep=NAN_SYMBOL)

def data_provider(exp_args:ExperimentArgs, flag:str) -> tuple[SuoYangCityDataset, DataLoader]:
    assert flag in ['train', 'validate', 'test'], f'Dataset flag should be [train validate test], but got {flag}!'
    dataset = SuoYangCityDataset(exp_args, flag)
    shuffle = True if flag == 'train' else False
    sampler = None if flag == 'train' else iter(range(0, len(dataset), exp_args['predict_length']))
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=exp_args['batch_size'], 
                            shuffle=shuffle, 
                            num_workers=exp_args['num_workers'], 
                            sampler=sampler
                            )
    return dataset, dataloader