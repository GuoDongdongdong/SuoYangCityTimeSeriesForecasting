'''
    chapter 4 experiment 4.
    patch length args experiment.
'''

import os
import configparser
import subprocess
from configparser import ConfigParser
from matplotlib import pyplot as plt

CONFIG_FILE_NAME = 'config.ini'
CONFIG_FILE_DIR  = '.'
TEMP_CONFIG_FILE_NAME = 'temp_config.ini'
TEMP_FILE_DIR = './temp'

PATCH_LENGTH_LIST = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180]

def common_args_define(config:ConfigParser) -> None:
    config['CommonArgs']['dataset_file_dir'] = 'str:TIEGAN_dataset'
    config['CommonArgs']['dataset_file_name'] = 'str:humidity.csv'
    config['CommonArgs']['model'] = 'str:MPformer'
    config['CommonArgs']['train_test'] = 'bool:True'
    config['CommonArgs']['model_save_path'] = 'str:None'
    config['CommonArgs']['targets'] = 'list:humidity'
    config['CommonArgs']['date_frequence'] = 'str:h'
    config['CommonArgs']['timeenc'] = 'str:timeF'
    config['CommonArgs']['lookback_length'] = 'int:192'
    config['CommonArgs']['predict_length'] = 'int:192'
    config['CommonArgs']['label_length'] = 'int:96'
    config['CommonArgs']['train_ratio'] = 'float:0.7'
    config['CommonArgs']['vali_ratio'] = 'float:0.1'
    config['CommonArgs']['random_seed'] = 'int:202221543'
    config['CommonArgs']['use_gpu'] = 'bool:True'
    config['CommonArgs']['use_multi_gpu'] = 'bool:False'
    config['CommonArgs']['gpu_id'] = 'str:0'
    config['CommonArgs']['use_amp'] = 'bool:False'
    config['CommonArgs']['batch_size'] = 'int:32'
    config['CommonArgs']['lr'] = 'float:1e-4'
    config['CommonArgs']['epochs'] = 'int:300'
    config['CommonArgs']['patience'] = 'int:5'
    config['CommonArgs']['num_workers'] = 'int:8'
    config['CommonArgs']['time_complexity'] = 'bool:False'

def run():
    args = ['-config_file_dir',
            TEMP_FILE_DIR, 
            '-config_file_name', 
            TEMP_CONFIG_FILE_NAME,
            ]
    result = subprocess.run(['python', 'run.py'] + args)

def shutdown():
    os.system("/usr/bin/shutdown")

def plot():
    results = [
        ('humidity',
         [1.20, 1.20, 1.11, 1.07, 1.02, 1.00, 0.99, 1.01, 0.97, 0.98, 0.96, 0.98, 0.95, 0.94, 0.95],
         [1.50, 1.52, 1.42, 1.35, 1.28, 1.27, 1.24, 1.27, 1.22, 1.24, 1.21, 1.21, 1.20, 1.17, 1.19]
        ),
        ('temperature',
         [1.20, 1.20, 1.11, 1.07, 1.02, 1.00, 0.99, 1.01, 0.97, 0.98, 0.96, 0.98, 0.95, 0.94, 0.95],
         [1.50, 1.52, 1.42, 1.35, 1.28, 1.27, 1.24, 1.27, 1.22, 1.24, 1.21, 1.21, 1.20, 1.17, 1.19]
        ),
        ('windspeed',
         [1.20, 1.20, 1.11, 1.07, 1.02, 1.00, 0.99, 1.01, 0.97, 0.98, 0.96, 0.98, 0.95, 0.94, 0.95],
         [1.50, 1.52, 1.42, 1.35, 1.28, 1.27, 1.24, 1.27, 1.22, 1.24, 1.21, 1.21, 1.20, 1.17, 1.19]
        ),
        ('water',
         [1.20, 1.20, 1.11, 1.07, 1.02, 1.00, 0.99, 1.01, 0.97, 0.98, 0.96, 0.98, 0.95, 0.94, 0.95],
         [1.50, 1.52, 1.42, 1.35, 1.28, 1.27, 1.24, 1.27, 1.22, 1.24, 1.21, 1.21, 1.20, 1.17, 1.19]
        ),
    ]
    def _plot_one_dataset(mae_list, rmse_list:list) -> None:
        font = {"family": "SimSun", "size": 16}
        for x in PATCH_LENGTH_LIST:
            plt.axvline(x, linestyle='--', color='#BFBFBF')
        plt.xticks(PATCH_LENGTH_LIST)
        plt.xlabel("Patch长度", fontdict=font)
        plt.plot(PATCH_LENGTH_LIST, mae_list, color='#FF4040', marker='o', label='MAE', linewidth=1.5)
        plt.plot(PATCH_LENGTH_LIST, rmse_list, color='#FF7F50', marker='*', label='RMSE', linewidth=1.5)
        legend = plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15))
        legend.get_frame().set_linewidth(0)
        legend.get_frame().set_edgecolor('none')
        plt.savefig(f'patch_length.svg', dpi=600)
    for result in results:
        file_name, mae_list, rmse_list = result
        _plot_one_dataset(mae_list, rmse_list)

def main():
    config_file_path = os.path.join(CONFIG_FILE_DIR, CONFIG_FILE_NAME)
    config = configparser.ConfigParser()
    config.read(config_file_path)
    os.makedirs(TEMP_FILE_DIR, exist_ok=True)
    for patch_length in PATCH_LENGTH_LIST:
        common_args_define(config)
        config['MPformer']['patch_length'] = f'int:{patch_length}'
        with open(os.path.join(TEMP_FILE_DIR, TEMP_CONFIG_FILE_NAME), 'w') as f:
            config.write(f)
        run()
    os.system(f'rm -rf {os.path.join(TEMP_FILE_DIR, TEMP_CONFIG_FILE_NAME)}')
    shutdown()

if __name__ == '__main__':
    plot()