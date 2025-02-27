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
         [0.603, 0.598, 0.589, 0.590, 0.614, 0.602, 0.589, 0.568, 0.573, 0.577, 0.574, 0.585, 0.593, 0.594, 0.597],
         [0.747, 0.737, 0.724, 0.728, 0.762, 0.740, 0.726, 0.700, 0.706, 0.706, 0.711, 0.720, 0.729, 0.725, 0.729]
        ),
        ('temperature',
         [1.084, 1.078, 1.038, 0.981, 0.946, 0.948, 0.912, 0.870, 0.860, 0.868, 0.900, 0.875, 0.885, 0.885, 0.928],
         [1.382, 1.401, 1.339, 1.272, 1.223, 1.228, 1.164, 1.111, 1.096, 1.123, 1.154, 1.130, 1.165, 1.144, 1.192]
        ),
        ('windspeed',
         [1.02, 1.02, 0.93, 0.89, 0.84, 0.82, 0.81, 0.77, 0.76, 0.77, 0.78, 0.78, 0.80, 0.79, 0.83],
         [1.321, 1.341, 1.241, 1.171, 1.101, 1.091, 1.061, 1.011, 0.991, 1.021, 1.031, 1.031, 1.061, 1.041, 1.091]
        ),
        ('water',
         [0.322, 0.200, 0.173, 0.159, 0.211, 0.211, 0.159, 0.153, 0.158, 0.144, 0.164, 0.154, 0.197, 0.155, 0.195],
         [0.405, 0.263, 0.245, 0.207, 0.271, 0.271, 0.212, 0.205, 0.213, 0.216, 0.221, 0.213, 0.241, 0.207, 0.247]
        ),
    ]
    font = {"family": "SimSun", "size": 16}
    for x in PATCH_LENGTH_LIST:
        plt.axvline(x, linestyle='--', color='#BFBFBF')
    plt.xticks(PATCH_LENGTH_LIST)
    plt.xlabel("Patch长度", fontdict=font)
    plt.plot(PATCH_LENGTH_LIST, results[3][1], color='#FF4040', marker='o', label='MAE', linewidth=1.5)
    plt.plot(PATCH_LENGTH_LIST, results[3][2], color='#FF7F50', marker='*', label='RMSE', linewidth=1.5)
    legend = plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15))
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_edgecolor('none')
    plt.savefig(f'{results[3][0]}_patch_length.svg', dpi=600)    

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