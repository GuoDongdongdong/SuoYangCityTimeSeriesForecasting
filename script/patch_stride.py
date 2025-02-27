'''
    chapter 4 experiment 5.
    patch stride args experiment.
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

PATCH_STRIDE_LIST = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]
def common_args_define(config:ConfigParser) -> None:
    config['CommonArgs']['dataset_file_dir'] = 'str:TIEGAN_dataset'
    config['CommonArgs']['dataset_file_name'] = 'str:windspeed.csv'
    config['CommonArgs']['model'] = 'str:MPformer'
    config['CommonArgs']['train_test'] = 'bool:True'
    config['CommonArgs']['model_save_path'] = 'str:None'
    config['CommonArgs']['targets'] = 'list:windspeed'
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
         [0.145, 0.127, 0.138, 0.129, 0.130, 0.121, 0.125, 0.120, 0.119, 0.121, 0.121, 0.122],
         [0.186, 0.166, 0.177, 0.166, 0.169, 0.157, 0.162, 0.155, 0.154, 0.155, 0.156, 0.157]
        ),
        ('temperature',
         [1.01, 0.971, 0.965, 0.969, 0.95, 0.96, 0.94, 0.94, 0.93, 0.93, 0.93, 0.93],
         [1.27, 1.226, 1.222, 1.230, 1.20, 1.24, 1.17, 1.17, 1.16, 1.16, 1.17, 1.15]
        ),
        ('windspeed',
         [1.010, 0.971, 0.965, 0.969, 0.948, 0.960, 0.938, 0.938, 0.930, 0.923, 0.933, 0.926],
         [1.271, 1.226, 1.222, 1.230, 1.199, 1.238, 1.165, 1.168, 1.160, 1.149, 1.166, 1.150]
        ),
        ('water',
         [0.205, 0.199, 0.196, 0.234, 0.203, 0.216, 0.199, 0.209, 0.198, 0.223, 0.198, 0.209],
         [0.278, 0.274, 0.269, 0.317, 0.279, 0.288, 0.273, 0.287, 0.268, 0.298, 0.268, 0.276]
        ),
    ]
    font = {"family": "SimSun", "size": 16}
    for x in PATCH_STRIDE_LIST:
        plt.axvline(x, linestyle='--', color='#BFBFBF')
    plt.xticks(PATCH_STRIDE_LIST)
    plt.xlabel("Patch stride长度", fontdict=font)
    plt.plot(PATCH_STRIDE_LIST, results[1][1], color='#FF4040', marker='o', label='MAE', linewidth=1.5)
    plt.plot(PATCH_STRIDE_LIST, results[1][2], color='#FF7F50', marker='*', label='RMSE', linewidth=1.5)
    legend = plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15))
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_edgecolor('none')
    plt.savefig(f'{results[1][0]}_patch_stride.svg', dpi=600)    

def main():
    config_file_path = os.path.join(CONFIG_FILE_DIR, CONFIG_FILE_NAME)
    config = configparser.ConfigParser()
    config.read(config_file_path)
    os.makedirs(TEMP_FILE_DIR, exist_ok=True)
    for patch_stride in PATCH_STRIDE_LIST:
        common_args_define(config)
        config['MPformer']['patch_length'] = 'int:96'
        config['MPformer']['patch_stride'] = f'int:{patch_stride}'
        with open(os.path.join(TEMP_FILE_DIR, TEMP_CONFIG_FILE_NAME), 'w') as f:
            config.write(f)
        run()
    os.system(f'rm -rf {os.path.join(TEMP_FILE_DIR, TEMP_CONFIG_FILE_NAME)}')
    shutdown()

if __name__ == '__main__':
    plot()