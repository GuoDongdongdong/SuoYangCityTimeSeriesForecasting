'''
    chapter 4 experiment 5.
    patch stride args experiment.
'''

import os
import configparser
import subprocess
from configparser import ConfigParser

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
    main()