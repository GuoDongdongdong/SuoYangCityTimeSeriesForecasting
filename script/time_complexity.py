'''
    chapter 4 experiment 3.
    model time complexity.
'''

import os
import configparser
import subprocess
from configparser import ConfigParser

CONFIG_FILE_NAME = 'config.ini'
CONFIG_FILE_DIR  = '.'
TEMP_CONFIG_FILE_NAME = 'temp_config.ini'
TEMP_FILE_DIR = './temp'

MODEL_LIST = [
    'MPformer',
    'PatchTST',
    'FEDformer',
    'Autoformer',
    'Informer',
    'Transformer',
]
DATASETS = [
    ('humidity.csv', 'humidity'),
    ('temperature.csv', 'temperature'),
    ('windspeed.csv', 'windspeed'),
    ('water.csv', 'water'),
]
PREDICT_LENGTH = [
    48, 64, 80, 96, 128, 160, 192, 256, 320, 384
]

def common_args_define(config:ConfigParser) -> None:
    config['CommonArgs']['dataset_file_dir'] = 'str:TIEGAN_dataset'
    config['CommonArgs']['dataset_file_name'] = 'str:None'
    config['CommonArgs']['model'] = 'str:None'
    config['CommonArgs']['train_test'] = 'bool:True'
    config['CommonArgs']['model_save_path'] = 'str:None'
    config['CommonArgs']['targets'] = 'list:None'
    config['CommonArgs']['date_frequence'] = 'str:h'
    config['CommonArgs']['timeenc'] = 'str:timeF'
    config['CommonArgs']['lookback_length'] = 'int:48'
    config['CommonArgs']['predict_length'] = 'str:None'
    config['CommonArgs']['train_ratio'] = 'float:0.7'
    config['CommonArgs']['vali_ratio'] = 'float:0.1'
    config['CommonArgs']['random_seed'] = 'int:202221543'
    config['CommonArgs']['use_gpu'] = 'bool:True'
    config['CommonArgs']['use_multi_gpu'] = 'bool:False'
    config['CommonArgs']['gpu_id'] = 'str:0'
    config['CommonArgs']['use_amp'] = 'bool:False'
    config['CommonArgs']['batch_size'] = 'int:32'
    config['CommonArgs']['lr'] = 'float:1e-3'
    config['CommonArgs']['epochs'] = 'int:300'
    config['CommonArgs']['patience'] = 'int:5'
    config['CommonArgs']['num_workers'] = 'int:8'

def mpformer_ps12(config:ConfigParser) -> None:
    config['MPformer']['patch_stride'] = 'int:12'

def mpformer_ps18(config:ConfigParser) -> None:
    config['MPformer']['patch_stride'] = 'int:18'

def mpformer_ps24(config:ConfigParser) -> None:
    config['MPformer']['patch_stride'] = 'int:24'


def run():
    args = ['-config_file_dir',
            TEMP_FILE_DIR, 
            '-config_file_name', 
            TEMP_CONFIG_FILE_NAME,
            ]
    result = subprocess.run(['python', 'run.py'] + args)

def shutdown():
    os.system("/usr/bin/shutdown")

if __name__ == '__main__':
    config_file_path = os.path.join(CONFIG_FILE_DIR, CONFIG_FILE_NAME)
    config = configparser.ConfigParser()
    config.read(config_file_path)
    os.makedirs(TEMP_FILE_DIR, exist_ok=True)
    mpformer_list = [mpformer_ps12, mpformer_ps18, mpformer_ps24]
    for dataset_name, targets_name in DATASETS:
        for predict_length in PREDICT_LENGTH:
            for model_name in MODEL_LIST:
                common_args_define(config)
                config['CommonArgs']['dataset_file_name'] = f'str:{dataset_name}'
                config['CommonArgs']['targets'] = f'list:{targets_name}'
                config['CommonArgs']['predict_length'] = f'int:{predict_length}'
                config['CommonArgs']['model'] = f'str:{model_name}'
                if model_name == 'MPformer':
                    for mpformer in mpformer_list:
                        mpformer(config)
                        with open(os.path.join(TEMP_FILE_DIR, TEMP_CONFIG_FILE_NAME), 'w') as f:
                            config.write(f)
                        run()
                else:
                    with open(os.path.join(TEMP_FILE_DIR, TEMP_CONFIG_FILE_NAME), 'w') as f:
                        config.write(f)
                    run()
            os.system(f'rm -rf {os.path.join(TEMP_FILE_DIR, TEMP_CONFIG_FILE_NAME)}')
    shutdown()
