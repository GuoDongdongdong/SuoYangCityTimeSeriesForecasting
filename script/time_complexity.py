'''
    chapter 4 experiment 3.
    model time complexity.
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

MODEL_LIST = [
    'MPformer',
    'PatchTST',
    'FEDformer',
    'Autoformer',
    'Informer',
    'Reformer',
    'Transformer',
]
DATASETS = [
    ('humidity.csv', 'humidity'),
    ('temperature.csv', 'temperature'),
    ('windspeed.csv', 'windspeed'),
    ('water.csv', 'water'),
]
LOOKBACK_LENGTH = [
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
    config['CommonArgs']['lookback_length'] = 'str:None'
    config['CommonArgs']['predict_length'] = 'int:48'
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
    config['CommonArgs']['time_complexity'] = 'bool:True'

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

def plot():
    models = ["Transformer", "MPformer-Ps12", "MPformer-Ps18", "MPformer-Ps24", "PatchTST", "Informer", "Reformer", "Autoformer"]
    lookbacks = [48, 64, 80, 96, 128, 160, 192, 256, 320, 384]
    args = { # Temperature
        "Transformer" : [[5.5,5.41,5.46,5.51,6.12,7.22,8.14,10.01,12.19,14.26], "#BF00BF", "o"],
        "MPformer-Ps12" : [[3.03,3.06,3.03,3.27,3.85,4.22,5.57,8.59,10.85,14.37], "#FF4040", "D"],
        "MPformer-Ps18" : [[2.86,3.04,3.02,3.21,3.31,3.91,4.81,6.56,9.54,12], "#B74848", "D"],
        "MPformer-Ps24" : [[3,2.91,3.04,3.02,3.32,3.7,4.6,6.22,7.43,10.18,], "#FFCFD7", "D"],
        "PatchTST" : [[2.95,3,3.01,2.99,3.32,3.1,3.02,3.05,3.03,3.14,], "#FF7F50", "*"],
        "Informer" : [[8.61,9.52,9.56,9,9.95,10.34,10.34,11.36,12.7,14.32,], "#00BFBF", "<"],
        "Reformer" : [[6.87,7.29,8.09,8.14,9.48,10.85,11.8,14.69,17.32,20.01,], "#000080", "v"],
        "Autoformer" : [[10.07,10.61,10.69,10.36,10.26,10.74,12.49,13.66,16.05,18.37], "#FFD700", "s"],
        "FEDformer" : [[24.69,27.51,33.34,35.08,47.58,54.37,57.09,61.2,66.23,72.69,], "#B93434", ">"]
    }
    # args = { # WindSpeed3s
    #     "Transformer" : [[6.08,6.08,5.82,6.25,6.07,7.08,8.23,9.85,12.44,14.2,], "#BF00BF", "o"],
    #     "MPformer-Ps12" : [[2.99,3.29,3.12,3.2,3.51,4.17,5.66,8.81,10.71,14.15,], "#FF4040", "D"],
    #     "MPformer-Ps18" : [[3.53,3.45,3.72,3.6,3.75,4.17,5.19,6.67,9.94,12.54,], "#B74848", "D"],
    #     "MPformer-Ps24" : [[3.65,3.43,3.49,3.27,3.42,3.81,4.8,6.4,7.74,10.31,], "#FFCFD7", "D"],
    #     "PatchTST" : [[3.04,2.91,2.88,3.13,2.94,2.89,3,2.93,2.96,3.13,], "#FF7F50", "*"],
    #     "Informer" : [[8.52,8.66,8.78,9.02,9.46,9.85,11.03,12.14,12.5,13.94,], "#00BFBF", "<"],
    #     "Reformer" : [[6.4,7.24,7.8,8.28,9.29,10.82,11.93,14.73,17.38,20.24,], "#000080", "v"],
    #     "Autoformer" : [[10.29,10.34,10.37,10.39,10.58,11.07,12.19,14.08,16.36,18.19,], "#FFD700", "s"],
    # }
    for x in lookbacks :
        plt.axvline(x, linestyle='--', color='#BFBFBF')
    for model in models:
        plt.plot(lookbacks, args[model][0], color=args[model][1], marker=args[model][2], label=model, linewidth=1.5)
    plt.xticks(lookbacks)
    plt.xlabel("lookback length")
    plt.ylabel("second")
    legend = plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15))
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_edgecolor('none')
    plt.show()

if __name__ == '__main__':
    config_file_path = os.path.join(CONFIG_FILE_DIR, CONFIG_FILE_NAME)
    config = configparser.ConfigParser()
    config.read(config_file_path)
    os.makedirs(TEMP_FILE_DIR, exist_ok=True)
    mpformer_list = [mpformer_ps12, mpformer_ps18, mpformer_ps24]
    for dataset_name, targets_name in DATASETS:
        for lookback_length in LOOKBACK_LENGTH:
            for model_name in MODEL_LIST:
                common_args_define(config)
                config['CommonArgs']['dataset_file_name'] = f'str:{dataset_name}'
                config['CommonArgs']['targets'] = f'list:{targets_name}'
                config['CommonArgs']['lookback_length'] = f'int:{lookback_length}'
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
