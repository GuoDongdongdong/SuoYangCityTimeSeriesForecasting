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
    config['CommonArgs']['label_length'] = 'str:None'
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
    humidity = {
        "Transformer" : [[4.73,5.01,5.78,6.10,7.03,8.47,10.16,12.78,16.02,19.04], "#BF00BF", "o"],
        "MPformer-Ps12" : [[2.62,2.92,2.83,3.13,3.60,4.62,5.82,10.21,14.52,19.32], "#FF4040", "D"],
        "MPformer-Ps18" : [[2.81,2.90,2.90,2.93,3.27,3.93,4.54,7.30,10.77,14.74], "#B74848", "D"],
        "MPformer-Ps24" : [[2.87,2.81,2.86,3.10,3.12,3.60,4.09,6.48,8.62,11.12], "#FFCFD7", "D"],
        "PatchTST" : [[2.63,3.07,2.68,3.18,2.87,2.76,2.95,2.84,3.09,2.98,], "#FF7F50", "*"],
        "Informer" : [[7.69,7.77,8.09,8.77,9.51,10.51,11.93,14.12,16.55,18.77,], "#00BFBF", "<"],
        "Reformer" : [[7.68,8.37,9.28,10.34,12.19,14.11,15.80,19.60,23.19,26.93,], "#000080", "v"],
        "Autoformer" : [[7.62,7.79,8.50,9.37,10.58,12.51,14.00,17.38,20.54,23.43], "#FFD700", "s"],
        # "FEDformer" : [[24.69,27.51,33.34,35.08,47.58,54.37,57.09,61.2,66.23,72.69,], "#B93434", ">"]
    }
    temperature = {
        "Transformer" : [[5.29,5.56,5.91,6.48,7.26,8.81,10.58,13.14,16.59,19.57], "#BF00BF", "o"],
        "MPformer-Ps12" : [[3.08,2.94,3.16,3.29,3.84,4.93,6.04,10.59,15.19,19.68], "#FF4040", "D"],
        "MPformer-Ps18" : [[3.04,3.19,3.15,3.29,3.61,4.22,4.81,7.61,11.26,15.03], "#B74848", "D"],
        "MPformer-Ps24" : [[3.00,3.29,3.15,3.38,3.61,3.96,4.47,6.73,8.70,11.61], "#FFCFD7", "D"],
        "PatchTST" : [[3.00,2.92,3.02,2.97,3.13,3.19,3.06,3.22,3.23,3.28,], "#FF7F50", "*"],
        "Informer" : [[8.02,8.39,9.25,9.41,10.33,11.48,12.55,14.99,17.45,19.67,], "#00BFBF", "<"],
        "Reformer" : [[8.19,9.09,9.96,11.11,13.01,15.07,16.84,20.93,24.78,28.76,], "#000080", "v"],
        "Autoformer" : [[8.15,8.95,9.44,10.22,11.47,13.59,15.32,18.50,21.88,25.11], "#FFD700", "s"],
        # "FEDformer" : [[24.69,27.51,33.34,35.08,47.58,54.37,57.09,61.2,66.23,72.69,], "#B93434", ">"]
    }
    windspeed = {
        "Transformer" : [[3.87,4.01,4.39,4.79,5.48,6.57,7.79,9.70,12.12,14.38], "#BF00BF", "o"],
        "MPformer-Ps12" : [[2.22,2.28,2.35,2.41,2.82,3.68,4.51,7.84,11.04,11.56], "#FF4040", "D"],
        "MPformer-Ps18" : [[2.42,2.42,2.31,2.35,2.62,3.14,3.56,5.55,8.24,11.10], "#B74848", "D"],
        "MPformer-Ps24" : [[2.21,2.26,2.39,2.37,2.51,2.76,3.18,4.94,6.39,8.39], "#FFCFD7", "D"],
        "PatchTST" : [[2.20,2.21,2.31,2.22,2.16,2.26,2.18,2.20,2.29,2.32,], "#FF7F50", "*"],
        "Informer" : [[6.03,6.07,6.55,6.85,7.53,8.32,9.10,10.88,12.58,14.25,], "#00BFBF", "<"],
        "Reformer" : [[6.00,6.58,7.27,8.12,9.41,10.76,12.10,14.94,17.62,20.41,], "#000080", "v"],
        "Autoformer" : [[5.98,6.08,6.69,7.35,8.20,9.52,10.74,13.20,15.51,17.70], "#FFD700", "s"],
        # "FEDformer" : [[24.69,27.51,33.34,35.08,47.58,54.37,57.09,61.2,66.23,72.69,], "#B93434", ">"]
    }
    water = {
        "Transformer" : [[3.19,3.32,3.77,3.91,4.41,5.26,6.24,7.64,9.51,11.21], "#BF00BF", "o"],
        "MPformer-Ps12" : [[1.82,1.84,2.22,2.26,2.26,2.84,3.62,6.14,8.56,11.23], "#FF4040", "D"],
        "MPformer-Ps18" : [[1.84,1.90,2.12,2.31,2.13,2.46,3.10,4.45,6.42,8.76], "#B74848", "D"],
        "MPformer-Ps24" : [[1.89,1.83,2.38,2.20,2.11,2.26,2.60,3.94,4.96,6.53,], "#FFCFD7", "D"],
        "PatchTST" : [[1.77,1.87,1.95,2.01,1.81,1.89,1.84,1.82,1.91,1.94,], "#FF7F50", "*"],
        "Informer" : [[4.69,4.82,6.07,5.60,6.01,6.54,7.45,8.58,9.91,11.14,], "#00BFBF", "<"],
        "Reformer" : [[4.82,5.29,5.94,6.46,7.50,8.64,9.62,11.76,13.77,15.84,], "#000080", "v"],
        "Autoformer" : [[4.77,4.89,5.81,6.19,6.51,7.70,8.72,10.38,12.19,14.06], "#FFD700", "s"],
        # "FEDformer" : [[24.69,27.51,33.34,35.08,47.58,54.37,57.09,61.2,66.23,72.69,], "#B93434", ">"]
    }
    arg = water
    font = {"family": "SimSun", "size": 16}
    for x in lookbacks:
        plt.axvline(x, linestyle='--', color='#BFBFBF')
    plt.xticks(lookbacks)
    plt.xlabel("输入长度", fontdict=font)
    plt.ylabel("训练阶段平均每轮时间开销", fontdict=font)
    for model in models:
        plt.plot(lookbacks, arg[model][0], color=arg[model][1], marker=arg[model][2], label=model, linewidth=1.5)
    legend = plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15))
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_edgecolor('none')
    plt.savefig(f'water.svg', dpi=600)

def main():
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
                config['CommonArgs']['label_length'] = f'int:{lookback_length // 2}'
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

if __name__ == '__main__':
    plot()