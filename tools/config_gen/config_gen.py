import os
import sys
import numpy as np
import random
import pycls.core.config as config

SPACE='darts'
DATASET='cifar10'
SOURCE_TASK='cls'
TARGET_TASK='psd250'

file_list=np.load(f'configs/sample_based/{SPACE}/{DATASET}/selected_files.npy',allow_pickle=True)
input_dir=f'configs/sample_based/{SPACE}/{DATASET}/{SOURCE_TASK}/'
output_dir=f'configs/sample_based/{SPACE}/{DATASET}/{TARGET_TASK}/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for f in file_list:
    f_name=f.split('.')[0]
    source_config=f'{input_dir}/{f}'
    target_config=f'{output_dir}/{f}'
    config.load_cfg(input_dir,f)
    config.assert_and_infer_cfg()
    config.cfg.TRAIN.PSD_LABEL_SPLIT = 5000
    config.cfg.TRAIN.PSD_UNLABEL_BATCH_SIZE = 100
    config.cfg.TRAIN.PSD_LABEL_BATCH_SIZE = 20
    config.cfg.TRAIN.PSD_THRESHOLD = 0.95
    config.cfg.LOG_PERIOD=100
    config.cfg.TASK='psd'
    config.cfg.OPTIM.MAX_EPOCH = 50
    config.cfg.OPTIM.BASE_LR=0.1
    config.cfg.OPTIM.MOMENTUM=0.9
    config.dump_cfg_to_file(target_config)
    print(source_config,target_config)
