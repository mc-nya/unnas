SPACE='darts'
DATASET='cifar10'
TASK='psd5000'
input_dir=f'configs/sample_based/{SPACE}/{DATASET}/{TASK}/'
output_dir=f'results/sample_based/{SPACE}/{DATASET}/{TASK}/'
import numpy as np
import os,sys

file_list=np.load(f'configs/sample_based/{SPACE}/{DATASET}/selected_files.npy',allow_pickle=True)
for f in file_list:
    f_name=f.split('.')[0]
    config=f'{input_dir}/{f_name}'
    out_file=f'{output_dir}/{f_name}'
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    with open(out_file+'/script.sh',mode='w',newline='\n') as script_file:
        script_file.write('#!/bin/bash -l\n')
        script_file.write('#SBATCH --nodes=1\n')
        script_file.write('#SBATCH --cpus-per-task=6\n')
        script_file.write('#SBATCH --mem=10G\n')
        script_file.write('#SBATCH --time=1-12:0:0\n')
        script_file.write('#SBATCH --partition=gpu,gputest\n')
        script_file.write('#SBATCH --gres=gpu:1\n')
        script_file.write(f'#SBATCH --job-name={f_name}_{SPACE}_{DATASET}_{TASK}\n')
        script_file.write(f'#SBATCH --output={out_file}/output.txt\n')
        script_file.write(f'#SBATCH --mail-user=mli176@ucr.edu\n')
        script_file.write(f'#SBATCH --mail-type=FAIL\n')
        script_file.write('source ~/.bashrc\n')
        script_file.write('conda activate pytorch\n')
        script_file.write('export PYTHONPATH=.\n')
        script_file.write(f'python tools/test_net.py --cfg {input_dir}/{f} OUT_DIR {out_file} \n')
    os.system(f'sbatch {out_file}/script.sh')
print(file_list)