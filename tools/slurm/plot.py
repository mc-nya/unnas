SPACE='darts'
DATASET='cifar10'
TASK1='psd5000'
TASK2='cls'
input_dir=f'configs/sample_based/{SPACE}/{DATASET}/{TASK1}/'
T1_dir=f'results/sample_based/{SPACE}/{DATASET}/{TASK1}/'
T2_dir=f'results/sample_based/{SPACE}/{DATASET}/{TASK2}/'
import numpy as np
import os,sys
import scipy.stats
file_list=np.load(f'configs/sample_based/{SPACE}/{DATASET}/selected_files.npy',allow_pickle=True)
acc1=[]
acc2=[]
for f in file_list:
    f_name=f.split('.')[0]
    T1_file=f'{T1_dir}/{f_name}/result.txt'
    T2_file=f'{T2_dir}/{f_name}/result.txt'
    if not os.path.exists(T1_file):
        continue
    if not os.path.exists(T2_file):
        continue
    
    with open(T1_file,mode='r') as T1:
        with open(T2_file,mode='r') as T2:
            temp1=float(T1.read())
            temp2=float(T2.read())
            acc1.append(temp1)
            acc2.append(temp2)
print(acc1)
print(acc2)
acc1=np.array(acc1)/100
acc2=np.array(acc2)
print(acc1-acc2)
import matplotlib.pyplot as plt
plt.scatter(acc2,acc1)
plt.title(scipy.stats.spearmanr(acc1,acc2)[0])
plt.savefig(f'{TASK1}_{TASK2}.pdf')
