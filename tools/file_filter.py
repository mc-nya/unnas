import os
import sys
import numpy as np
import random
input_path='configs/sample_based/darts/cifar10/archived_files/'
for root,dirs,files in os.walk(input_path):
    if not len(files)==0:
        selected_files=np.sort(random.sample(files,200))
        break
output_path=input_path+'/../filtered_file/'
if not os.path.exists(output_path):
    os.makedirs(output_path,exist_ok=True)
np.save(output_path+'selected_files.npy',selected_files)
np.savetxt(output_path+'selected_files.txt',selected_files,delimiter=' ',fmt='%s')

for root,dirs,files in os.walk(input_path):
    if not len(dirs)==0:
        for d in dirs:
            in_dir=input_path+d
            out_dir=output_path+d
            if not os.path.exists(out_dir):
                os.makedirs(out_dir,exist_ok=True)
            for f in selected_files:
                in_f=in_dir+'/'+f
                out_f=out_dir+'/'+f
                os.system(f'cp {in_f} {out_f}')
        print(dirs)
