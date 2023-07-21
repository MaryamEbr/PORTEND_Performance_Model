#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import json
import subprocess as sp
import seaborn as sns
import pandas as pd
import torch
import time
import pickle
import random
from os import listdir
from os.path import isfile, join
import sys

sys.path.append('ins_folder/codes')
from helper_functions import read_exp_results

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


read_again_flag = False
exp_result_folder = "exp_folder/Experiment73"
model_mode = 'full'
if '60' in exp_result_folder or '73' in exp_result_folder:
    dataset = 'ImageNet'
    model_name = 'EfficientNetB0'

if '61' in exp_result_folder or '71' in exp_result_folder:
    dataset = 'CIFAR10'
    model_name = 'ResNet20' 

if '62' in exp_result_folder or '72' in exp_result_folder:
    dataset = 'CIFAR10'
    model_name = 'ResNet110'
if '74' in exp_result_folder:
    dataset = 'AudioSet'
    model_name = 'VGGish'


if read_again_flag == True:
    total_df = read_exp_results(exp_result_folder) 
else:
    total_df = pd.read_pickle(f'{exp_result_folder}/total_df.pkl')




rows = total_df.loc[(total_df['model_mode']==model_mode) & (total_df['source_wait']==707)].sort_values(['partitioning'], kind='mergesort')

### drop abormality 
# mask = rows[['sum_computations_tail_mean', 'estimated_computation_latency']].apply(lambda x: abs(x[0] - x[1]) < 20, axis=1)
# rows = rows[mask]
# mask = rows[['sum_communications_tail_mean', 'estimated_communication_latency']].apply(lambda x: abs(x[0] - x[1]) < 20, axis=1)
# rows = rows[mask]



experiment_comp_latency = np.array(rows[['sum_computations_tail_mean']].to_numpy()[:,0])

estimated_comp_latency = np.array(rows[['estimated_computation_latency']].to_numpy()[:,0])
estimated_pickel_latency = np.array(rows[['estimated_pickle_latency']].to_numpy()[:,0])



fig,ax = plt.subplots(figsize=(9,6))
plt.scatter(experiment_comp_latency, estimated_comp_latency+estimated_pickel_latency, color='darkgreen')
plt.plot(np.arange(0, 4000, 0.01), np.arange(0, 4000, 0.01), color='black')



plt.title(f"model {model_name}({model_mode}), dataset {dataset}")
plt.xlabel('experiment computation latency')
plt.ylabel('estimated computation latency')
plt.grid()
plt.show()