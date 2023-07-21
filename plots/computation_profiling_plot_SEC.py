#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
from cProfile import label
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
from sklearn.metrics import mean_squared_error
from os import listdir
from os.path import isfile, join
import sys

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['font.size'] = 9

sys.path.append('ins_folder/codes')
from helper_functions import read_exp_results

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


read_again_flag = False
exp_result_folder = "exp_folder/Experiment22"

if read_again_flag == True:
    total_df = read_exp_results(exp_result_folder) 
else:
    total_df = pd.read_pickle(f'{exp_result_folder}/total_df.pkl')


model_name = 'ResNet20'
model_mode = 'full'
dataset = 'CIFAR10'


rows = total_df.loc[(total_df['model']==model_name) & (total_df['dataset']==dataset)  & (total_df['ruined_flag']==False)\
                        & (total_df['threshold'] != 90.0) & (total_df['model_mode']==model_mode)].sort_values(['partitioning'], kind='mergesort')


# drop abnormality
drop_ind = rows[(rows['partitioning'].astype(str) == '[[1, 6]]') & (rows['placement'].astype(str) == '[1]') \
                & (rows['threshold'] == 0.0) ].index
rows = rows.drop(drop_ind)


sum_computations_mean_list = np.array(rows[['sum_computations_tail_mean']].to_numpy()[:,0])
sum_communications_mean_list = np.array(rows[['sum_communications_tail_mean']].to_numpy()[:,0])

estimated_computation_latency_layerwise = rows[['estimated_computation_latency_layerwise']].to_numpy()[:,0]
estimated_computation_latency_blockwise = rows[['estimated_computation_latency_blockwise']].to_numpy()[:,0]
estimated_computation_latency_partitionwise = rows[['estimated_computation_latency_partitionwise']].to_numpy()[:,0]
estimated_computation_latency_partitionwise_plus = np.array(rows[['estimated_computation_latency_partitionwise_plus']].to_numpy()[:,0])

estimated_communication_latency = np.array(rows[['estimated_communication_latency']].to_numpy()[:,0])

placement_list = rows[['placement']].values[:,0]
partitionin_list = rows[['partitioning']].values[:,0]
threshold_list = rows[['threshold']].values[:,0]
exp_exit_rate_list = rows[['exit_rate_all']].values[:,0]
est_exit_rate_list = rows[['estimated_exit_rate']].values[:,0]
model_mode_list = rows[['model_mode']].values[:,0]


# dist_layer = np.round(np.linalg.norm(sum_computations_mean_list - estimated_computation_latency_layerwise), 4)
# dist_block = np.round(np.linalg.norm(sum_computations_mean_list - estimated_computation_latency_blockwise), 4)
# dist_partition_plus = np.round(np.linalg.norm(sum_computations_mean_list - estimated_computation_latency_partitionwise_plus), 4)

dist_layer = np.round(np.sqrt(mean_squared_error(sum_computations_mean_list,estimated_computation_latency_layerwise)), 3)
dist_block = np.round(np.sqrt(mean_squared_error(sum_computations_mean_list,estimated_computation_latency_blockwise)), 3)
dist_partition_plus = np.round(np.sqrt(mean_squared_error(sum_computations_mean_list,estimated_computation_latency_partitionwise_plus)), 3)

r=1.4
fig, ax = plt.subplots(figsize=(r*3.8, r*1.9))   
# ax.set_axisbelow(True)
# ax.grid(color='lightgray')

plt.scatter(sum_computations_mean_list, estimated_computation_latency_layerwise, label=f"layerwise, RMSE {dist_layer}", marker='o', color='tab:blue', s=12)
plt.scatter(sum_computations_mean_list, estimated_computation_latency_blockwise, label=f"blockwise, RMSE {dist_block}", marker='^', color='tab:orange', s=12)
plt.scatter(sum_computations_mean_list, estimated_computation_latency_partitionwise_plus,label=f"partitionwise, RMSE {dist_partition_plus}", marker='s', color='tab:red', s=12)

# plt.scatter(sum_communications_mean_list+sum_computations_mean_list, estimated_communication_latency+estimated_computation_latency_partitionwise_plus, color='darkred')
plt.plot(np.arange(0, 6, 0.01), np.arange(0, 6,0.01), color='black', zorder=-1)

# plt.plot(np.arange(0, max(sum_computations_mean_list), 0.01), np.arange(0, max(sum_computations_mean_list), 0.01), color='black')


# plt.title(f"model {model_name}({model_mode}), dataset {dataset}, all partitions, all thresholds, estimated exitrate")
plt.xlabel('Measured Latency (ms)')
plt.ylabel('Estimated Latency (ms)')
# ax.legend(bbox_to_anchor=(0.705, 0.705), handlelength=1, handletextpad=0.2)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handlelength=1, handletextpad=0.2)
plt.xticks([0, 1, 2, 3, 4, 5, 6])
plt.yticks([0, 1, 2, 3, 4, 5, 6])
plt.tight_layout()
plt.show()