#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
from torch import Tensor 
import torchvision.transforms as transforms
import os
import sys
from collections import Counter
from functools import partial
import pandas as pd
import itertools
from itertools import combinations
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 11

sys.path.append('ins_folder/codes')
from inference import load_dataset
from helper_functions_last import get_lookup_tables, get_topology_info, add_estimated_exitrate_validation,load_partitioning_list,\
    add_estimated_accuracy_validation, add_estimated_computation, add_estimated_pickle_size, add_estimated_communication, save_df,\
        add_estimated_e2e_latency
        
        
### some info
dataset_info_dict = {'CIFAR10': {'num_class': 10, 'input_size': 32, 'num_channels': 3, 'validation_size': 5000, 'test_size': 5000},
                    'CIFAR100': {'num_class': 100, 'input_size': 32, 'num_channels': 3, 'validation_size': 5000, 'test_size': 5000},
                    'ImageNet': {'num_class': 1000, 'input_size_B0': 224, 'input_size_B7': 600 ,'crop_size_B0': 256, 'crop_size_B7': 600, 'num_channels': 3, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'validation_size': 25000, 'test_size': 25000},
                    'AudioSet': {'num_class': 527, 'input_size': [96, 64], 'num_channels': 1},
                    }
    
model_info_dict = {'ResNet20': {'model_n': 3, 'branch_number': 10},
                    'ResNet110': {'model_n': 18, 'branch_number': 55},
                    'EfficientNetB0': {'branch_number': 8, 'dropout': 0.2, 'width_mult': 1.0, 'depth_mult': 1.0, 'norm_layer': nn.BatchNorm2d},
                    'EfficientNetB7': {'branch_number': 8, 'dropout': 0.5, 'width_mult': 2.0, 'depth_mult': 3.1, 'norm_layer': partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)},
                    'VGGish': {'branch_number': 4}, 
                    }

source_tier_ind = 0
destination_tier_ind = 0
latency_between_tiers_list = [[20, 20]]
bw_between_tiers_list = [ [1, 500], [2, 500], [3, 500], [4, 500], [5, 500], [6, 500], [7, 500], [8, 500], [9, 500], [10, 500]]


dataset = 'ImageNet'
model = 'EfficientNetB7'
model_mode = 'full'

threshold_list =  [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
placement_list = [[0], [1], [2], [0, 1], [1, 2], [0, 2]]

partial_exec = True # whether to use entire model for inference or not



total_list = []

all_list =  list(itertools.product(threshold_list, placement_list, latency_between_tiers_list, bw_between_tiers_list))
for threshold, placement, latency_between_tiers, bw_between_tiers  in all_list:


    for partition in combinations(range(1, model_info_dict[model]['branch_number']+1), len(placement)):
        
        # if partition != (3, 8):
        #     continue

        if model_info_dict[model]['branch_number'] not in partition and not partial_exec:
            continue
        
        if len(partition) == 1 and threshold != 0:
            continue

        # print('threshold: ', threshold, 'placement: ', placement, 'partitioning: ', partition)
        beg = 1
        partitioning_list = []
        for p in partition:

            partitioning_list.append([beg, p])
            beg = p+1
        
        if len(partitioning_list) == 1 and threshold != 0:
            continue
        
        
        partial_flag = False
        if partitioning_list[-1][-1] != model_info_dict[model]['branch_number']:
            partial_flag = True
        
        # print('threshold: ', threshold, 'placement: ', placement, 'partitioning: ', partitioning_list)
        total_list.append([threshold, bw_between_tiers, latency_between_tiers,\
                source_tier_ind, destination_tier_ind, dataset, model, model_mode,\
                placement, partitioning_list, len(placement), partial_flag])




total_df = pd.DataFrame(total_list, columns = ['threshold',
                                    'bw_list', 'latency_list',\
                                    'source_tier', 'destination_tier', 'dataset', 'model', 'model_mode',\
                                    'placement', 'partitioning', 'num_partitions', 'partial_flag'])



total_df = add_estimated_exitrate_validation(total_df)
total_df = add_estimated_accuracy_validation(total_df)
total_df = add_estimated_computation(total_df)
total_df = add_estimated_pickle_size(total_df)
total_df = add_estimated_communication(total_df)
total_df = add_estimated_e2e_latency(total_df)



# get the lowest latency for each bw in bw_list
x_list = [bw[0] for bw in bw_between_tiers_list]
y_02_list = []
y_12_list = []
y_01_list = []
y_0_list = []
y_1_list = []
y_2_list = []

rows = total_df.loc[total_df['estimated_accuracy']>0.84]
for bw in bw_between_tiers_list:
    rows_02 = rows.loc[(rows['bw_list'].astype(str)==str(bw)) & ((rows['placement'].astype(str)=='[0, 2]'))]\
        .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False]).head(1)
        
    rows_12 = rows.loc[(rows['bw_list'].astype(str)==str(bw)) & ((rows['placement'].astype(str)=='[1, 2]'))]\
        .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False]).head(1)
        
    rows_01 = rows.loc[(rows['bw_list'].astype(str)==str(bw)) & ((rows['placement'].astype(str)=='[0, 1]'))]\
        .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False]).head(1)
        
    rows_0 = rows.loc[(rows['bw_list'].astype(str)==str(bw)) & ((rows['placement'].astype(str)=='[0]'))]\
        .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False]).head(1)
        
    rows_1 = rows.loc[(rows['bw_list'].astype(str)==str(bw)) & ((rows['placement'].astype(str)=='[1]'))]\
        .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False]).head(1)
        
    rows_2 = rows.loc[(rows['bw_list'].astype(str)==str(bw)) & ((rows['placement'].astype(str)=='[2]'))]\
        .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False]).head(1)
        
    y_02_list.append(rows_02['estimated_e2e_latency'].values[0])
    y_12_list.append(rows_12['estimated_e2e_latency'].values[0])
    y_01_list.append(rows_01['estimated_e2e_latency'].values[0])
    y_0_list.append(rows_0['estimated_e2e_latency'].values[0])
    y_1_list.append(rows_1['estimated_e2e_latency'].values[0])
    y_2_list.append(rows_2['estimated_e2e_latency'].values[0])
    
    

# ax.set_axisbelow(True)
# plt.grid(True, color='lightgray') 

r=1.7
fig, ax = plt.subplots(figsize=(r*3.8, r*2))  

plt.plot(x_list, np.array(y_02_list)/1000.0, label='tier 0, tier 2', marker='o', color='maroon')
# plt.plot(x_list, y_12_list, label='tier1 -> tier2', marker='o')
# plt.plot(x_list, y_01_list, label='tier0 -> tier1', marker='o')
# plt.plot(x_list, y_0_list, label='tier0', marker='o')
# plt.plot(x_list, y_1_list, label='tier1', marker='o')
plt.plot(x_list,np.array(y_2_list)/1000.0, label='tier 2', marker='o', color='royalblue')


# plt.title('EfficientNetB7, ImageNet', fontsize=12)
plt.legend()
plt.ylabel("End to End Latency (sec)")
plt.xlabel("End Device Bandwidth (Mbps)")

plt.xticks(x_list, x_list)

plt.tight_layout()
plt.show()
