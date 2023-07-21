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

sys.path.append('ins_folder/codes')
from inference import load_dataset
from helper_functions_last import get_lookup_tables, get_topology_info, add_estimated_exitrate_validation,load_partitioning_list,\
    add_estimated_accuracy_validation, add_estimated_computation, add_estimated_pickle_size, add_estimated_communication, save_df,\
        add_estimated_e2e_latency

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 12



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
# latency_between_tiers_list = [[10, 490],  [50, 450], [100, 400], [200, 300], [300, 200], [400, 100], [450, 50], [490, 10]]
# latency_between_tiers_list = [[10, 490], [20, 480], [30, 470], [40, 460], [50, 450], [60, 440], [70, 430], [80, 420], [90, 410], [100, 400], [200, 300], [300, 200], [400, 100], [450, 50], [490, 10]]
# latency_between_tiers_list = [[490, 10], [480, 20], [470, 30], [460, 40], [450, 50], [440, 60], [430, 70], [420, 80], [410, 90], [400, 100], [300, 200], [200, 300], [100, 400], [50, 450], [10, 490]]
latency_between_tiers_list = [[5, 20, 50]]
bw_between_tiers_list = [[10, 600, 600]]

dataset = 'ImageNet'
model = 'EfficientNetB7'
model_mode = 'full'

threshold_list =  [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
# placement_list = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]]
# placement_list = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]
# placement_list = [[0], [1], [0, 1]]
placement_list = [[0]]

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
        
        
        # print('placement', placement, 'coeff list', coeff_list)
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

### 1 tier
rows_1t = total_df.loc[total_df['estimated_accuracy']>0.84].sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
### 2 tier
rows_2t = total_df.loc[total_df['estimated_accuracy']>0.84].sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])

### 3 tier
rows_3t = total_df.loc[total_df['estimated_accuracy']>0.84].sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
### 4 tier
rows_4t = total_df.loc[total_df['estimated_accuracy']>0.84].sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])

r=1.8
fig, ax = plt.subplots(figsize=(r*3, r*2))  
ax.set_axisbelow(True)
plt.grid(True, color='lightgray') 


x_list = [1, 2, 3, 4]
t1_list = 20502.629227
t2_list = 5715.138036	
t3_list = 3581.606101
t4_list = 3581.606101	


plt.plot(x_list, [t1_list, t2_list, t3_list, t4_list], marker='o', color='maroon')  



# plt.title('EfficientNetB7, ImageNet', fontsize=12)
# plt.legend(title='Placement')
plt.ylabel("End to End Latency (ms)")
plt.xlabel("Number of Tiers")
plt.xticks(x_list)

plt.tight_layout()
plt.show()
