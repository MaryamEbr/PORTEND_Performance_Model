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


################################# some functions for plot hover
def update_annot(ind):
    ind = ind["ind"][0]
    pos = sc.get_offsets()[ind]
    annot.xy = pos
    text = f"*partition: {partitionin_list[ind]} *placement: {placement_list[ind]} \n*threshold: {threshold_list[ind]}  *destination: {destination_list[ind]}"

    annot.set_text(text)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

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




dataset = 'ImageNet'
model = 'EfficientNetB0'
model_mode = 'full'

source_tier_ind = 0
destination_tier_ind = 0
latency_between_tiers_list = [[20, 40]]
bw_between_tiers_list = [[10, 400]]

threshold_list =  [0, 0.1, 0.2, 0.3, 0.4,  0.6, 0.7, 0.8, 0.9, 1]
placement_list = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]


partial_exec = True # whether to use entire model for inference or not


total_list = []

all_list =  list(itertools.product(threshold_list, placement_list, latency_between_tiers_list, bw_between_tiers_list))
for threshold, placement, latency_between_tiers, bw_between_tiers  in all_list:


    for partition in combinations(range(1, model_info_dict[model]['branch_number']+1), len(placement)):
        
        

        if model_info_dict[model]['branch_number'] not in partition and not partial_exec:
            continue
        
        if len(partition) == 1 and threshold != 0:
            continue

        print('threshold: ', threshold, 'placement: ', placement, 'partitioning: ', partition)
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

total_df = total_df.loc[total_df['estimated_accuracy']>0.60]
save_df (total_df, 'performance_model_results_last/')

filtered_df = total_df
estimated_latency = filtered_df['estimated_e2e_latency'].values
estimated_comm_latency = filtered_df['estimated_communication_latency'].values
estimated_comp_latency = filtered_df['estimated_computation_latency'].values
estimated_accuracy = filtered_df['estimated_accuracy'].values
partitionin_list = filtered_df['partitioning'].values
placement_list = filtered_df['placement'].values
threshold_list = filtered_df['threshold'].values
destination_list = filtered_df['destination_tier'].values




fig,ax = plt.subplots(figsize=(9,6))

sc = plt.scatter(estimated_latency, estimated_accuracy, marker='o', color='blue')


# annotation
annot = ax.annotate("", xy=(0,0), xytext=(-120,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.grid()
plt.xlabel('latency (ms)')
plt.ylabel('accuracy')
plt.show()









