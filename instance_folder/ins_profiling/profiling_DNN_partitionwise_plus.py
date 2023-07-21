#!/usr/bin/python3
import os
import torch
import numpy as np
import time
import sys
import pickle
import itertools
import matplotlib.pyplot as plt
import timeit
import json
import pandas as pd
import torch.nn as nn
from functools import partial


# sys.path.append('/home/ubuntu/ins_folder/codes')
sys.path.append('ins_folder/codes')

from ResNet_CIFAR import ResNet_CIFAR_model_final
from EfficientNet import EfficientNet_final
from inference import load_partition, test_model


exp_num = 100
exp_repeat = 3
result_folder = "ins_folder/ins_profiling/profiling_results/"


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)

### get model and dataset info
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

# model_ds_combinations = [['ResNet20', 'CIFAR10'], ['ResNet110', 'CIFAR10'], ['EfficientNetB0', 'ImageNet'], ['EfficientNetB7', 'ImageNet']]
model_ds_combinations = [['EfficientNetB7', 'ImageNet']]
for [model_name, ds_name] in model_ds_combinations:
    print("---- ", model_name, ds_name)
    ### get input_size for each possible partition
    branch_number = model_info_dict[model_name]['branch_number']
    selected_exits = np.array([a+1 for a in range(branch_number)])
    if 'ResNet' in model_name:
        input_size = dataset_info_dict[ds_name]['input_size']
        num_channels = dataset_info_dict[ds_name]['num_channels']
        model = ResNet_CIFAR_model_final(model_info_dict[model_name]['model_n'], model_info_dict[model_name]['branch_number'], dataset_info_dict[ds_name]['num_class'], selected_exits, device).to(device)
    if 'EfficientNetB0' in model_name:
        input_size = dataset_info_dict[ds_name][f"input_size_{model_name[-2:]}"]
        num_channels = dataset_info_dict[ds_name]['num_channels']
        model = EfficientNet_final(model_info_dict[model_name]['dropout'], model_info_dict[model_name]['width_mult'], model_info_dict[model_name]['depth_mult'], model_info_dict[model_name]['norm_layer'], dataset_info_dict[ds_name]['num_class'], selected_exits).to(device)



    # # pass an input through the model to get detailed intermediate shape
    # input_tensor = torch.rand(input_size*input_size*num_channels).view(num_channels, input_size, input_size).unsqueeze(dim=0).detach().clone().to(device)
    # _, intermediate_list = model(input_tensor)

    # # the length of intermediate list will always be branch_number+1. 
    # # just ignore the last element. you don't need the intermediate size after last partition.
    # partition_input_size = {i+1:a.shape for i,a in enumerate(intermediate_list) if i!=branch_number}
    
    partition_input_size = {1: torch.Size([1, 3, 600, 600]), 2: torch.Size([1, 64, 300, 300]), 3: torch.Size([1, 32, 300, 300]), 4: torch.Size([1, 48, 150, 150]), 5: torch.Size([1, 80, 75, 75]), 6: torch.Size([1, 160, 38, 38]), 7: torch.Size([1, 224, 38, 38]), 8: torch.Size([1, 384, 19, 19])}
    print(partition_input_size)

    # method two, just running the partition, in eval mode, with no_grad
    print("computation latency profiling starts....")
    method2_list = []
    for i in range(1, branch_number+1):
        for j in range(i, branch_number+1):
            partition = [i, j]
            print(partition, flush=True)

            p_input_size = partition_input_size[partition[0]][-1]
            p_num_channels = partition_input_size[partition[0]][1]
            
            with torch.no_grad():
                # load the partition, with last exit branch, from full configuration model
                model, loss_fn = load_partition(model_name, ds_name, partition, 'full', device)
                input_tensor = torch.rand(p_input_size*p_input_size*p_num_channels).view(p_num_channels, p_input_size, p_input_size).unsqueeze(dim=0).detach().clone().to(device)

                model.eval()
                latency_partition = np.array(timeit.repeat(stmt='output = model(input_tensor)', number=exp_num, repeat=exp_repeat, globals=globals()))/exp_num*1000
            
            method2_list.append([partition[0], partition[-1], latency_partition])


    method2_list = pd.DataFrame(method2_list, columns = ['partition_start', 'partition_end', 'latency_partition'])
    profiling2_file = open(f"{result_folder}profiling_partition_{model_name}_{ds_name}.txt", 'w')
    json.dump(method2_list.to_json(), profiling2_file)

