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
import soundfile as sf


# sys.path.append('/home/ubuntu/ins_folder/codes')
sys.path.append('ins_folder/codes')

from ResNet_CIFAR import ResNet_CIFAR_model_final
from EfficientNet import EfficientNet_final
from vggish import VGGish_final
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
                    'ImageNet': {'num_class': 1000, 'input_size_B0': 224, 'input_size_B7': 600 ,'crop_size_B0': 256, 'crop_size_B7': 600, 'num_channels': 3, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'validation_size': 5000, 'test_size': 5000},
                    'AudioSet': {'num_class': 527, 'input_size': [96, 64], 'num_channels': 1},
                    }
    
model_info_dict = {'ResNet20': {'model_n': 3, 'branch_number': 10},
                    'ResNet110': {'model_n': 18, 'branch_number': 55},
                    'EfficientNetB0': {'branch_number': 8, 'dropout': 0.2, 'width_mult': 1.0, 'depth_mult': 1.0, 'norm_layer': nn.BatchNorm2d},
                    'EfficientNetB7': {'branch_number': 8, 'dropout': 0.5, 'width_mult': 2.0, 'depth_mult': 3.1, 'norm_layer': partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)},
                    'VGGish': {'branch_number': 4}, 
                    }



model_name = 'VGGish'
ds_name = 'AudioSet'


### get info for loading the model
branch_number = model_info_dict[model_name]['branch_number']
selected_exits = np.array([a+1 for a in range(branch_number)])

### loading a wav sample
wav_data, _ = sf.read('/home/ec2-user/ins_folder/ins_profiling/sample.wav', dtype='int16')

### loading the full configuration model
model = VGGish_final(selected_exits, device).to(device)

### passing an input through the model to get detailed intermediate shape
_, intermediate_list = model(torch.tensor(wav_data), device)

# the length of intermediate list will always be branch_number+1. 
# just ignore the last element. you don't need the intermediate size after last partition.
partition_input_size = {i+1:a.shape for i,a in enumerate(intermediate_list) if i!=branch_number}

print(partition_input_size)

# method two, just running the partition, in eval mode, with no_grad
print("computation latency profiling starts....")
method2_list = []
for i in range(1, branch_number+1):
    for j in range(i, branch_number+1):
        partition = [i, j]
        print(partition, flush=True)
        
        if i == 1:
            input_tensor = torch.tensor(wav_data).detach().clone()
        else:
            torch_size = partition_input_size[partition[0]-1]
            input_tensor = torch.rand(torch_size[0]*torch_size[1]*torch_size[2]*torch_size[3]).view(torch_size[0],torch_size[1],torch_size[2],torch_size[3]).detach().clone().to(device)

        
        with torch.no_grad():
            # load the partition, with last exit branch, from full configuration model
            model, _ = load_partition(model_name, ds_name, partition, 'sth', 'full', device)
            
            model.eval()
            latency_partition = np.array(timeit.repeat(stmt='output = model(input_tensor)', number=exp_num, repeat=exp_repeat, globals=globals()))/exp_num*1000
        
        method2_list.append([partition[0], partition[-1], latency_partition])


method2_list = pd.DataFrame(method2_list, columns = ['partition_start', 'partition_end', 'latency_partition'])
profiling2_file = open(f"{result_folder}profiling_partition_{model_name}_{ds_name}.txt", 'w')
json.dump(method2_list.to_json(), profiling2_file)

