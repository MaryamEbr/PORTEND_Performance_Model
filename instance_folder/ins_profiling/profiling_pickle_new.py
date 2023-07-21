#!/usr/bin/python3
import os
import torch
import numpy as np
import time
import sys
import pickle
import itertools
import timeit
import json
import pandas as pd
import tqdm
from functools import reduce


exp_num = 100
exp_repeat = 3
result_folder = "ins_folder/ins_profiling/profiling_results/"


################################################################################ pickle
flag = "flag_string"
counter = 1000
threshold = 0.1
partition = [[1, 4], [5, 6]]
latency_stack = [["s", 0, 'snd', time.time()], ["r", 1, 'rcv', time.time()], ["s", 2, 'snd', time.time()], ["r", 3, 'rcv', time.time]]
label_tensor = torch.randint(low=1, high=1000, size=(1,))
pickle_data = []

####warmup
obj = [flag, counter, threshold, partition, latency_stack,  torch.randn(10000), label_tensor]
for run in range(1000):
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)

model_ds_sets = [['VGGish', 'AudioSet'], ['ResNet110', 'CIFAR10'], ['ResNet20', 'CIFAR10'], ['EfficientNetB0', 'ImageNet'], ['EfficientNetB7', 'ImageNet']]

pickle_dict = {}

# load the intermediate dict
with open('ins_folder/ins_profiling/intermediate_dict.pkl', 'rb') as f:
    intermediate_dict = pickle.load(f)

for model_name, ds_name in model_ds_sets:
    curr_pickle_data = []
    print("-----------  ", model_name, "  ", ds_name)
    
    curr_intermediate_dict = intermediate_dict[model_name]
    # print("            ", curr_intermediate_dict)
    
    for key in curr_intermediate_dict:
        # print("                 ", key)
        
        mul_size = reduce(lambda x, y: x*y, curr_intermediate_dict[key])
        input_tensor = torch.rand(mul_size).view(curr_intermediate_dict[key]).detach().clone()
        label_tensor = torch.randint(low=1, high=1000, size=(1,)).detach().clone()


        obj = [flag, counter, threshold, partition, latency_stack, input_tensor, label_tensor]
        
        dump_latency = np.mean(timeit.repeat(stmt='pickled = pickle.dumps(obj)', number=exp_num, repeat=exp_repeat, globals=globals()))/exp_num*1000
        pickled = pickle.dumps(obj)
        load_latency = np.mean(timeit.repeat(stmt='unpickled = pickle.loads(pickled)', number=exp_num, repeat=exp_repeat, globals=globals()))/exp_num*1000

        curr_pickle_data.append([key, len(pickled), dump_latency, load_latency])
    # print(curr_pickle_data)

    pickle_dict[model_name] = curr_pickle_data


with open('ins_folder/ins_profiling/profiling_results/profiling_pickle.pkl', 'wb') as f:
    pickle.dump(pickle_dict, f)