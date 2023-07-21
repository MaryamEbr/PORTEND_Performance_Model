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



exp_num = 100
exp_repeat = 3
result_folder = "ins_folder/ins_profiling/profiling_results/"


################################################################################ pickle
flag = "flag_string"
counter = 1000
threshold = 0.1
partition = [[1, 4], [5, 6]]
latency_stack = [["s", 0, 'snd', time.time()], ["r", 1, 'rcv', time.time()], ["s", 2, 'snd', time.time()], ["r", 3, 'rcv', time.time]]
label_tensor = torch.randint(low=1, high=10, size=(1,))
pickle_data = []

####warmup
obj = [flag, counter, threshold, partition, latency_stack,  torch.randn(10000), label_tensor]
for run in range(1000):
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)



num_channels_list = [384, 224, 192, 160, 112, 80, 64, 48, 40, 32, 24, 16, 3]
input_size_list = [600, 300, 224, 150, 112, 75, 56, 38, 32, 28, 19, 16, 14, 8, 7]     
relu_data = []

all_list =  list(itertools.product(num_channels_list, input_size_list))

for num_channels, input_size in all_list:

    if num_channels==384 and input_size==600:
        continue
    if num_channels==224 and input_size==600:
        continue
    if num_channels==192 and input_size==600:
        continue
    if num_channels==160 and input_size==600:
        continue
    
    print(num_channels, input_size, end=" - ", flush=True)
    input_tensor = torch.rand(input_size*input_size*num_channels).view(num_channels, input_size, input_size)
    label_tensor = torch.randint(low=1, high=10, size=(1,))
    

    input_tensor = torch.unsqueeze(input_tensor, dim=0).detach().clone()
    label_tensor = torch.unsqueeze(label_tensor, dim=0).detach().clone()

    obj = [flag, counter, threshold, partition, latency_stack, input_tensor, label_tensor]
    
    dump_latency = np.mean(timeit.repeat(stmt='pickled = pickle.dumps(obj)', number=exp_num, repeat=exp_repeat, globals=globals()))/exp_num*1000
    pickled = pickle.dumps(obj)
    load_latency = np.mean(timeit.repeat(stmt='unpickled = pickle.loads(pickled)', number=exp_num, repeat=exp_repeat, globals=globals()))/exp_num*1000

    pickle_data.append([num_channels, input_size, len(pickled), dump_latency, load_latency])


pickle_df = pd.DataFrame(pickle_data, columns = ['num_channels', 'input_size', 'pickled_size', 'dump_latency', 'load_latency'])
profiling_file = open(f"{result_folder}profiling_pickle.txt", 'w')
json.dump(pickle_df.to_json(), profiling_file)
print("pickle done.")
