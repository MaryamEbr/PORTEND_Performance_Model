#!/opt/conda/envs/pytorch/bin/python3
import torch
# torch.set_num_interop_threads(1)
# torch.set_num_threads(1)

import os
import sys
import numpy as np
import time

import zmq
import argparse
import json
from itertools import combinations


import warnings
warnings.filterwarnings("ignore")


from inference import load_dataset
from helper_functions import get_topology_info, get_tier_info, send_data, load_partitioning_list, get_partitioin_placement_info

parser = argparse.ArgumentParser()
parser.add_argument('-threshold', default=-1, type=int)
parser.add_argument('-nodata', default=False, action='store_true')
parser.add_argument('-sndhwm', default=1000, type=int)
parser.add_argument('-sndbuf', default=-1, type=int)
parser.add_argument('-wait', default=0.0, type=float)
parser.add_argument('-detailed', default="False")
parser.add_argument('-placement', default='[]')
parser.add_argument('-sample_counter', default=5000, type=int)
parser.add_argument('-dataset', default='CIFAR10')
parser.add_argument('-model', default='ResNet20')
parser.add_argument('-mode', default='full')
parser.add_argument('-file_pp', default="True")
args = parser.parse_args()
nodata_mode = args.nodata
sndhwm_value = args.sndhwm
sndbuf_value = args.sndbuf
src_wait = args.wait
threshold_num = args.threshold
detailed_flag = (args.detailed == 'True')
placement_name_list = args.placement.split(',')
sample_counter = args.sample_counter
dataset_name = args.dataset
model_name = args.model
model_mode = args.mode
file_pp = (args.file_pp == "True")
print("args ", args)

if threshold_num != 1:
    # threshold_list = np.round(np.linspace(0, 1, num=threshold_num), 1)
    threshold_list = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
else:
    threshold_list = [0]
print("inference threshold: ", threshold_list)


### get topology and tier info
number_of_tiers, type_of_tiers, count_of_tiers, \
    source_tier_name, destination_tier_name, latency_between_tiers, bw_between_tiers = \
        get_topology_info('ins_folder/files/topology_file.txt')

current_tier_name, topology_name_list, topology_ip_list = \
    get_tier_info('ins_folder/files/tier_file.txt')

print("topology name list:", topology_name_list, " ip list:", topology_ip_list)


### current tier info
current_tier_ind = [idx for idx, s in enumerate(topology_name_list) if current_tier_name in s][0]
current_tier_ip = topology_ip_list[current_tier_ind]
print("current tier name:", current_tier_name, " ind:", current_tier_ind, " ip:", current_tier_ip)

### source tier info
source_tier_ind = [idx for idx, s in enumerate(topology_name_list) if source_tier_name in s][0]
source_tier_ip = topology_ip_list[source_tier_ind]
print("source tier name:", source_tier_name, " ind:", source_tier_ind, " ip:", source_tier_ip)

### this is source agent, so if current tier is not source tier, exit
if current_tier_ind != source_tier_ind:
    print("in source agent: this is not the source tier")
    sys.exit()

### first tier info
first_tier_name = placement_name_list[0]
first_tier_ind = [idx for idx, s in enumerate(topology_name_list) if first_tier_name in s][0]
first_tier_ip = topology_ip_list[first_tier_ind]
print("first tier name:", first_tier_name, " ind:", first_tier_ind, " ip:", first_tier_ip)


### zmq context
context = zmq.Context()

### socket for first tier to start sending data
forward_first_port = str(5555+first_tier_ind)
socket_push_first = context.socket(zmq.PUSH)
# socket_push_first.setsockopt(zmq.SNDHWM, sndhwm_value) 
# socket_push_first.setsockopt(zmq.SNDBUF, sndbuf_value) 
socket_push_first.connect("tcp://"+first_tier_ip+":"+forward_first_port)


# load the test data
test_loader, _ = load_dataset(dataset_name, model_name, f"ins_folder/test_datasets/{dataset_name.lower()}/")
print("NEW ", len(test_loader), flush=True)

### load the partitioning list for experiment, from file or loop based on file_pp
if file_pp:
    partitioning_list_exp,_ = get_partitioin_placement_info('ins_folder/files/partition_placement.txt')
    partitioning_list_exp = [partitioning_list_exp]
else:
    tier_number = len(placement_name_list)
    partitioning_list_exp = load_partitioning_list(model_mode , model_name, dataset_name, tier_number)

# for each available partitioning in partitioning list
for partitioning in partitioning_list_exp:

    partition_list = []
    for i, part in enumerate(partitioning):
        if i == 0:
            partition_list.append([1, partitioning[i]])
        else:
            partition_list.append([partitioning[i-1]+1, partitioning[i]])


    print("in source agent: starting with partition: ", partition_list, flush=True)
    new_partition_flag = 1 # to inform inference agent to change the loaded partition


    # for each threshold in the list
    for threshold in threshold_list:
        time.sleep(30) # just to give time to previous samples to reach the destination
        print("in source agent: threshold ", threshold, flush=True)

        counter = 0

        # for each sample
        for batch in test_loader:
            for inputs, labels in zip(batch[0], batch[1]):

                # # instead of time.sleep(): (more accurate)
                # target_time = time.time_ns() + src_wait*1000000000
                # while time.time_ns() < target_time:
                #     pass


                time.sleep(src_wait)


                # add a dimension in batch position
                inputs = torch.unsqueeze(inputs, dim=0).detach().clone()
                labels = torch.unsqueeze(labels, dim=0).detach().clone()


                # print("inputs: ", inputs.shape, " labels: ", labels.shape, flush=True)
                # send data to inference agent of first tier
                send_data(socket_push_first, [f"flag_src{new_partition_flag}", counter, inputs, labels, threshold, partition_list, []], current_tier_ind, 's', detailed_flag)
                counter += 1
                new_partition_flag = 0


                # if we reach the sample counter for the experiment, exit
                if counter >= sample_counter:
                    break
            if counter >= sample_counter:
                break


send_data(socket_push_first, ["flag_done", []], current_tier_ind, 's', detailed_flag)

### closing sockets and context
print("in source agent: closing sockets and context", flush=True)
socket_push_first.close()
context.term()

