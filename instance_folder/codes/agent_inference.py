#!/opt/conda/envs/pytorch/bin/python3
import torch
# torch.set_num_interop_threads(1)
# torch.set_num_threads(1)


import os
import numpy as np
import time
import zmq
import argparse
import sys
import json


import warnings
warnings.filterwarnings("ignore")


from inference import test_model, load_partition
from helper_functions import get_topology_info, get_tier_info, send_data, wait_for_data

parser = argparse.ArgumentParser()
parser.add_argument('-nocomp', '-nocomputation', default=False, action='store_true')
parser.add_argument('-sndhwm', default=1000, type=int)
parser.add_argument('-rcvhwm', default=1000, type=int)
parser.add_argument('-sndbuf', default=-1, type=int)
parser.add_argument('-rcvbuf', default=-1, type=int)
parser.add_argument('-detailed', default="False")
parser.add_argument('-placement', default='[]')
parser.add_argument('-dataset', default='CIFAR10')
parser.add_argument('-model', default='ResNet20')
parser.add_argument('-mode', default='full')
args = parser.parse_args()
nocomp_mode = args.nocomp
sndhwm_value = args.sndhwm
rcvhwm_value = args.rcvhwm
sndbuf_value = args.sndbuf
rcvbuf_value = args.rcvbuf
detailed_flag = (args.detailed == 'True')
placement_name_list = args.placement.split(',')
dataset_name = args.dataset
model_name = args.model
model_mode = args.mode
print("args ", args)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Used device: ", device)

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



### if current_partition is empty, this tier is not included in placement setting
if current_tier_name not in placement_name_list:
    print("in inference agent: this tier is not included in placement setting")
    sys.exit()


current_tier_placement = [idx for idx, s in enumerate(placement_name_list) if current_tier_name in s][0]
print("placement order list:", placement_name_list, " current placement:", current_tier_placement)


### last tier (in placement) info
last_tier_name = placement_name_list[-1]
last_tier_ind = [idx for idx, s in enumerate(topology_name_list) if last_tier_name in s][0]
last_tier_ip = topology_ip_list[last_tier_ind]
print("last tier name:", last_tier_name, " ind:", last_tier_ind, " ip:", last_tier_ip)


### if this is not the last tier, next tier (in placement) info
if last_tier_ind != current_tier_ind:
    next_tier_name = placement_name_list[current_tier_placement+1]
    next_tier_ind = [idx for idx, s in enumerate(topology_name_list) if next_tier_name in s][0]
    next_tier_ip = topology_ip_list[next_tier_ind]
    print("next tier name:", next_tier_name, " ind:", next_tier_ind, " ip:", next_tier_ip)



### destination tier info
destination_tier_ind = [idx for idx, s in enumerate(topology_name_list) if destination_tier_name in s][0]
destination_tier_ip = topology_ip_list[destination_tier_ind]
print("destination tier name:", destination_tier_name, " ind:", destination_tier_ind, " ip:", destination_tier_ip)



### zmq context 
context = zmq.Context()

### if this is not last tier, socket to push data to next tier 
if last_tier_ind != current_tier_ind:
    forward_data_port = str(5555+next_tier_ind)
    socket_push_next = context.socket(zmq.PUSH)
    # socket_push_next.setsockopt(zmq.SNDHWM, sndhwm_value)
    # socket_push_next.setsockopt(zmq.SNDBUF, sndbuf_value)
    socket_push_next.connect("tcp://"+next_tier_ip+":"+forward_data_port)

### socket to pull from previous tier
listening_data_port = str(5555+current_tier_ind)
socket_pull_prev = context.socket(zmq.PULL)
# socket_pull_prev.setsockopt(zmq.RCVHWM, rcvhwm_value)
# socket_pull_prev.setsockopt(zmq.RCVBUF, rcvbuf_value)
socket_pull_prev.bind("tcp://*:"+listening_data_port)

### socket to push results to destination tier
forward_dest_port = str(6666+destination_tier_ind)
socket_push_dest = context.socket(zmq.PUSH)
# socket_push_dest.setsockopt(zmq.SNDHWM, sndhwm_value)
# socket_push_dest.setsockopt(zmq.SNDBUF, sndbuf_value)
socket_push_dest.connect("tcp://"+destination_tier_ip+":"+forward_dest_port)



# initializing for no computation mode (nocomp_mode = True)
test_loss = 0
test_corrects = 0
entropy = 1


while (True):

    msg = wait_for_data(socket_pull_prev, current_tier_ind, 'i'+str(current_tier_placement), detailed_flag)
    

    flag = msg[0]
    if "done" in flag:

        # if this is not the last tier, pass the flag_done to next tier
        if current_tier_ind != last_tier_ind:
            send_data(socket_push_next, ["flag_done", []], current_tier_ind, 'i'+str(current_tier_placement), detailed_flag)

        # if this is the last tier, pass the flag_done to destination tier
        else:
            send_data(socket_push_dest, ["flag_done", []], current_tier_ind, 'i'+str(current_tier_placement), detailed_flag)

        # break, listening is over
        break

    counter, inputs, labels, threshold, partition_list, latency_stack = msg[1:]


    # the partitioin is changed, load the new model
    if "1" in flag:
        print("changing and loading model partition", end=' ', flush=True)
        # load partition from model
        model, loss_fn = load_partition(model_name, dataset_name, partition_list[current_tier_placement], partition_list, model_mode, device)  

        print("partition: ", partition_list[current_tier_placement], flush=True)
    
    

    # do inference
    if nocomp_mode == False:
        if detailed_flag==True: print("tier"+str(current_tier_ind)+"^"+"torch1"+"^", time.time())
        [test_loss, test_corrects, inputs, entropy] = test_model(model, dataset_name, loss_fn, inputs, labels, device)
        if detailed_flag==True: print("tier"+str(current_tier_ind)+"^"+"torch2"+"^", time.time())

    # if entropy > threshold and this is not the last tier, send intermediate data to next tier
    if entropy > threshold and current_tier_ind != last_tier_ind:
        send_data(socket_push_next, ["flag_inter"+flag[-1], counter, inputs, labels, threshold, partition_list, latency_stack], current_tier_ind, 'i'+str(current_tier_placement), detailed_flag)


    # else, the inference is over, sent results to destination tier
    else:
        send_data(socket_push_dest, ["flag_exit", counter, [test_loss, test_corrects], threshold, partition_list, latency_stack], current_tier_ind, 'i'+str(current_tier_placement), detailed_flag)



### closing sockets and context
print("in inference agent: closing sockets and context", flush=True)
socket_push_next.close()
socket_pull_prev.close()
socket_push_dest.close()
context.term()
