#!/opt/conda/envs/pytorch/bin/python3
import torch
# torch.set_num_interop_threads(1)
# torch.set_num_threads(1)

import os
import numpy as np
import time
import zmq
import sys
import json
import argparse


import warnings
warnings.filterwarnings("ignore")


from helper_functions import get_topology_info, get_tier_info, wait_for_data

parser = argparse.ArgumentParser()
parser.add_argument('-rcvhwm', default=1000, type=int)
parser.add_argument('-rcvbuf', default=-1, type=int)
parser.add_argument('-detailed', default="False")
parser.add_argument('-wait', default=0.0, type=float)
args = parser.parse_args()
rcvhwm_value = args.rcvhwm
rcvbuf_value = args.rcvbuf
src_wait = args.wait
detailed_flag = (args.detailed == 'True')
# print("args ", args) 

### get topology and tier info
number_of_tiers, type_of_tiers, count_of_tiers, \
    source_tier_name, destination_tier_name, latency_between_tiers, bw_between_tiers = \
        get_topology_info('ins_folder/files/topology_file.txt')

current_tier_name, topology_name_list, topology_ip_list = \
    get_tier_info('ins_folder/files/tier_file.txt')

# print("topology name list:", topology_name_list, " ip list:", topology_ip_list)


### current tier info
current_tier_ind = [idx for idx, s in enumerate(topology_name_list) if current_tier_name in s][0]
current_tier_ip = topology_ip_list[current_tier_ind]
# print("current tier name:", current_tier_name, " ind:", current_tier_ind, " ip:", current_tier_ip)

### destination tier info
destination_tier_ind = [idx for idx, s in enumerate(topology_name_list) if destination_tier_name in s][0]
destination_tier_ip = topology_ip_list[destination_tier_ind]
# print("destination tier name:", destination_tier_name, " ind:", destination_tier_ind, " ip:", destination_tier_ip)

### this is destination agent, so if current tier is not destination tier, exit
if current_tier_ind != destination_tier_ind:
    print("in destination agent: this is not the destination tier")
    sys.exit()



### zmq context
context = zmq.Context()


### socket for destination tier
listening_dest_port = str(6666+current_tier_ind)
socket_pull_dest = context.socket(zmq.PULL)
# socket_pull_dest.setsockopt(zmq.RCVHWM, rcvhwm_value)
# socket_pull_dest.setsockopt(zmq.RCVBUF, rcvbuf_value)
socket_pull_dest.bind("tcp://*:"+listening_dest_port)

while (True):

    msg = wait_for_data(socket_pull_dest, current_tier_ind, 'd', detailed_flag)

    flag = msg[0]
    if "flag_done" in flag:
        break

    counter, results, threshold, partition_list, latency_stack = msg[1:]

    print(json.dumps([counter, partition_list, threshold, results, latency_stack]), flush=True)



### closing sockets and context
# print("in destination agent: closing sockets and context", flush=True)
socket_pull_dest.close()
context.term()

