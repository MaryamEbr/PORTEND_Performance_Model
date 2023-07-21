#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import paramiko
import sys
import time
import numpy as np
import subprocess as sp
import itertools
import matplotlib.pyplot as plt
from itertools import permutations

sys.path.append('ins_folder/codes')
from helper_functions import describe_instance_cmd, set_ssh_sftp, run_command_ssh, get_topology_info, get_tier_info, get_partitioin_placement_info


# fixed parameters
detailed_flag = False # if true, the experiment will use detailed mode
file_pp = False # if true, the experiment will use the partition placement file, otherwise it will use the loop
sample_counter = 5000  #the number of test sample to do experiments with, it will be always on test set!
threshold_num = 1 #the number of thresholds between 0-1. 1 means lowest threshold (0)
hwm = 100 #number of msg
buf = 10000000 #byte
source_wait = 270 #ms
model_name = 'ResNet20'
dataset_name = 'CIFAR10'
model_mode = 'full' # use final models or ful model in experiment
filter_name = 'mobile_topology6' # the name of the filter to use in the experiment

###### take care of destination

# the folder to copy results to
result_folder = 'exp_folder/Experiment71'


# get topology info
number_of_tiers, type_of_tiers, count_of_tiers, \
        source_tier_name, destination_tier_name, latency_between_tiers, bw_between_tiers = get_topology_info('ins_folder/files/topology_file.txt')

bw_str = str([int(bw) for bw in bw_between_tiers]).replace(" ", "")
lat_str = str([int(lat) for lat in latency_between_tiers]).replace(" ", "")

_, topology_name_list, _ = get_tier_info('ins_folder/files/tier_file.txt')
# print(topology_name_list)
destination_tier_ind = [idx for idx, s in enumerate(topology_name_list) if destination_tier_name in s][0]
source_tier_ind = [idx for idx, s in enumerate(topology_name_list) if source_tier_name in s][0]
print("source tier ind", source_tier_ind, "destination tier ind", destination_tier_ind)

key_name = "ee_key"
describe_ins_list = describe_instance_cmd(key_name, filter_name)
print("aws instances:")
print(*describe_ins_list, sep='\n')


ssh_list, sftp_list = set_ssh_sftp(key_name, describe_ins_list)


# some temporary file transfer to tiers before experiment starts.
for i in range(len(describe_ins_list)):
    sftp_list[i].put("ins_folder/files/partition_placement.txt", "ins_folder/files/partition_placement.txt")
    # sftp_list[i].put("ins_folder/files/topology_file.txt", "ins_folder/files/topology_file.txt")
    # sftp_list[i].put("ins_folder/codes/helper_functions.py", "ins_folder/codes/helper_functions.py")
    # sftp_list[i].put("ins_folder/codes/agent_inference.py", "ins_folder/codes/agent_inference.py")
    # sftp_list[i].put("ins_folder/codes/agent_source.py", "ins_folder/codes/agent_source.py")
    # sftp_list[i].put("ins_folder/codes/agent_destination.py", "ins_folder/codes/agent_destination.py")
    # sftp_list[i].put("ins_folder/codes/inference.py", "ins_folder/codes/inference.py")
    # sftp_list[i].put("ins_folder/codes/ResNet_CIFAR.py", "ins_folder/codes/ResNet_CIFAR.py")
    # sftp_list[i].put("ins_folder/codes/EfficientNet.py", "ins_folder/codes/EfficientNet.py")
time.sleep(5)


# # set tc/ip route for new bw and latency between tiers
# for i in range(len(describe_ins_list)):
#     sftp_list[i].put("ins_folder/files/topology_file.txt", "ins_folder/files/topology_file.txt")
#     run_command_ssh(ssh_list[i], "./ins_folder/codes/./routing_tc_script.py &")
#     print("tc and ip route table are set in tier", i)
# time.sleep(5)


### placement from file or loop, based on file_pp
if file_pp:
    _, placement_name_list = get_partitioin_placement_info('ins_folder/files/partition_placement.txt')
    placement_name_list = [placement_name_list]

else:
    placement_list =  [[0], [1], [2]]
    placement_name_list = []
    
    for placement in placement_list:
        placement_string_list = []
        for p in placement:
            placement_string_list.append(f"tier{p}_{filter_name}")
        placement_name_list.append(placement_string_list)


for placement in placement_name_list:
    print("--- the placement ", placement)
    placement = str(placement)[1:-2].replace(" ", "").replace("'", "")

    
    # renew ssh connections
    ssh_list, sftp_list = set_ssh_sftp(key_name, describe_ins_list)

    # kill previous background python processes
    for i in range(len(describe_ins_list)):
        print("kill previous background python processes in tier", i)
        pids = run_command_ssh(ssh_list[i], "ps -ef | grep 'ins_folder/codes/agent_' | grep -v grep | awk '{print $2}'")
        for pid in pids:
            pid = pid.replace('\n', '')
            print("kill pid", pid)
            run_command_ssh(ssh_list[i], f"sudo kill -9 {pid}")

    time.sleep(10)
    # run 3 main agents in the background
    source_args = f"-sndhwm {hwm} -sndbuf {buf} -wait {source_wait/1000} -threshold {threshold_num} -placement {placement} -sample_counter {sample_counter} -dataset {dataset_name} -model {model_name} -mode {model_mode} -file_pp {file_pp} -detailed {detailed_flag}"
    inference_args = f"-sndhwm {hwm} -rcvhwm {hwm} -rcvbuf {buf} -sndbuf {buf} -placement {placement} -dataset {dataset_name} -model {model_name} -mode {model_mode} -detailed {detailed_flag}"
    destination_args = f"-rcvhwm {hwm} -rcvbuf {buf} -wait {source_wait/1000} -detailed {detailed_flag}"

    cmd_source = f"./ins_folder/codes/agent_source.py {source_args} > ins_folder/results/src_w{source_wait}_t{threshold_num}_sample{sample_counter}_bw{bw_str}_latency{lat_str}_placement[{placement}]_source[{source_tier_name}]_destination[{destination_tier_name}]_dataset[{dataset_name}]_model[{model_name}]_mode[{model_mode}].txt &"
    cmd_destination = f"./ins_folder/codes/agent_destination.py {destination_args} > ins_folder/results/dst_w{source_wait}_t{threshold_num}_sample{sample_counter}_bw{bw_str}_latency{lat_str}_placement[{placement}]_source[{source_tier_name}]_destination[{destination_tier_name}]_dataset[{dataset_name}]_model[{model_name}]_mode[{model_mode}].txt &"
    cmd_inference = f"./ins_folder/codes/agent_inference.py {inference_args} > ins_folder/results/inf_w{source_wait}_t{threshold_num}_sample{sample_counter}_bw{bw_str}_latency{lat_str}_placement[{placement}]_source[{source_tier_name}]_destination[{destination_tier_name}]_dataset[{dataset_name}]_model[{model_name}]_mode[{model_mode}].txt &"


    print("running 3 agents...", flush=True)
    for i in range(len(describe_ins_list)):
        print("i", i)
        print("cmd_source", cmd_source)
        run_command_ssh(ssh_list[i], cmd_source)
        print("cmd_destination", cmd_destination)
        run_command_ssh(ssh_list[i], cmd_destination)
        print("cmd_inference", cmd_inference)
        run_command_ssh(ssh_list[i], cmd_inference)
    print("3 agents are running")


    # wait for the experiment to be done
    # by checking whether the background processes are done in destination tier or not every 10 second
    cmd = "ps -ef | grep 'ins_folder/codes/agent_' | grep -v grep | awk '{print $2}'"
    while(True): 
        print("waiting ... ",flush=True)
        time.sleep(530) 
        if len(run_command_ssh(ssh_list[destination_tier_ind], cmd)) == 0:

            # renew ssh connections
            ssh_list, sftp_list = set_ssh_sftp(key_name, describe_ins_list)

            # copy file from destination
            sftp_list[destination_tier_ind].get(f"ins_folder/results/dst_w{source_wait}_t{threshold_num}_sample{sample_counter}_bw{bw_str}_latency{lat_str}_placement[{placement}]_source[{source_tier_name}]_destination[{destination_tier_name}]_dataset[{dataset_name}]_model[{model_name}]_mode[{model_mode}].txt", \
                f"{result_folder}/dst-w{source_wait}-t{threshold_num}-sample{sample_counter}-bw{bw_str}-lat{lat_str}-place[{placement}]-source[{source_tier_name}]-destination[{destination_tier_name}]-dataset[{dataset_name}]-model[{model_name}]-mode[{model_mode}].txt")

            break
