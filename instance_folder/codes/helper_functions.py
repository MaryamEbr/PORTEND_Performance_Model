import subprocess as sp
import sys
import pickle
import time
import json
import matplotlib.pyplot as plt
import paramiko
import numpy as np
import torch.nn as nn
import pandas as pd
import torch
from os import access, listdir
import itertools
import soundfile as sf
from os.path import isfile, join
from functools import partial
from ResNet_CIFAR import ResNet_CIFAR_model_final
from EfficientNet import EfficientNet_final
from vggish import VGGish_final


### get model and dataset info
dataset_info_dict = {'CIFAR10': {'num_class': 10, 'input_size': 32, 'num_channels': 3, 'validation_size': 5000, 'test_size': 5000},
                    'CIFAR100': {'num_class': 100, 'input_size': 32, 'num_channels': 3, 'validation_size': 5000, 'test_size': 5000},
                    'ImageNet': {'num_class': 1000, 'input_size_B0': 224, 'input_size_B7': 600 ,'crop_size_B0': 256, 'crop_size_B7': 600, 'num_channels': 3, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'validation_size': 25000, 'test_size': 5000},
                    'AudioSet': {'num_class': 527, 'input_size': [96, 64], 'num_channels': 1, 'validation_size': 5000, 'test_size': 1000},
                    }
    
model_info_dict = {'ResNet20': {'model_n': 3, 'branch_number': 10},
                    'ResNet110': {'model_n': 18, 'branch_number': 55},
                    'EfficientNetB0': {'branch_number': 8, 'dropout': 0.2, 'width_mult': 1.0, 'depth_mult': 1.0, 'norm_layer': nn.BatchNorm2d},
                    'EfficientNetB7': {'branch_number': 8, 'dropout': 0.5, 'width_mult': 2.0, 'depth_mult': 3.1, 'norm_layer': partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)},
                    'VGGish': {'branch_number': 4}, 
                    }

used_model_ds = [('ResNet20', 'CIFAR10'), ('ResNet110', 'CIFAR10'), ('EfficientNetB0', 'ImageNet'), ('EfficientNetB7', 'ImageNet'), ('VGGish', 'AudioSet')]
used_devices = ['a1.medium', 'm4.large', 'g4dn.xlarge', 'a1.large', 'g5.xlarge']

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def get_topology_info(file_address):
    with open(file_address) as file:
        for line in file:
            line = line.replace(' ', '').replace('\n', '')
            if 'number_of_tiers' in line:
                number_of_tiers = int(line.split('>')[1])
            if 'type_of_tiers' in line:
                type_of_tiers = line.split('>')[1].split('^')
            if 'count_of_tiers' in line:
                count_of_tiers = [int(c) for c in line.split('>')[1].split('^')]
            if 'source_tier_name' in line:
                source_tier_name = line.split('>')[1]
            if 'destination_tier_name' in line:
                destination_tier_name = line.split('>')[1]
            if 'latency_between_tiers' in line:
                latency_between_tiers = [float(c) for c in line.split('>')[1].split('^')]
            if 'bw_between_tiers' in line:
                bw_between_tiers = [float(c) for c in line.split('>')[1].split('^')]

    return number_of_tiers, type_of_tiers, count_of_tiers, \
        source_tier_name, destination_tier_name, latency_between_tiers, bw_between_tiers

def get_tier_info(file_address):
    name_list = []
    private_ip_list = []
    with open(file_address) as file:
        for line in file:
            line = line.replace(' ', '').replace('\n', '')
            if 'current_tier_name' in line:
                current_tier_name = line.split('>')[1]
            if '>>' in line:
                line = line.split('>>')
                name_list.append(line[0])
                private_ip_list.append(line[1])

    ## sort based on tier names
    [name_list, private_ip_list] = zip(*sorted(zip(name_list, private_ip_list)))
    return current_tier_name, name_list, private_ip_list

def get_partitioin_placement_info(file_address):
    partition_list = []
    placement_name_list = []

    with open(file_address) as file:
        for i, line in enumerate(file):
            line = line.replace(' ', '').replace('\n', '').split(":")
            placement_name_list.append(line[0])
            partition_list.append(int(line[1].split("-")[1]))
                    
    return partition_list, placement_name_list

def send_data(socket, data, current_tier_ind, agent, detailed_flag):
    if detailed_flag==True: print("tier"+str(current_tier_ind)+"^"+"snd1"+"^", time.time())

    data[-1].append([agent, current_tier_ind,'snd', time.time()])
    data = pickle.dumps(data)

    # print("sent data size ", len(data), flush=True)
    if detailed_flag==True: print("tier"+str(current_tier_ind)+"^"+"snd2"+"^", time.time())

    socket.send(data)

    if detailed_flag==True: print("tier"+str(current_tier_ind)+"^"+"snd3"+"^", time.time())

def wait_for_data(socket, current_tier_ind, agent, detailed_flag):
    data = socket.recv()

    if detailed_flag==True: print("tier"+str(current_tier_ind)+"^"+"rcv1"+"^", time.time())

    # print("rcv sdata len ", len(data))
    data = pickle.loads(data)
    

    data[-1].append([agent, current_tier_ind, 'rcv', time.time()])

    if detailed_flag==True: print("tier"+str(current_tier_ind)+"^"+"rcv2"+"^", time.time())

    return data

def run_command(cmd):
    res = sp.run(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, text=True)

    if (res.stderr != ""):
        sys.exit("error_command: " + cmd + "  ->   " + res.stderr)

    return res.stdout

def run_command_ssh(ssh, cmd):

    stdin, stdout, stderr = ssh.exec_command(cmd)

    timeout = 3
    endtime = time.time() + timeout
    while not stdout.channel.eof_received:
        time.sleep(1)
        if time.time() > endtime:
            stdin.channel.close()
            stdout.channel.close()
            break

    return stdout.readlines()

def set_ssh_sftp (key_name, describe_ins_list):
    # set up ssh and sftp to instances 
    key = paramiko.RSAKey.from_private_key_file(key_name+".pem")
    ssh_list = []
    sftp_list = []

    for i in range(len(describe_ins_list)):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        un="ec2-user" if i==0 else "ubuntu"
        ssh.connect(hostname=describe_ins_list[i]['public_ip'], username=un, pkey=key)
        sftp = ssh.open_sftp()
        ssh_list.append(ssh)
        sftp_list.append(sftp)

    return ssh_list, sftp_list

def print_file(file, s):
    file.write(s)
    file.write('\n')
    file.flush()

def launch_aws_instances(number_of_tiers, ami, count_of_tiers, type_of_tiers, \
    subnet, security_group, key_name, filter_name):

    for i in range(number_of_tiers):
        name = "tier"+str(i)+"_"+filter_name
        used_ami = ami[0] if i==0 else ami[1]
        run_command(f"aws ec2 run-instances --image-id {used_ami} --count {count_of_tiers[i]}\
            --instance-type {type_of_tiers[i]} --subnet-id {subnet}\
            --security-group-ids {security_group} --key-name {key_name}\
            --tag-specifications 'ResourceType=instance,Tags=[{{Key=Name,Value={name}}}]'")
        print(f"*** {name} is launched ***")

    time.sleep(10) 
    describe_ins_list = describe_instance_cmd(key_name, filter_name)
    for ins in describe_ins_list:
        run_command(f"aws ec2 modify-instance-attribute \
            --instance-id {ins['id']} --no-source-dest-check")

def start_aws_instances(describe_ins_list):
    for ins in describe_ins_list:
        if ins['state'] == 'stopped':
            run_command(f"aws ec2 start-instances --instance-ids {ins['id']}")

def wait_until_running(number_of_tiers, key_name, filter_name):
    flag = [False for i in range(number_of_tiers)]
    
    while all(flag) == False:
        print("waiting for instances to be ready ...")
        describe_ins_list = describe_instance_cmd(key_name, filter_name)
        for i, ins in enumerate(describe_ins_list):
            if ins['state'] == 'running':
                flag[int(ins['name'][4])] = True

def describe_instance_cmd(key_name, filter_name):
    json_str = run_command(f"aws ec2 describe-instances \
    --filters 'Name=key-name,Values={key_name}'  \
        --query 'Reservations[*].Instances[*].{{name:Tags[0].Value, id:InstanceId, \
            private_ip:PrivateIpAddress, public_ip:PublicIpAddress, public_dns:PublicDnsName, \
                state:State.Name}}' --output json")

    ins_list =  json.loads(json_str)
    ins_list = [ins[0] for ins in ins_list if (ins[0]['state'] != 'terminated' and filter_name in ins[0]['name'])]
    return sorted(ins_list, key=lambda d: list(d['name']))

def edit_ssh_config(key_name, filter_name):
    ind = 0
    file_address = '/Users/maryamebrahimi/.ssh/config'
    describe_ins_list = describe_instance_cmd(key_name, filter_name)
    print("in edit     ", describe_ins_list)
    with open(file_address, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if 'HostName ec2' in line and ind<len(describe_ins_list):
            lines[i] = 'HostName '+describe_ins_list[ind]['public_dns']+'\n'
            ind += 1
    
    with open(file_address, 'w') as file:
        for line in lines:
            file.write(line)

def plot_with_mean(data, label, tail_length):
    end = len(data)
    plt.plot(range(0, end), data, label=label+" (avg:"+str(np.round(np.mean(data[len(data)-tail_length:]), 3))+")")
    


# get the partitioning list
# if it's final mode, from the available weights, if it's full mode, from combination function
def load_partitioning_list(full_or_final ,model, dataset, tier_number):
    final_weights_folder = ('ins_folder/weights')
    partitioning_list_exp = []
    # if full_or_final == 'final':
        # # read all files in the folder
        # onlyfiles = [f for f in listdir(final_weights_folder) if isfile(join(final_weights_folder, f))]
        # for file_name_orig in onlyfiles:
        #     if model in file_name_orig and dataset in file_name_orig:
        #         partitioning = file_name_orig[file_name_orig.find("[")+1:file_name_orig.find("]")]
        #         partitioning = [int(a) for a in partitioning.split(",") if len(a)!=0]
        #         if len(partitioning) == tier_number:
        #             partitioning_list_exp.append(partitioning)
        

    # if full_or_final == 'full':
        # branch_number = model_info_dict[model]['branch_number']
        # for pp in itertools.combinations(range(1, branch_number+1), tier_number):
        #     if pp[-1] < 7:
        #         continue
        #     if pp[0] < 3:
        #         continue
        #     if pp[0]%2 != 0:
        #         continue
        #     if len(pp)>2 and pp[1]%2 != 0:
        #         continue

            # partitioning_list_exp.append(list(pp))
        # partitioning_list_exp =  [[5, 7], [5, 10], [6, 7], [6, 10], [7, 8], [7, 10], [9, 10]]
        # partitioning_list_exp = [[7], [8], [9], [10]]
        # effb0
        # partitioning_list_exp = [[7], [8]]
        # new motiv resnet110 experiments with final model
    partitioning_list_exp = [[55]]
    return partitioning_list_exp


def read_exp_results(exp_result_folder):

    # total list
    total_list = [] #[source_wait, input_rate, threshold, sample_counter, hwm, buf, 
                    # bw_list, latency_list, 
                    # source, destination, dataset, model, model_mode,
                    # partitioning, placement,
                    # stablization_duration, ruined_flag, 
                    # throughput_tail,
                    # e2e_list, e2e_tail_mean, e2e_tail_std, e2e_tail_p95, 
                    # sum_communications_list, sum_communications_tail_mean, sum_communications_tail_std, sum_communications_tail_95,
                    # sum_computations_list, sum_computations_tail_mean, sum_computations_tail_std, sum_computations_tail_p95
                    # all_communications_lists, all_computations_lists, (for more detailed analysis)
                    # accuracy, loss
                    # exit_rate_all, exit_rate_tail]



    # read all files in the exp_result_folder
    onlyfiles = [f for f in listdir(exp_result_folder) if isfile(join(exp_result_folder, f))]
    # print("onlyfiles--------------", onlyfiles)
    for file_counter, file_name_orig in enumerate(onlyfiles):
        if 'dst-' not in file_name_orig:
            continue

        # read all file names and extract independent variables from file name
        file_name = file_name_orig.strip('dst').strip('.txt').split("-")[1:]
        print(file_name)

        source_wait = int(file_name[0][1:])
        input_rate = np.round(1000/source_wait, 2)
        threshold_num = int(file_name[1][1:])
        sample_counter = int(file_name[2][6:])
        hwm = 100
        buf = 10000000
  

        
        bw_list = [int(a) for a in file_name[3][3:-1].split(',')]
        latency_list = [int(a) for a in file_name[4][4:-1].split(',')]

        placement = [int(a[4]) for a in file_name[5][6:-1].split(',')]
        source_tier = int(file_name[6][7:-1][4])
        destination_tier = int(file_name[7][12:-1][4])
        dataset = file_name[8][8:-1]
        model = file_name[9][6:-1]
        model_mode = file_name[10][5:-1]

        # number of last samples to consider for latencies and throughput
        # I expect the tail part to be stable for not ruined experiments
        tail_length = sample_counter-100
        # very special case
        # if model == 'ResNet20' and destination_tier==2 and placement==[1]:
        #     print("SSssssssssssssssssssssssssssssssssssssssssssssss")
        #     tail_length = 100

        file = open(f"{exp_result_folder}/{file_name_orig}")
        lines = file.readlines()
        line_list = []

        # get all the relevant lines and store the in line_list
        # remove extra lines (for example when detailed flag is true)
        # the lensth of what remains is sample_counter * threshold_num * possible_partitioning_num
        for line in lines:
            if 'tier' in line or 'len' in line:
                continue
            line_list.append(json.loads(line))

        possible_partitioning_num = int(len(line_list)/(sample_counter*threshold_num))


        # seperating experiments based on partitioning and threshold
        for partitioning_ind in range(possible_partitioning_num):
            for threshold_ind in range(threshold_num):
                
                start_ind = (partitioning_ind * (threshold_num * sample_counter)) + (threshold_ind * sample_counter)
                end_ind = (partitioning_ind * (threshold_num * sample_counter)) + ((threshold_ind+1) * sample_counter)
                current_lines = line_list[start_ind: end_ind]

                partitioning = current_lines[0][1]
                partial_flag = False
                if partitioning[-1][-1] != model_info_dict[model]['branch_number']:
                    partial_flag = True
                threshold = current_lines[0][2]

                src_agent_list = []
                inf_agent_list = []
                dst_agent_list = []
                acc_list = []
                loss_list = []
                
                print("partitioning->", partitioning, " threshold->", threshold)
                num_partitions = len(partitioning)
                for current_line in current_lines:
                    acc_list.append(current_line[3][1])
                    loss_list.append(current_line[3][0])

                    for lat in current_line[4]:

                        # source send timestamp
                        if 's' in lat[0] and 'snd' in lat[2]:
                            src_agent_list.append(lat[3])

                        # destination receive timestamp
                        elif 'd' in lat[0] and 'rcv' in lat[2]:
                            dst_agent_list.append(lat[3])

                        # inference receive and send timestamps
                        elif 'i0' in lat[0] and 'rcv' in lat[2]:
                            inf_agent_list.append([lat[3]])
                        else:
                            inf_agent_list[-1].append(lat[3])

                # *1000 to convert to milisec
                src_agent_list = np.array(src_agent_list)*1000
                inf_agent_list = np.array([np.array(a) for a in inf_agent_list], dtype=object)*1000
                dst_agent_list = np.array(dst_agent_list)*1000
    
                # end-to-end latency 
                e2e_list = dst_agent_list - src_agent_list
                # first and last communication latency
                src_inf0 = list(np.array([inf[0] for inf in inf_agent_list]) - src_agent_list)
                inf_dst = list(dst_agent_list - np.array([inf[-1] for inf in inf_agent_list]))

                # making communication and computation latency lists (for all types)
                all_communications_lists = []
                all_computations_lists = []
                max_comp =  len(placement)
                for sample in range(sample_counter):
                    sample_comp = []
                    sample_comm = []

                    sample_comm.append(src_inf0[sample])
                    for i in range(0, len(inf_agent_list[sample]), 2):
                        sample_comp.append (inf_agent_list[sample][i+1] - inf_agent_list[sample][i])

                        if i/2 < (len(inf_agent_list[sample])/2)-1:
                            sample_comm.append(inf_agent_list[sample][i+2] - inf_agent_list[sample][i+1])

                    sample_comm.append(inf_dst[sample])

                    all_communications_lists.append(sample_comm)
                    all_computations_lists.append(sample_comp)

                # there's zero latency for samples the don't go all the way until the end
                # also calculating exit rate in the loop
                exit_rate_all = np.array([0 for a in range(max_comp)])
                exit_rate_tail = np.array([0 for a in range(max_comp)])
                accuracy_by_branch = np.array([0 for a in range(max_comp)])
                accuracy_by_branch_noee = np.array([0 for a in range(max_comp)])
                for sample in range(sample_counter):
                    # for exit rate
                    exit_rate_all[len(all_computations_lists[sample])-1] += 1
                    accuracy_by_branch[len(all_computations_lists[sample])-1] += acc_list[sample]
                    
                    if sample > sample_counter-tail_length:
                        exit_rate_tail[len(all_computations_lists[sample])-1] += 1
                    while len(all_communications_lists[sample]) < max_comp+1:
                        all_communications_lists[sample].insert(-1, 0)
                    while len(all_computations_lists[sample]) < max_comp:
                        all_computations_lists[sample].append(0)
                exit_rate_tail = exit_rate_tail/sum(exit_rate_tail)
                exit_rate_all = exit_rate_all/sum(exit_rate_all)
                accuracy_by_branch = accuracy_by_branch/dataset_info_dict[dataset]['test_size']
                accuracy_by_branch_noee = [np.round(a/(b+0.00001),6) for a,b in zip(accuracy_by_branch, exit_rate_all)]

                all_communications_lists = np.array(all_communications_lists)
                all_computations_lists = np.array(all_computations_lists)

                sum_communications_list = np.sum(all_communications_lists, axis=1)
                sum_computations_list = np.sum(all_computations_lists, axis=1)


                # calculating the tail mean, std and percentile of all latencies
                e2e_tail_mean = np.mean(e2e_list[sample_counter-tail_length:])
                e2e_tail_std = np.std(e2e_list[sample_counter-tail_length:])
                e2e_tail_p95 = np.percentile(e2e_list[sample_counter-tail_length:], 95)

                sum_communications_tail_mean = np.mean(sum_communications_list[sample_counter-tail_length:])
                sum_communications_tail_std = np.std(sum_communications_list[sample_counter-tail_length:])
                sum_communications_tail_p95 = np.percentile(sum_communications_list[sample_counter-tail_length:], 95)

                sum_computations_tail_mean = np.mean(sum_computations_list[sample_counter-tail_length:])
                sum_computations_tail_std = np.std(sum_computations_list[sample_counter-tail_length:])
                sum_computations_tail_p95 = np.percentile(sum_computations_list[sample_counter-tail_length:], 95)

                
                
                # throughput, number of processed samples per sec
                throughput_tail = tail_length/((dst_agent_list[-1]-dst_agent_list[sample_counter-tail_length])/1000) # 1000 is to change back ms to s


                # accuracy of the experiment
                accuracy = np.mean(acc_list)
                loss = np.mean(loss_list)


                total_list.append([source_wait, input_rate, threshold, sample_counter, hwm, buf,\
                                bw_list, latency_list,\
                                source_tier, destination_tier, dataset, model, model_mode,\
                                placement, partitioning, partial_flag, num_partitions,\
                                throughput_tail,\
                                e2e_list, e2e_tail_mean, e2e_tail_std, e2e_tail_p95,\
                                sum_communications_list, sum_communications_tail_mean, sum_communications_tail_std, sum_communications_tail_p95,\
                                sum_computations_list, sum_computations_tail_mean, sum_computations_tail_std, sum_computations_tail_p95,\
                                all_communications_lists, all_computations_lists,\
                                accuracy, loss, accuracy_by_branch, accuracy_by_branch_noee,\
                                exit_rate_all, exit_rate_tail])


    total_df = pd.DataFrame(total_list, columns = ['source_wait', 'input_rate', 'threshold', 'sample_counter', 'hwm', 'buf',\
                                                    'bw_list', 'latency_list',\
                                                    'source_tier', 'destination_tier', 'dataset', 'model', 'model_mode',\
                                                    'placement', 'partitioning', 'partial_flag', 'num_partitions',\
                                                    'throughput_tail',\
                                                    'e2e_list', 'e2e_tail_mean', 'e2e_tail_std', 'e2e_tail_p95',\
                                                    'sum_communications_list', 'sum_communications_tail_mean', 'sum_communications_tail_std', 'sum_communications_tail_p95',\
                                                    'sum_computations_list', 'sum_computations_tail_mean', 'sum_computations_tail_std', 'sum_computations_tail_p95',\
                                                    'all_communications_lists', 'all_computations_lists',\
                                                    'accuracy', 'loss', 'accuracy_by_branch', 'accuracy_by_branch_noee', \
                                                    'exit_rate_all', 'exit_rate_tail'])




    ###### add extimated exit rates to the total_df
    total_df = add_estimated_exitrate_validation(total_df)

    ###### add estimated accuracy to the total_df
    total_df = add_estimated_accuracy_validation(total_df)

    ###### add estimated computation latency to the total_df
    total_df = add_estimated_computation(total_df)

    ###### add pickled data size to the total_df
    total_df = add_estimated_pickle_size(total_df)

    ###### add estimated communiction latency to the total_df (propagation + tranmission)
    total_df = add_estimated_communication(total_df)

    ###### add estimated e2e latency to the total_df (communication + computation + pickling)
    total_df = add_estimated_e2e_latency(total_df)


    ###### recheck the ruined flag with new estimated communication latency
    # because the old method cannot catch experiments with early exit 
    # there is unstability and the reason is not being ruined
    total_df = add_ruined_flag(total_df)


    save_df (total_df, exp_result_folder)
    return total_df 


def save_df (total_df, exp_result_folder):
    no_csv_cols = ['e2e_list', 'sum_communications_list', 'sum_computations_list', 'all_communications_lists', 'all_computations_lists' ]
    total_df.to_pickle(f'{exp_result_folder}/total_df.pkl')
    total_df.to_csv(f'{exp_result_folder}/total_df.csv', index=False, columns = [a for a in total_df.columns if a not in no_csv_cols])


def add_estimated_exitrate_validation(total_df):
    exits_corrects_partitionwise_dict_dict = {}


    for (model_name, dataset_name) in used_model_ds:
        exits_corrects_partitionwise_dict_dict[model_name, dataset_name] = get_profiled_exits_corrects(model_name, dataset_name)
    
    df_estimated_exitrate_partitionwise = []
    df_estimated_stayrate_partitionwise = []



    for index, row in total_df.iterrows():
        # get the profiled exitrate based on threshold
        
        exits_corrects_partitionwise_dict = exits_corrects_partitionwise_dict_dict[row['model'], row['dataset']]


        # calcualte the estiated exit rate and stary rate
        exitrate_partitionwise, stayrate_partitionwise = estimate_exitrate_partitionwise_profiled (row['partitioning'], row['threshold'], exits_corrects_partitionwise_dict, dataset_info_dict[row['dataset']]['validation_size'])


        df_estimated_exitrate_partitionwise.append(exitrate_partitionwise)
        df_estimated_stayrate_partitionwise.append(stayrate_partitionwise)


    total_df['estimated_exit_rate'] = pd.Series(df_estimated_exitrate_partitionwise, index=total_df.index)
    total_df['estimated_stay_rate'] = pd.Series(df_estimated_stayrate_partitionwise, index=total_df.index)

    return total_df


def add_estimated_accuracy_validation(total_df):
    exits_corrects_partitionwise_dict_dict = {}
    

    for (model_name, dataset_name) in used_model_ds:
        exits_corrects_partitionwise_dict_dict[model_name, dataset_name] = get_profiled_exits_corrects(model_name, dataset_name)
    
    df_estimated_accuracy_partitionwise = []

    for index, row in total_df.iterrows():

        exits_corrects_partitionwise_dict = exits_corrects_partitionwise_dict_dict[row['model'], row['dataset']]

        acc = estimate_accuracy_partitionwise_profiled (row['partitioning'], row['estimated_exit_rate'], row['threshold'], exits_corrects_partitionwise_dict)

        df_estimated_accuracy_partitionwise.append(acc)

    total_df['estimated_accuracy'] = pd.Series(df_estimated_accuracy_partitionwise, index=total_df.index)

    return total_df


def add_estimated_computation(total_df):
    
    # get device tier map
    number_of_tiers, type_of_tiers, _, _, _, _, _ = get_topology_info('ins_folder/files/topology_file.txt')
    tier_device_map = dict(zip(range(number_of_tiers), type_of_tiers))

    # lookup tables for computation latencies for different devices
    # add new devices here ???
    lookup_table_dict = {}
    for dev in used_devices:
        lookup_table_dict[dev] = get_lookup_tables(dev)


    model_dict = {}
    intermediate_shape_dict = {}

    for (model_name, dataset) in used_model_ds:

        branch_number = model_info_dict[model_name]['branch_number']
        selected_exits = np.array([a+1 for a in range(branch_number)])

        with open('/Users/maryamebrahimi/Desktop/AWS/ee_aws/all_profiling_results/intermediate_dict.pkl', 'rb') as f:
            intermediate_shape_dict = pickle.load(f)
    
    # going through all experiment to estimate computation latencies
    df_dnn_comp_partitionwise_plus_est_er = []
    df_pickle_comp_est_er = []

    for index, row in total_df.iterrows():
        # print("-------------------------------------------------------")

        dnn_comp_partitionwise_plus_est_er = []
        pickle_comp_est_er = []
        

        estimated_stayrate_list = row['estimated_stay_rate']
        estimated_exitrate_list = row['estimated_exit_rate']
        row_source = row['source_tier']
        row_placement = row['placement']
        row_model = row['model']


        # source to inference0 pickle latency
        source_pickle_dict = lookup_table_dict[tier_device_map[row_source]]['pickle'][row_model]
        inf0_pickle_dict = lookup_table_dict[tier_device_map[row_placement[0]]]['pickle'][row_model]
        pickle_latency = source_pickle_dict[0][2]+inf0_pickle_dict[0][3]
        
        pickle_comp_est_er.append(pickle_latency)
        
        # print("source to inf1 pickle   ", "num channel: ", source_channels_num, "input size: ", source_input_size, "latency: ", pickle_latency)

        # getting the computation latency for each partition
        for i, (curr_partition, curr_placement) in enumerate(zip(row['partitioning'], row['placement'])):
            # print("      curr_partition: ", curr_partition, "curr_placement: ", curr_placement)
            # for dnn latency, PARTITION_WISE_PLUS (2)
            dnn_latency = ResNet_partition_latency_partitionwise_plus (curr_partition, row['model'], row['dataset'], lookup_table_dict[tier_device_map[curr_placement]])
            # print("      dnn_latency: ", dnn_latency)
            dnn_comp_partitionwise_plus_est_er.append(dnn_latency*(estimated_stayrate_list[i]+estimated_exitrate_list[i]))


            # for pickle latency
            if i != len(row['placement'])-1:
                next_placement = row['placement'][i+1]

                curr_pickle_dict = lookup_table_dict[tier_device_map[curr_placement]]['pickle'][row_model]
                next_pickle_dict = lookup_table_dict[tier_device_map[next_placement]]['pickle'][row_model]
                pickle_latency = curr_pickle_dict[curr_partition[-1]][2] + next_pickle_dict[curr_partition[-1]][3]
                    

                pickle_comp_est_er.append(pickle_latency*estimated_stayrate_list[i])


        df_dnn_comp_partitionwise_plus_est_er.append(sum(dnn_comp_partitionwise_plus_est_er))
        df_pickle_comp_est_er.append(sum(pickle_comp_est_er))


    total_df['estimated_computation_latency'] = pd.Series(df_dnn_comp_partitionwise_plus_est_er, index=total_df.index)
    total_df['estimated_pickle_latency'] = pd.Series(df_pickle_comp_est_er, index=total_df.index)

    return total_df


# this function build the pickled object that is sent during the experiments, as close as possible
def add_estimated_pickle_size(total_df):


    model_dict = {}
    intermediate_data_dict = {}


    for (model_name, dataset) in used_model_ds:

        branch_number = model_info_dict[model_name]['branch_number']
        selected_exits = np.array([a+1 for a in range(branch_number)])
        if 'ResNet' in model_name:
            input_size = dataset_info_dict[dataset]['input_size']
            num_channels = dataset_info_dict[dataset]['num_channels']
            model_dict[(model_name, dataset)] = ResNet_CIFAR_model_final(model_info_dict[model_name]['model_n'], model_info_dict[model_name]['branch_number'], dataset_info_dict[dataset]['num_class'], selected_exits, device).to(device)
            
             # pass an input through the model to get detailed intermediate shape
            input_tensor = torch.rand(input_size*input_size*num_channels).view(num_channels, input_size, input_size).unsqueeze(dim=0).detach().clone()
            _, intermediate_list = model_dict[(model_name, dataset)](input_tensor)
            intermediate_data_dict[(model_name, dataset)] = intermediate_list
            
        if 'EfficientNet' in model_name:
            input_size = dataset_info_dict[dataset][f"input_size_{model_name[-2:]}"]
            num_channels = dataset_info_dict[dataset]['num_channels']
            model_dict[(model_name, dataset)] = EfficientNet_final(model_info_dict[model_name]['dropout'], model_info_dict[model_name]['width_mult'], model_info_dict[model_name]['depth_mult'], model_info_dict[model_name]['norm_layer'], dataset_info_dict[dataset]['num_class'], selected_exits).to(device)
            
             # pass an input through the model to get detailed intermediate shape
            input_tensor = torch.rand(input_size*input_size*num_channels).view(num_channels, input_size, input_size).unsqueeze(dim=0).detach().clone()
            _, intermediate_list = model_dict[(model_name, dataset)](input_tensor)
            intermediate_data_dict[(model_name, dataset)] = intermediate_list

        if 'VGGish' in model_name:
            ### loading a wav sample
            wav_data, _ = sf.read('/Users/maryamebrahimi/Desktop/AWS/ee_aws/ins_folder/ins_profiling/sample.wav', dtype='int16')

            ### loading the full configuration model
            model_dict[(model_name, dataset)] = VGGish_final(selected_exits, device).to(device)

            ### passing an input through the model to get detailed intermediate shape
            _, intermediate_list = model_dict[(model_name, dataset)](torch.tensor(wav_data), device)
            intermediate_data_dict[(model_name, dataset)] = intermediate_list

       


    df_pickle_size = []
    for index, row in total_df.iterrows():
        row_model_name= row['model']
        row_dataset = row['dataset']
        row_partitioning = row['partitioning']
        num_class  = dataset_info_dict[row_dataset]['num_class']


        intermediate_list =  intermediate_data_dict[(row_model_name, row_dataset)]
        pickled_size = []

        label_tensor = torch.randint(low=1, high=num_class, size=(1,)).detach().clone()

        ### input (source) object
        obj = ["flag_src0", 1000, intermediate_list[0], label_tensor, 0.5, row_partitioning, []]
        obj[-1].append(["s"+str(0), 2,'snd', time.time()])

        pickled_size.append(len(pickle.dumps(obj))+100)

        for i, part in enumerate(row_partitioning):
            # this one goes to destination agent
            if i == len(row_partitioning)-1:
                obj[-1].append(["s"+str(i), 2,'rcv', time.time()])
                obj = ["flag_exit", 1000, [1.111, 1], 0.5, row_partitioning, obj[-1]]
                obj[-1].append(["s"+str(i), 2,'snd', time.time()])

            else:
                exit_ind = part[-1]
                intermediate_tensor = intermediate_list[exit_ind]
                
                obj[-1].append(["s"+str(i), 2,'rcv', time.time()])
                obj = ["flag_inter"+str(i), 1000, intermediate_tensor, label_tensor, 0.5, row_partitioning, obj[-1]]
                obj[-1].append(["s"+str(i), 2,'snd', time.time()])

            pickled_size.append(len(pickle.dumps(obj))+100)
        
        df_pickle_size.append(pickled_size)
    total_df['estimated_pickle_size'] = pd.Series(df_pickle_size, index=total_df.index)
    return total_df


def add_estimated_communication(total_df):
    propagation_latency_list_est_er = []
    transmission_latency_list_est_er = []

    for index, row in total_df.iterrows():
        # print("----------------------------------------------------------------")
        
        row_src = row['source_tier']
        row_dest = row['destination_tier']
        row_placement = row['placement']
        row_latency = row['latency_list']
        estimated_stayrate_list = row['estimated_stay_rate']
        estimated_exitrate_list = row['estimated_exit_rate']
        row_pickle_size = row['estimated_pickle_size']
        row_bw = np.array(row['bw_list'])

        prop_est_er = 0
        trans_est_er = 0

        # if src and inf0 are not on same tier, add the propagation/transmission latency between them
        if row_src <= row_placement[0]:
            start_ind = row_src
            end_ind = row_placement[0]
        else:
            start_ind = row_placement[0]
            end_ind = row_src
        
        
        prop_est_er += sum(row_latency[start_ind:end_ind])
        
        # tranmission latency, especial case for multihop case and large object(>9000)
        if (len(row_bw[start_ind:end_ind])>1)  and (row_pickle_size[0]>9000):

            BWs = row_bw[start_ind:end_ind]
            MTU = 9000 
            Ls = [(MTU*8)/(bw*1000)  for bw in BWs]
            max_L = max(Ls)
            N = row_pickle_size[0]/MTU

            trans_est_er += (N*max_L + sum([a for a in Ls if a!=max_L]))
            # print("way 1 special  ", (N*max_L + sum([a for a in Ls if a!=max_L])))

        else: # normal case for transmission latency
            trans_est_er += ( (row_pickle_size[0]*8/1000) * sum(1/row_bw[start_ind:end_ind]) )
            # print("way 2 normal ",  ( (row_pickle_size[0]*8/1000) * sum(1/row_bw[start_ind:end_ind]) ))
        
        



        # intermediate propagation/transmission latencies
        for i, current_tier in enumerate(row_placement):
            
            next_tier = -1
            if i != len(row_placement)-1:
                next_tier = row_placement[i+1]
            
            # those that exit -> to destination
            # check whether the current tier is destination tier -> no latency then
            if current_tier <= row_dest:
                start_ind = current_tier
                end_ind = row_dest
            else:
                start_ind = row_dest
                end_ind = current_tier

            prop_est_er += (estimated_exitrate_list[i] * sum(row_latency[start_ind:end_ind]))

            # no special case for tranmission latency here, size of the packet is smaller than 9000 (mtu)
            trans_est_er += (estimated_exitrate_list[i] * (row_pickle_size[-1]*8/1000) * sum(1/row_bw[start_ind:end_ind]))
            # print("way 3 normal ", estimated_exitrate_list[i] * (row_pickle_size[-1]*8/1000) * sum(1/row_bw[start_ind:end_ind]))

            # those that stay -> to next inf
            # we don't care about last element in stayrate list
            if next_tier != -1:
                if current_tier <= next_tier:
                    start_ind = current_tier
                    end_ind = next_tier
                else:
                    start_ind = next_tier
                    end_ind = current_tier

                prop_est_er += (estimated_stayrate_list[i]*sum(row_latency[start_ind:end_ind]))
                
                #  special case for transmission latency
                if (len(row_bw[start_ind:end_ind])>1)  and (row_pickle_size[i+1]>9000):



                    BWs = row_bw[start_ind:end_ind]
                    MTU = 9000 
                    Ls = [(MTU*8)/(bw*1000)  for bw in BWs]
                    max_L = max(Ls)
                    N = row_pickle_size[i+1]/MTU

                    trans_est_er += ( estimated_stayrate_list[i] *(N*max_L + sum([a for a in Ls if a!=max_L])) )
                    # print("way 4 special ", estimated_stayrate_list[i] *(N*max_L + sum([a for a in Ls if a!=max_L])))

                else:  # normal case for transmission latency
                    trans_est_er += (estimated_stayrate_list[i] * (row_pickle_size[i+1]*8/1000) * sum(1/row_bw[start_ind:end_ind]))
                    # print("way 5 normal ", estimated_stayrate_list[i] * (row_pickle_size[i+1]*8/1000) * sum(1/row_bw[start_ind:end_ind]))

        
        
        # print(prop_est_er, trans_est_er)
        propagation_latency_list_est_er.append(prop_est_er)
        transmission_latency_list_est_er.append(trans_est_er)


    total_df['estimated_communication_latency'] = pd.Series(np.array(propagation_latency_list_est_er)+np.array(transmission_latency_list_est_er), index=total_df.index)
    return total_df


def add_estimated_e2e_latency(total_df):
    e2e_list = []

    for index, row in total_df.iterrows():
        row_comm = row['estimated_communication_latency']
        row_comp = row['estimated_computation_latency']
        row_pickle = row['estimated_pickle_latency']
        e2e_list.append(row_comm+row_comp+row_pickle)


    total_df['estimated_e2e_latency'] = pd.Series(np.array(e2e_list), index=total_df.index)
    return total_df

def add_ruined_flag(total_df):
    df_ruined_flag = []
    df_stabilization_duration = []

    for index, row in total_df.iterrows():
        row_estimated_communication = row['estimated_communication_latency']
        row_experiment_communication = row['sum_communications_tail_mean']
        row_experiment_e2e = row['e2e_tail_mean']
        row_estimated_e2e = row['estimated_e2e_latency']
        row_threshold = row['threshold']
        row_source_wait = row['source_wait']
        row_sample_counter = row['sample_counter']
        row_sum_comm = row['sum_communications_list']
        
        if  np.abs(row_experiment_e2e-row_estimated_e2e) > 100:
            df_ruined_flag.append(True)

            df_stabilization_duration.append(row_sample_counter-1)
        else:

            df_ruined_flag.append(False)
            
            flag_temp = False
            moving_comm_avg_window = 10
            for i in range(row_sample_counter-moving_comm_avg_window):
                if  np.abs(np.mean(row_sum_comm[i:i+moving_comm_avg_window]) - row_estimated_communication) < 10 :
                    df_stabilization_duration.append(i+moving_comm_avg_window)
                    flag_temp = True
                    break
            if flag_temp == False:
                print("HHHHHERE")
                df_stabilization_duration.append(row_sample_counter-1)

    total_df['ruined_flag'] = pd.Series(df_ruined_flag, index=total_df.index)
    total_df['stabilization_duration'] = pd.Series(df_stabilization_duration, index=total_df.index)
    return total_df


### get profiled information in lookup tables
def get_lookup_tables(device_name):
    lookup_tables = {}
    onlyfiles = [f for f in listdir(f'all_profiling_results/{device_name}/') if isfile(join(f'all_profiling_results/{device_name}', f))]
    for file_name in onlyfiles:

        file = open(f"all_profiling_results/{device_name}/"+file_name, 'rb')
        
        if 'pickle' in file_name and 'pkl' in file_name:
            lookup_tables['pickle'] = pickle.load(file)

        if 'conv' in file_name:
            lookup_tables['conv'] = pd.read_json(json.load(file))

        if 'bn' in file_name:
            lookup_tables['bn'] = pd.read_json(json.load(file))

        if 'relu' in file_name:
            lookup_tables['relu'] = pd.read_json(json.load(file))

        if 'pool' in file_name:
            lookup_tables['pool'] = pd.read_json(json.load(file))

        if 'flat' in file_name:
            lookup_tables['flatten'] = pd.read_json(json.load(file))

        if 'dense' in file_name:
            lookup_tables['dense'] = pd.read_json(json.load(file))

        if 'add' in file_name:
            lookup_tables['add'] = pd.read_json(json.load(file))

        if 'exit_block' in file_name:
            lookup_tables['exit_block'] = pd.read_json(json.load(file))

        if 'resnet_block' in file_name:
            lookup_tables['resnet_block'] = pd.read_json(json.load(file))

        if 'init_block' in file_name:
            lookup_tables['init_block'] = pd.read_json(json.load(file))

        if 'partition' in file_name:

            model_name = file_name.split('_')[-2]
            dataset = file_name.split('_')[-1].split('.')[0]
            # print("^^^^^^^  ", model_name, dataset, device_name, "   ^^^^^^^^")
            lookup_tables[f'partition_{model_name}_{dataset}'] = pd.read_json(json.load(open(f"all_profiling_results/{device_name}/"+file_name, 'r')))        


    return lookup_tables



### get profiled exit rate/ accuracy for each model and dataset
def get_profiled_exits_corrects (model_name, dataset):
    exits_corrects_partitionwise_dict = pickle.load(open(f"all_profiling_results/accuracy_exitrate/profiling_exitrate_accuracy_partitionwise_model[{model_name}]_dataset[{dataset}]_validation.txt", 'rb'))
    # print("^^^^^^^   ", exits_corrects_partitionwise_dict, "   ^^^^^^^^")
    return exits_corrects_partitionwise_dict



### PARTITION_WISE computation estimation
def ResNet_partition_latency_partitionwise_plus(partition, model_name, dataset, lookup_tables):
    # print("partition: ", lookup_tables)
    the_df = lookup_tables[f'partition_{model_name}_{dataset}']
    latency = the_df.loc[(the_df['partition_start']==partition[0]) & (the_df['partition_end']==partition[-1])]['latency_partition'].to_numpy()[0]
    return np.max(latency)



### function to estimate exit rate
def estimate_exitrate_partitionwise_profiled (partitioning, threshold, exitrate_dict, validation_size):
    partitioning_exitrate = []
    partitioning_stayrate = []
    partitioning_exitrate_plus = []
    # print("------------------------- partitioning: ", partitioning, "---------------------------")
    # print("exitrate_dict: ", exitrate_dict)
    prev_er = 0
    for partition in partitioning:
        # print("       partition: ", partition)
        new_er = exitrate_dict[f'[1, {partition[-1]}]'][threshold][0]/validation_size - prev_er
        partitioning_exitrate.append(np.round(new_er, 4))
        prev_er = exitrate_dict[f'[1, {partition[-1]}]'][threshold][0]/validation_size

        # print("   new re", new_er, "prev_er", prev_er)
    # for partial executions, every thing that remain exit from last branch
    if sum(partitioning_exitrate) < 0.999:
        partitioning_exitrate[-1] = 1-sum(partitioning_exitrate[:-1])


    # calculating exitrate_plus (alpha_plus) for stayrate
    # stayrate = 1 - exitrate_plus
    current = 0
    for e in partitioning_exitrate:
        current += e
        partitioning_exitrate_plus.append(np.round(current, 5))
    
    partitioning_stayrate = [np.round(1-a, 5) for a in partitioning_exitrate_plus]

    return partitioning_exitrate, partitioning_stayrate



### function to estimate accuracy
def estimate_accuracy_partitionwise_profiled (partitioning, exit_rate_list, threshold, exitrate_accuracy_dict):
    # print("************  ","partitioning: ", partitioning, "exit rate ", exit_rate_list, "threshold ", threshold, "************")
    total_acc = 0
    for i, partition in enumerate(partitioning):
        # print("  threshold1  ", exitrate_accuracy_dict[str([1, partition[-1]])][1], f"    threhold{threshold}  ", exitrate_accuracy_dict[str([1, partition[-1]])][threshold])
        er_acc_list = exitrate_accuracy_dict[str([1, partition[-1]])][1]
        branch_acc = er_acc_list[1]/(er_acc_list[0]+0.000000001)
        total_acc += branch_acc*exit_rate_list[i]
    return total_acc





