#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import json
import subprocess as sp
import seaborn as sns
import pandas as pd
import torch
import random
from os import listdir
from os.path import isfile, join
import sys
import mplcursors

sys.path.append('ins_folder/codes')
from helper_functions import read_exp_results

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



# folders
exp_result_folder = "exp_folder/Experiment72"
dataset = 'CIFAR10'
model_name = 'ResNet110'

destination_tier = 0
model_mode = 'final'
read_again_flag = False





# read the experiment files from csv file or original txt files
if read_again_flag == True:
    total_df = read_exp_results(exp_result_folder) 
else:
    total_df = pd.read_pickle(f'{exp_result_folder}/total_df.pkl')



rows = total_df.loc[(total_df['ruined_flag']==False)  &
                    (total_df['destination_tier']==destination_tier) & (total_df['model_mode']==model_mode)]

rows_2p_ns = rows.loc[ (rows['partitioning'].astype(str)=='[[1, 20], [21, 55]]')]
rows_2p_ns = rows_2p_ns.loc[rows_2p_ns['e2e_tail_mean']<130]
rows_2p_ns['accuracy'] = 0.9254

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(rows)


# all tier 0
all_tier_0_latency = rows.loc[(rows['placement'].astype(str)=='[0]')&(total_df['partial_flag']==False)]['e2e_tail_mean'].values
all_tier_0_comp_latency = rows.loc[(rows['placement'].astype(str)=='[0]')&(total_df['partial_flag']==False)]['sum_computations_tail_mean'].values
all_tier_0_comm_latency = rows.loc[(rows['placement'].astype(str)=='[0]')&(total_df['partial_flag']==False)]['sum_communications_tail_mean'].values
all_tier_0_accuracy = rows.loc[(rows['placement'].astype(str)=='[0]')&(total_df['partial_flag']==False)]['accuracy'].values

# all tier 1
all_tier_1_latency = rows.loc[(rows['placement'].astype(str)=='[1]')&(total_df['partial_flag']==False)]['e2e_tail_mean'].values
all_tier_1_comp_latency = rows.loc[(rows['placement'].astype(str)=='[1]')&(total_df['partial_flag']==False)]['sum_computations_tail_mean'].values
all_tier_1_comm_latency = rows.loc[(rows['placement'].astype(str)=='[1]')&(total_df['partial_flag']==False)]['sum_communications_tail_mean'].values
all_tier_1_accuracy = rows.loc[(rows['placement'].astype(str)=='[1]')&(total_df['partial_flag']==False)]['accuracy'].values

# all tier 2
all_tier_2_latency = rows.loc[(rows['placement'].astype(str)=='[2]')&(total_df['partial_flag']==False)]['e2e_tail_mean'].values
all_tier_2_comp_latency = rows.loc[(rows['placement'].astype(str)=='[2]')&(total_df['partial_flag']==False)]['sum_computations_tail_mean'].values
all_tier_2_comm_latency = rows.loc[(rows['placement'].astype(str)=='[2]')&(total_df['partial_flag']==False)]['sum_communications_tail_mean'].values
all_tier_2_accuracy = rows.loc[(rows['placement'].astype(str)=='[2]')&(total_df['partial_flag']==False)]['accuracy'].values

# ns
all_tier_0_1_latency_ns = rows_2p_ns.loc[(rows_2p_ns['placement'].astype(str)=='[0, 1]') ]['e2e_tail_mean'].values
all_tier_0_1_accuracy_ns = rows_2p_ns.loc[(rows_2p_ns['placement'].astype(str)=='[0, 1]') ]['accuracy'].values
all_tier_0_2_latency_ns = rows_2p_ns.loc[(rows_2p_ns['placement'].astype(str)=='[0, 2]') ]['e2e_tail_mean'].values
all_tier_0_2_accuracy_ns = rows_2p_ns.loc[(rows_2p_ns['placement'].astype(str)=='[0, 2]') ]['accuracy'].values


# out method
all_tier_latency_ee_1p = rows.loc[(total_df['num_partitions']==1)]['e2e_tail_mean'].values
all_tier_accuracy_ee_1p = rows.loc[(total_df['num_partitions']==1)]['accuracy'].values

all_tier_latency_ee_2p = rows.loc[(total_df['num_partitions']==2)]['e2e_tail_mean'].values
all_tier_accuracy_ee_2p = rows.loc[(total_df['num_partitions']==2)]['accuracy'].values

all_tier_latency_ee_3p = rows.loc[(total_df['num_partitions']==3)]['e2e_tail_mean'].values
all_tier_accuracy_ee_3p = rows.loc[(total_df['num_partitions']==3)]['accuracy'].values


plt.figure(figsize=(9,6))


plt.plot(all_tier_0_latency, all_tier_0_accuracy, '*', ms=15, color='darkred', label='1 partition, tier 0 (edge only)', zorder=10)
plt.plot(all_tier_1_latency, all_tier_1_accuracy, '*', ms=15, color='darkblue', label='1 partition, tier 1', zorder=10)
plt.plot(all_tier_2_latency, all_tier_2_accuracy, '*', ms=15, color='darkgreen', label='1 partition, tier 2 (cloud only)', zorder=10)




plt.plot(all_tier_latency_ee_3p, all_tier_accuracy_ee_3p, 'o', color='teal', label='3 partitions (our method)')
plt.plot(all_tier_latency_ee_2p, all_tier_accuracy_ee_2p, 'o', color='plum', label='2 partitions (our method)')
plt.plot(all_tier_latency_ee_1p, all_tier_accuracy_ee_1p, 'o', color='olive', label='1 partitions (our method)')

plt.plot(all_tier_0_1_latency_ns, all_tier_0_1_accuracy_ns, 'X', ms=12, color='purple', label='2 partitions, no early exit (neurosurgeon)')
# plt.plot(all_tier_0_2_latency_ns, all_tier_0_2_accuracy_ns, 'X', ms=10, color='purple', )

plt.title(f"model {model_name}(final models), dataset {dataset}")
plt.legend(title='placement/partitioning', loc='lower right')
plt.grid()
plt.xlabel('latency (ms)')
plt.ylabel('accuracy')
plt.show()

















plt.figure(figsize=(9,6))


plt.bar(1, all_tier_0_comm_latency , color='orange', width=0.3, label='communication latency')
plt.bar(1, all_tier_0_comp_latency, bottom=all_tier_0_comm_latency, color='brown', width=0.3, label='computation latency')

plt.bar(2, all_tier_2_comm_latency, width=0.3,color='orange')
plt.bar(2, all_tier_2_comp_latency, bottom=all_tier_2_comm_latency,  width=0.3,color='brown')

plt.xticks([1,2,3], ['tier 0', 'tier 1', 'tier 2'])
plt.title(f"model {model_name}, dataset {dataset}")
plt.xlabel('placement (full model)')
plt.ylabel('latency (ms)')

plt.legend()
plt.show()