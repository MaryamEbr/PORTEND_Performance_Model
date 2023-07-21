#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import json
import subprocess as sp
import pandas as pd
import random
from os import listdir
from os.path import isfile, join
import sys


sys.path.append('ins_folder/codes')
from helper_functions import plot_with_mean, read_exp_results

read_again_flag = True 
detailed_plot_flag = True # to show all latency components in the plot or not

#parameters
source_wait = 250
threshold = 0.1
sample_counter = 5000
bw_list = str([10, 400])
latency_list = str([20, 40])
partitioning = str([[1, 5], [6, 10], [11, 15]])
placement = str([0, 1, 2])
model_mode = 'final'
model = 'ResNet110'
dataset = 'CIFAR10'
des_tier = 0


# folders
exp_result_folder = "exp_folder/Experiment74"


color_list = ['black', 'red', 'blue', 'green', 'orange', 'cyan', 'fuchsia', 'yellow', 'pink', 'tan', 'purple',\
    'olive', 'crimson', 'lime', 'slategray', 'tomato', 'indigo', 'teal']


# read the experiment files from csv file or original txt files
if read_again_flag == True:
    total_df = read_exp_results(exp_result_folder) 
else:
    total_df = pd.read_pickle(f'{exp_result_folder}/total_df.pkl')



# filter the total dataframe based on given parameters
# it should remain 1 row at the end
row = total_df.loc[ (total_df['source_wait']==source_wait) & (total_df['threshold']==threshold) & (total_df['destination_tier']==des_tier) &
                            (total_df['sample_counter']==sample_counter) & (total_df['bw_list'].astype(str)==bw_list) &
                            (total_df['latency_list'].astype(str)==latency_list) & (total_df['partitioning'].astype(str)==partitioning) &
                            (total_df['placement'].astype(str)==placement) & (total_df['model']==model) & (total_df['dataset']==dataset) & (total_df['model_mode']==model_mode)]

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(row)
if len(row) != 1:
    print("the size of filtered df is not 1, it's ", len(row))
    exit()



####################################### the individual plot #####################################
plt.figure(figsize=(9,6))
tail_length = sample_counter-50 

e2e_list = row[['e2e_list']].to_numpy()[0][0]
sum_communications_list = row[['sum_communications_list']].to_numpy()[0][0]
sum_computations_list = row[['sum_computations_list']].to_numpy()[0][0]
all_communications_lists = row[['all_communications_lists']].to_numpy()[0][0]
all_computations_lists = row[['all_computations_lists']].to_numpy()[0][0]
# stabilization_duration = row[['stabilization_duration']].to_numpy()[0][0]
e2e_tail_mean = row[['e2e_tail_mean']].to_numpy()[0][0]
e2e_tail_std = row[['e2e_tail_std']].to_numpy()[0][0]
ruined_flag = row[['ruined_flag']].values[0][0]

max_comp = row[['num_partitions']].to_numpy()[0][0]
exit_rate = row[['exit_rate_all']].to_numpy()[0][0]
est_exit_rate = row[['estimated_exit_rate']].to_numpy()[0][0]


# end-to-end 
plot_with_mean(e2e_list, 'end_to_end latency', tail_length)

# sum communications
plot_with_mean(sum_communications_list, 'sum communication latency', tail_length)

# sum computations
plot_with_mean(sum_computations_list, 'sum computation latency', tail_length)


# detailed latency plots
if detailed_plot_flag == True:
    # source to inf0 communication
    plot_with_mean(all_communications_lists[:, 0], 'source to inference0', tail_length)

    for i in range(max_comp):
        # intermediate computations 
        plot_with_mean(all_computations_lists[:, i], f'inference{i} computation', tail_length)

        # intermediate communications
        if i < max_comp-1:
            plot_with_mean(all_communications_lists[:, i+1], f'inference{i} to inference{i+1}', tail_length)

    # last inference to destination
    plot_with_mean(all_communications_lists[:, -1], f'last inference to destination', tail_length)


plt.xlabel('samples')
plt.ylabel('latency(ms)')

plt.title(f'model:{model}({model_mode}) wait:{source_wait} place:{placement} part:{partitioning} bw:{bw_list} lat:{latency_list} \nthresh:{threshold}  exp_exit_rate:{exit_rate} est_exit_rate:{est_exit_rate} ruined:{ruined_flag}')

plt.legend(bbox_to_anchor=(0,1.11), loc="lower left", ncol=2)
plt.grid(visible=True)
plt.subplots_adjust(top=0.7)
plt.show()



