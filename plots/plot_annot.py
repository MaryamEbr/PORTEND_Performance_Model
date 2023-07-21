#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import json
import subprocess as sp
import seaborn as sns
import pandas as pd
import torch
import time
import pickle
import random
from os import listdir
from os.path import isfile, join
import sys

sys.path.append('ins_folder/codes')
from helper_functions import read_exp_results, save_df

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# some helper functions for hovering in plot
def update_annot(ind):
    ind = ind["ind"][0]
    pos = sc.get_offsets()[ind]
    annot.xy = pos
    text = f"*partition: {partitionin_list[ind]} *placement: {placement_list[ind]} \n*threshold: {threshold_list[ind]}"

    annot.set_text(text)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

read_again_flag = False
exp_result_folder = "exp_folder/Experiment72"
model_mode = 'full'
if '60' in exp_result_folder or '70' in exp_result_folder:
    dataset = 'ImageNet'
    model_name = 'EfficientNetB0'
if '61' in exp_result_folder or '71' in exp_result_folder:
    dataset = 'CIFAR10'
    model_name = 'ResNet20' 
if '62' in exp_result_folder or '72' in exp_result_folder:
    dataset = 'CIFAR10'
    model_name = 'ResNet110'
    
if '74' in exp_result_folder:
    dataset = 'AudioSet'
    model_name = 'VGGish'




if read_again_flag == True:
    total_df = read_exp_results(exp_result_folder) 
else:
    total_df = pd.read_pickle(f'{exp_result_folder}/total_df.pkl')



row = total_df.loc[(total_df['ruined_flag']==False) &(total_df['model_mode']==model_mode) & (total_df['source_wait']==270) ].sort_values(['partitioning'], kind='mergesort')


# experiment_latency = np.array(row[['e2e_tail_mean']].to_numpy()[:,0])
# estimated_latency = np.array(row[['estimated_e2e_latency']].to_numpy()[:,0])

# experiment_latency = np.array(row[['sum_computations_tail_mean']].to_numpy()[:,0])
# estimated_latency = np.array(row[['estimated_computation_latency']].to_numpy()[:,0])


experiment_latency = np.array(row[['sum_communications_tail_mean']].to_numpy()[:,0])
estimated_latency = np.array(row[['estimated_communication_latency']].to_numpy()[:,0])

partitionin_list = np.array(row[['partitioning']].to_numpy()[:,0])
placement_list = np.array(row[['placement']].to_numpy()[:,0])
threshold_list = np.array(row[['threshold']].to_numpy()[:,0])


fig,ax = plt.subplots(figsize=(9,6))
sc = plt.scatter(experiment_latency, estimated_latency, color='darkgreen')
plt.plot(np.arange(0, 400, 0.01), np.arange(0, 400,0.01), color='black')


# annotation
annot = ax.annotate("", xy=(0,0), xytext=(-120,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

fig.canvas.mpl_connect("motion_notify_event", hover)

# plt.legend()
plt.title(f"model {model_name}, dataset {dataset}")
plt.xlabel('experiment  latency')
plt.ylabel('estimated  latency')
# plt.legend()
plt.grid()
plt.show()