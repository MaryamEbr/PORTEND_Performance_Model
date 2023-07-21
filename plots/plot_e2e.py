#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
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
from sklearn.metrics import r2_score

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 12

sys.path.append('ins_folder/codes')
from helper_functions import read_exp_results

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


import numpy
# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = numpy.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = numpy.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = numpy.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = numpy.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = numpy.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results


read_again_flag = False
exp_result_folder = "exp_folder/Experiment71"
model_mode = 'full'
################ change sourcewate too


if '60' in exp_result_folder or '73' in exp_result_folder:
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




rows = total_df.loc[(total_df['ruined_flag']==False) & (total_df['model_mode']==model_mode) & (total_df['source_wait']!=270)].sort_values(['partitioning'], kind='mergesort')

### drop abormality 
# mask = rows[['sum_computations_tail_mean', 'estimated_computation_latency']].apply(lambda x: abs(x[0] - x[1]) < 20, axis=1)
# rows = rows[mask]
# mask = rows[['sum_communications_tail_mean', 'estimated_communication_latency']].apply(lambda x: abs(x[0] - x[1]) < 20, axis=1)
# rows = rows[mask]



experiment_e2e_latency = np.array(rows[['e2e_tail_mean']].to_numpy()[:,0])

estimated_e2e_latency = np.array(rows[['estimated_e2e_latency']].to_numpy()[:,0])

print("NNNNNNNNNNNNNN", len(experiment_e2e_latency))

r=0.75
fig, ax = plt.subplots(figsize=(r*4.7, r*4.1))  

plt.scatter(experiment_e2e_latency, estimated_e2e_latency, color='crimson')
# plt.plot(np.arange(0, 150, 0.01), np.arange(0, 150,0.01), color='black')
# plt.plot(np.arange(0, 200, 0.01), np.arange(0, 200,0.01), color='black')
# plt.plot(np.arange(350, 670, 0.01), np.arange(350, 670,0.01), color='black')
plt.plot(np.arange(1800, 2400, 0.01), np.arange(1800, 2400,0.01), color='black')

beta, intercept =polyfit(experiment_e2e_latency, estimated_e2e_latency, degree=1)['polynomial']
r_squared = polyfit(experiment_e2e_latency, estimated_e2e_latency, degree=1)['determination']

print(f'beta: {beta}, intercept: {intercept}, r_squared: {r_squared}')
ax.text(0.97, 0.03, r'$\beta = %.2f, R^2=%.2f$' % (beta, r_squared), transform=ax.transAxes, ha='right', va='bottom')


# #resnet20
# plt.xticks([0, 50, 100, 150])  # Set the ticks of the x axis
# plt.yticks([0, 50, 100, 150])  # Set the ticks of the y axis

# #resnet110
# plt.xticks([0, 50, 100, 150, 200])  # Set the ticks of the x axis
# plt.yticks([0, 50, 100, 150, 200])  # Set the ticks of the y axis


# #efficientnetb0
# plt.xticks([350, 450, 550, 650])  # Set the ticks of the x axis
# plt.yticks([350, 450, 550, 650])  # Set the ticks of the y axis


#vggish
plt.xticks([1800, 2000, 2200, 2400])  # Set the ticks of the x axis
plt.yticks([1800, 2000, 2200, 2400])  # Set the ticks of the y axis

plt.title(f"{model_name}, {dataset}", fontsize=12)
# plt.legend()
plt.xlabel('Experiment')
plt.ylabel('Estimated')
# plt.grid()
plt.tight_layout()
plt.show()