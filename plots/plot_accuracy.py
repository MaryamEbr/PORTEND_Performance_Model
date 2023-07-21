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
exp_result_folder = "exp_folder/Experiment74"
################ change sourcewate too

model_mode = 'full'
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


# read the experiment files from csv file or original txt files
if read_again_flag == True:
    total_df = read_exp_results(exp_result_folder) 
else:
    total_df = pd.read_pickle(f'{exp_result_folder}/total_df.pkl')



rows = total_df.loc[ (total_df['ruined_flag']==False) & (total_df['model_mode']==model_mode)& (total_df['source_wait']!=3007)]





accuracy_list = rows[['accuracy']].to_numpy()[:,0]
estimate_accuracy_list = rows[['estimated_accuracy']].to_numpy()[:,0]

# Store the lists in a file
import pickle
with open('resnet110.pickle', 'wb') as f:
    pickle.dump((accuracy_list, estimate_accuracy_list), f)

r=0.75
fig, ax = plt.subplots(figsize=(r*4.7, r*4.1))  

sc = plt.scatter(accuracy_list, estimate_accuracy_list, color='dimgray')
plt.plot(np.arange(0.6, 1, 0.01), np.arange(0.6, 1, 0.01), color='black')

beta, intercept =polyfit(accuracy_list, estimate_accuracy_list, degree=1)['polynomial']
r_squared = polyfit(accuracy_list, estimate_accuracy_list, degree=1)['determination']

print(f'beta: {beta}, r_squared: {r_squared}')
ax.text(0.97, 0.03, r'$\beta = %.2f, R^2=%.2f$' % (beta, r_squared), transform=ax.transAxes, ha='right', va='bottom')

plt.xticks([0.6, 0.7, 0.8, 0.9, 1])  # Set the ticks of the x axis
plt.yticks([0.6, 0.7, 0.8, 0.9, 1])  # Set the ticks of the y axis

plt.title(f"{model_name}, {dataset}", fontsize=12)
# plt.legend()
plt.xlabel('Experiment')
plt.ylabel('Estimated')
# plt.grid()
plt.tight_layout()
plt.show()