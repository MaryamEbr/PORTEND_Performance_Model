#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 12


performance_model_folder = 'performance_model_results_top1/'
experiment_folder = 'exp_folder/'
destination_tier = 0

resnet20_accuracy_constraint = 0.88
resnet110_accuracy_constraint = 0.88
efficientnetb0_accuracy_constraint = 0.72
vggish_accuracy_constraint = 0.71


resnet20_exp_df = pd.read_pickle(f'{experiment_folder}Experiment71/total_df.pkl')
resnet110_exp_df = pd.read_pickle(f'{experiment_folder}Experiment72/total_df.pkl')
effb0_exp_df = pd.read_pickle(f'{experiment_folder}Experiment73/total_df.pkl')
vggish_exp_df = pd.read_pickle(f'{experiment_folder}Experiment74/total_df.pkl')


# resnet20
#  [list([[1, 9], [10, 10]]) list([0, 1]) 0.7
our_latency_exp_resnet20 = resnet20_exp_df.loc[(resnet20_exp_df['placement'].astype(str) == '[0, 1]') & (resnet20_exp_df['partitioning'].astype(str) == '[[1, 9], [10, 10]]') & (resnet20_exp_df['threshold'] == 0.7) & (resnet20_exp_df['destination_tier']==destination_tier) & (resnet20_exp_df['source_wait']!=270) & (resnet20_exp_df['model_mode']=='full')]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
# resnet110
#  [list([[1, 25], [26, 40]]) list([0, 1]) 0.1
our_latency_exp_resnet110 = resnet110_exp_df.loc[(resnet110_exp_df['placement'].astype(str) == '[0, 1]') & (resnet110_exp_df['partitioning'].astype(str) == '[[1, 25], [26, 40]]') & (resnet110_exp_df['threshold'] == 0.1) & (resnet110_exp_df['destination_tier']==destination_tier) & (resnet110_exp_df['source_wait']!=270) & (resnet110_exp_df['model_mode']=='full') ]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
# effb0
#  [list([[1, 4], [5, 8]]) list([0, 1]) 0.3 
our_latency_exp_effnetb0 = effb0_exp_df.loc[(effb0_exp_df['placement'].astype(str) == '[0, 1]') & (effb0_exp_df['partitioning'].astype(str) == '[[1, 4], [5, 8]]') & (effb0_exp_df['threshold'] == 0.3) & (effb0_exp_df['destination_tier']==destination_tier) & (effb0_exp_df['source_wait']!=707) & (effb0_exp_df['model_mode']=='full') ]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
# # vggish
# #  [list([[1, 4]]) list([2]) 0.0 1880.1220841633308]
# our_latency_exp_vggish = vggish_exp_df.loc[(vggish_exp_df['placement'].astype(str) == '[2]') & (vggish_exp_df['partitioning'].astype(str) == '[[1, 4]]') & (vggish_exp_df['threshold'] == 0) & (vggish_exp_df['destination_tier']==destination_tier) & (vggish_exp_df['source_wait']!=3007) & (vggish_exp_df['model_mode']=='full')]\
#                             .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])



all_comm = np.sum(our_latency_exp_effnetb0['all_communications_lists'].to_numpy()[0], 1)
all_comp = np.sum(our_latency_exp_effnetb0['all_computations_lists'].to_numpy()[0], 1)
all_e2e = all_comm + all_comp





# create a sorted list of the numbers
all_e2e_sorted = np.sort(all_e2e)

# calculate the cumulative probabilities
cum_probs = np.arange(len(all_e2e_sorted)) / float(len(all_e2e_sorted))



r=1.3
fig, ax = plt.subplots(figsize=(r*2.5, r*2))   


# plot the CDF
plt.plot(all_e2e_sorted, cum_probs, color='darkblue', linewidth=2)

# add labels and title
plt.xlabel('End to End Latency (ms)')
plt.ylabel('CDF')
# plt.title('ResNet20, CIFAR10', fontsize=12)
# plt.xticks([50, 100, 150])
# plt.xlim(20, 170)
# show the plot
plt.tight_layout()
plt.show()
