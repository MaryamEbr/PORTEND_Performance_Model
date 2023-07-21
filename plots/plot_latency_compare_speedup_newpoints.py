#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 12

# Add the numbers on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points", ha='center', va='bottom')



performance_model_folder = 'performance_model_results_top1/'
experiment_folder = 'exp_folder/'
destination_tier = 0

resnet20_accuracy_constraint = 0.88
resnet110_accuracy_constraint = 0.88
efficientnetb0_accuracy_constraint = 0.72
vggish_accuracy_constraint = 0.71




# #### loading the dataframes for performance models and experiments
# resnet20_perf_df = pd.read_pickle(f'{performance_model_folder}/total_df_resnet20.pkl')
# resnet110_perf_df = pd.read_pickle(f'{performance_model_folder}/total_df_resnet110.pkl')
# effb0_perf_df = pd.read_pickle(f'{performance_model_folder}/total_df_effb0.pkl')
# vggish_perf_df = pd.read_pickle(f'{performance_model_folder}/total_df_vggish.pkl')

resnet20_exp_df = pd.read_pickle(f'{experiment_folder}Experiment71/total_df.pkl')
resnet110_exp_df = pd.read_pickle(f'{experiment_folder}Experiment72/total_df.pkl')
effb0_exp_df = pd.read_pickle(f'{experiment_folder}Experiment73/total_df.pkl')
vggish_exp_df = pd.read_pickle(f'{experiment_folder}Experiment74/total_df.pkl')



# # best edge only in each model
# edge_latency_perf_resnet20 = resnet20_perf_df.loc[(resnet20_perf_df['placement'].astype(str) == '[0]') & (resnet20_perf_df['partitioning'].astype(str) == '[[1, 10]]') & (resnet20_perf_df['estimated_accuracy'] >= resnet20_accuracy_constraint) & (resnet20_perf_df['destination_tier']==destination_tier)]\
#                             .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
# edge_latency_perf_resnet110 = resnet110_perf_df.loc[(resnet110_perf_df['placement'].astype(str) == '[0]') & (resnet110_perf_df['partitioning'].astype(str) == '[[1, 55]]') & (resnet110_perf_df['estimated_accuracy'] >= resnet110_accuracy_constraint) & (resnet110_perf_df['destination_tier']==destination_tier)]\
#                             .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
# edge_latency_perf_effnetb0 = effb0_perf_df.loc[(effb0_perf_df['placement'].astype(str) == '[0]') & (effb0_perf_df['partitioning'].astype(str) == '[[1, 8]]') & (effb0_perf_df['estimated_accuracy'] >= efficientnetb0_accuracy_constraint)& (effb0_perf_df['destination_tier']==destination_tier)]\
#                             .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
# edge_latency_perf_vggish = vggish_perf_df.loc[(vggish_perf_df['placement'].astype(str) == '[0]') & (vggish_perf_df['partitioning'].astype(str) == '[[1, 4]]') & (vggish_perf_df['estimated_accuracy'] >= vggish_accuracy_constraint) & (vggish_perf_df['destination_tier']==destination_tier)]\
#                              .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])


# # best cloud only in each model
# cloud_latency_perf_resnet20 = resnet20_perf_df.loc[(resnet20_perf_df['placement'].astype(str) == '[2]') & (resnet20_perf_df['partitioning'].astype(str) == '[[1, 10]]') & (resnet20_perf_df['estimated_accuracy'] >= resnet20_accuracy_constraint) & (resnet20_perf_df['destination_tier']==destination_tier)]\
#                             .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
# cloud_latency_perf_resnet110 = resnet110_perf_df.loc[(resnet110_perf_df['placement'].astype(str) == '[2]') & (resnet110_perf_df['partitioning'].astype(str) == '[[1, 55]]') & (resnet110_perf_df['estimated_accuracy'] >= resnet110_accuracy_constraint) & (resnet110_perf_df['destination_tier']==destination_tier)]\
#                             .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
# cloud_latency_perf_effnetb0 = effb0_perf_df.loc[(effb0_perf_df['placement'].astype(str) == '[2]') & (effb0_perf_df['partitioning'].astype(str) == '[[1, 8]]') & (effb0_perf_df['estimated_accuracy'] >= efficientnetb0_accuracy_constraint)& (effb0_perf_df['destination_tier']==destination_tier)]\
#                             .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
# cloud_latency_perf_vggish = vggish_perf_df.loc[(vggish_perf_df['placement'].astype(str) == '[2]') & (vggish_perf_df['partitioning'].astype(str) == '[[1, 4]]') & (vggish_perf_df['estimated_accuracy'] >= vggish_accuracy_constraint) & (vggish_perf_df['destination_tier']==destination_tier)]\
#                             .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])

# # best ns in each model
# ns_latency_perf_resnet20 = resnet20_perf_df.loc[ (resnet20_perf_df['partial_flag'] == False) & (resnet20_perf_df['threshold'] == 0) & (resnet20_perf_df['placement'].astype(str)!='[0, 1]') & (resnet20_perf_df['placement'].astype(str)!='[1]') & (resnet20_perf_df['estimated_accuracy'] >= resnet20_accuracy_constraint)]\
#                         .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
                        
# ns_latency_perf_resnet110 = resnet110_perf_df.loc[ (resnet110_perf_df['partial_flag'] == False) & (resnet110_perf_df['threshold'] == 0) & (resnet110_perf_df['placement'].astype(str)!='[0, 1]') & (resnet110_perf_df['placement'].astype(str)!='[1]') & (resnet110_perf_df['estimated_accuracy'] >= resnet110_accuracy_constraint)]\
#                         .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
                        
# ns_latency_perf_effnetb0 = effb0_perf_df.loc[(effb0_perf_df['partial_flag'] == False) & (effb0_perf_df['threshold'] == 0) & (effb0_perf_df['placement'].astype(str)!='[0, 1]') & (effb0_perf_df['placement'].astype(str)!='[1]') & (effb0_perf_df['estimated_accuracy'] >= efficientnetb0_accuracy_constraint)]\
#                         .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
                        
# ns_latency_perf_vggish = vggish_perf_df.loc[(vggish_perf_df['partial_flag'] == False) & (vggish_perf_df['threshold'] == 0) & (vggish_perf_df['placement'].astype(str)!='[0, 1]') & (vggish_perf_df['placement'].astype(str)!='[1]') & (vggish_perf_df['estimated_accuracy'] >= vggish_accuracy_constraint)]\
#                         .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])

# # best our method in each model
# our_latency_perf_resnet20 = resnet20_perf_df.loc[(resnet20_perf_df['estimated_accuracy'] >= resnet20_accuracy_constraint)]\
#                         .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
# our_latency_perf_resnet110 = resnet110_perf_df.loc[(resnet110_perf_df['estimated_accuracy'] >= resnet110_accuracy_constraint)]\
#                         .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
# our_latency_perf_effnetb0 = effb0_perf_df.loc[(effb0_perf_df['estimated_accuracy'] >= efficientnetb0_accuracy_constraint)]\
#                         .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
# our_latency_perf_vggish = vggish_perf_df.loc[(vggish_perf_df['estimated_accuracy'] >= vggish_accuracy_constraint)]\
#                         .sort_values(by=['estimated_e2e_latency', 'estimated_accuracy'], ascending=[True, False])
                



# # Define the data for the plot
# edge_latency_perf_resnet20_0 = edge_latency_perf_resnet20['estimated_e2e_latency'].values[0] if len(edge_latency_perf_resnet20.values) > 0 else 0
# edge_latency_perf_resnet110_0 = edge_latency_perf_resnet110['estimated_e2e_latency'].values[0] if len(edge_latency_perf_resnet110.values) > 0 else 0
# edge_latency_perf_effnetb0_0 = edge_latency_perf_effnetb0['estimated_e2e_latency'].values[0] if len(edge_latency_perf_effnetb0.values) > 0 else 0
# edge_latency_perf_vggish_0 = edge_latency_perf_vggish['estimated_e2e_latency'].values[0] if len(edge_latency_perf_vggish.values) > 0 else 0

# cloud_latency_perf_resnet20_0 = cloud_latency_perf_resnet20['estimated_e2e_latency'].values[0] if len(cloud_latency_perf_resnet20.values) > 0 else 0
# cloud_latency_perf_resnet110_0 = cloud_latency_perf_resnet110['estimated_e2e_latency'].values[0] if len(cloud_latency_perf_resnet110.values) > 0 else 0
# cloud_latency_perf_effnetb0_0 = cloud_latency_perf_effnetb0['estimated_e2e_latency'].values[0] if len(cloud_latency_perf_effnetb0.values) > 0 else 0
# cloud_latency_perf_vggish_0 = cloud_latency_perf_vggish['estimated_e2e_latency'].values[0] if len(cloud_latency_perf_vggish.values) > 0 else 0

# ns_latency_perf_resnet20_0 = ns_latency_perf_resnet20['estimated_e2e_latency'].values[0] if len(ns_latency_perf_resnet20.values) > 0 else 0
# ns_latency_perf_resnet110_0 = ns_latency_perf_resnet110['estimated_e2e_latency'].values[0] if len(ns_latency_perf_resnet110.values) > 0 else 0
# ns_latency_perf_effnetb0_0 = ns_latency_perf_effnetb0['estimated_e2e_latency'].values[0] if len(ns_latency_perf_effnetb0.values) > 0 else 0
# ns_latency_perf_vggish_0 = ns_latency_perf_vggish['estimated_e2e_latency'].values[0] if len(ns_latency_perf_vggish.values) > 0 else 0

# our_latency_perf_resnet20_0 = our_latency_perf_resnet20['estimated_e2e_latency'].values[0] if len(our_latency_perf_resnet20.values) > 0 else 0
# our_latency_perf_resnet110_0 = our_latency_perf_resnet110['estimated_e2e_latency'].values[0] if len(our_latency_perf_resnet110.values) > 0 else 0
# our_latency_perf_effnetb0_0 = our_latency_perf_effnetb0['estimated_e2e_latency'].values[0] if len(our_latency_perf_effnetb0.values) > 0 else 0
# our_latency_perf_vggish_0 = our_latency_perf_vggish['estimated_e2e_latency'].values[0] if len(our_latency_perf_vggish.values) > 0 else 0



# bar1 = [cloud_latency_perf_resnet20_0/cloud_latency_perf_resnet20_0, cloud_latency_perf_resnet110_0/cloud_latency_perf_resnet110_0, cloud_latency_perf_effnetb0_0/cloud_latency_perf_effnetb0_0, cloud_latency_perf_vggish_0/cloud_latency_perf_vggish_0]
# bar2 = [cloud_latency_perf_resnet20_0/edge_latency_perf_resnet20_0, cloud_latency_perf_resnet110_0/edge_latency_perf_resnet110_0, cloud_latency_perf_effnetb0_0/edge_latency_perf_effnetb0_0, cloud_latency_perf_vggish_0/edge_latency_perf_vggish_0]
# bar3 = [cloud_latency_perf_resnet20_0/ns_latency_perf_resnet20_0, cloud_latency_perf_resnet110_0/ns_latency_perf_resnet110_0, cloud_latency_perf_effnetb0_0/ns_latency_perf_effnetb0_0, cloud_latency_perf_vggish_0/ns_latency_perf_vggish_0]
# bar4 = [cloud_latency_perf_resnet20_0/our_latency_perf_resnet20_0, cloud_latency_perf_resnet110_0/our_latency_perf_resnet110_0, cloud_latency_perf_effnetb0_0/our_latency_perf_effnetb0_0, cloud_latency_perf_vggish_0/our_latency_perf_vggish_0]

# labels = ['ResNet20+CIFAR10', 'ResNet110+CIFAR10', 'EfficientNetB0+ImageNet', 'VGGish+AudioSet']

# # Set up the plot
# fig, ax = plt.subplots(figsize=(10, 6))

# # Set the bar width and positions
# width = 0.2
# x_pos1 = np.arange(len(bar1))
# x_pos2 = [x + width for x in x_pos1]
# x_pos3 = [x + width for x in x_pos2]
# x_pos4 = [x + width for x in x_pos3]

# # Plot the bars
# rects1 = ax.bar(x_pos1, bar1, width, label='cloud only', color='darkblue')
# rects2 = ax.bar(x_pos2, bar2, width, label='edge only', color='darkgreen')
# rects3 = ax.bar(x_pos3, bar3, width, label='neurosurgeon', color='darkred')
# rects4 = ax.bar(x_pos4, bar4, width, label='our method', color='darkorange')


# # a line showing 1 speedup
# plt.axhline(y=1, color='black', linestyle='-')

# # # add numbers at top of bars
# # autolabel(rects1)
# # autolabel(rects2)
# # autolabel(rects3)
# # autolabel(rects4)


# # Add labels, title, and legend
# ax.set_ylabel('latency speedup')
# ax.set_title('latency comparison (destination tier0)/performance model results')
# ax.set_xticks([x + 1.5*width for x in range(len(labels))])
# ax.set_xticklabels(labels)
# ax.set_yticks(np.arange(0, 13, 1))
# ax.legend()

# # Show the plot
# plt.show()



# now that we did the performance model plot, lets do the experiment plot
# first we need to get the best config from perfomance model
# but edge only and cloud only approach don't need that
# best edge only in each model
edge_latency_exp_resnet20 = resnet20_exp_df.loc[(resnet20_exp_df['source_wait']==270) & (resnet20_exp_df['model_mode']=='full') & (resnet20_exp_df['placement'].astype(str) == '[0]') & (resnet20_exp_df['partitioning'].astype(str) == '[[1, 10]]') & (resnet20_exp_df['accuracy'] >= resnet20_accuracy_constraint) & (resnet20_exp_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
edge_latency_exp_resnet110 = resnet110_exp_df.loc[(resnet110_exp_df['source_wait']==270) & (resnet110_exp_df['model_mode']=='full') & (resnet110_exp_df['placement'].astype(str) == '[0]') & (resnet110_exp_df['partitioning'].astype(str) == '[[1, 55]]') & (resnet110_exp_df['accuracy'] >= resnet110_accuracy_constraint) & (resnet110_exp_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
edge_latency_exp_effnetb0 = effb0_exp_df.loc[(effb0_exp_df['source_wait']==707) & (effb0_exp_df['model_mode']=='full') & (effb0_exp_df['placement'].astype(str) == '[0]') & (effb0_exp_df['partitioning'].astype(str) == '[[1, 8]]') & (effb0_exp_df['accuracy'] >= efficientnetb0_accuracy_constraint)& (effb0_exp_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
edge_latency_exp_vggish = vggish_exp_df.loc[(vggish_exp_df['model_mode']=='full')& (vggish_exp_df['placement'].astype(str) == '[0]') & (vggish_exp_df['partitioning'].astype(str) == '[[1, 4]]') & (vggish_exp_df['accuracy'] >= vggish_accuracy_constraint) & (vggish_exp_df['destination_tier']==destination_tier)]\
                             .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])


# best cloud only in each model
cloud_latency_exp_resnet20 = resnet20_exp_df.loc[(resnet20_exp_df['source_wait']==270) & (resnet20_exp_df['model_mode']=='full') & (resnet20_exp_df['placement'].astype(str) == '[2]') & (resnet20_exp_df['partitioning'].astype(str) == '[[1, 10]]') & (resnet20_exp_df['accuracy'] >= resnet20_accuracy_constraint) & (resnet20_exp_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
cloud_latency_exp_resnet110 = resnet110_exp_df.loc[(resnet110_exp_df['source_wait']==270) & (resnet110_exp_df['model_mode']=='full') & (resnet110_exp_df['placement'].astype(str) == '[2]') & (resnet110_exp_df['partitioning'].astype(str) == '[[1, 55]]') & (resnet110_exp_df['accuracy'] >= resnet110_accuracy_constraint) & (resnet110_exp_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
cloud_latency_exp_effnetb0 = effb0_exp_df.loc[(effb0_exp_df['source_wait']==707) & (effb0_exp_df['model_mode']=='full') & (effb0_exp_df['placement'].astype(str) == '[2]') & (effb0_exp_df['partitioning'].astype(str) == '[[1, 8]]') & (effb0_exp_df['accuracy'] >= efficientnetb0_accuracy_constraint)& (effb0_exp_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
cloud_latency_exp_vggish = vggish_exp_df.loc[(vggish_exp_df['model_mode']=='full')& (vggish_exp_df['placement'].astype(str) == '[2]') & (vggish_exp_df['partitioning'].astype(str) == '[[1, 4]]') & (vggish_exp_df['accuracy'] >= vggish_accuracy_constraint) & (vggish_exp_df['destination_tier']==destination_tier)]\
                             .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])

# print("************************************************************************")
# print(edge_latency_exp_effnetb0[['sum_communications_tail_mean', 'sum_computations_tail_mean', 'e2e_tail_mean', 'accuracy']])
# print("************************************************************************")
# print(cloud_latency_exp_effnetb0[['sum_communications_tail_mean', 'sum_computations_tail_mean', 'e2e_tail_mean', 'accuracy']])
# print("************************************************************************")


# # for ns and our method, we need to get the best config from performance model
# print("ns best config")
# print("resnet20\n", ns_latency_perf_resnet20.head(10)[['partitioning', 'placement', 'threshold', 'estimated_e2e_latency']].values)
# print("resnet110\n", ns_latency_perf_resnet110.head(10)[['partitioning', 'placement', 'threshold', 'estimated_e2e_latency']].values)
# print("effb0\n", ns_latency_perf_effnetb0.head(10)[['partitioning', 'placement', 'threshold', 'estimated_e2e_latency']].values)
# print("vggish\n", ns_latency_perf_vggish.head(10)[['partitioning', 'placement', 'threshold', 'estimated_e2e_latency']].values)

# print("our best config")
# print("resnet20\n", our_latency_perf_resnet20.head(20)[['partitioning', 'placement', 'threshold', 'estimated_e2e_latency']].values)
# print("resnet110\n", our_latency_perf_resnet110.head(100)[['partitioning', 'placement', 'threshold', 'estimated_e2e_latency']].values)
# print("effb0\n", our_latency_perf_effnetb0.head(20)[['partitioning', 'placement', 'threshold', 'estimated_e2e_latency']].values)
# print("vggish\n", our_latency_perf_vggish.head(20)[['partitioning', 'placement', 'threshold', 'estimated_e2e_latency']].values)

# chosen ones:
# NS
# resnet20
#  [[list([[1, 10]]) list([0]) 0.0 15.768533840007562]
ns_latency_exp_resnet20 = resnet20_exp_df.loc[(resnet20_exp_df['placement'].astype(str) == '[0]') & (resnet20_exp_df['partitioning'].astype(str) == '[[1, 10]]') & (resnet20_exp_df['threshold'] == 0) & (resnet20_exp_df['destination_tier']==destination_tier) & (resnet20_exp_df['source_wait']==270) & (resnet20_exp_df['model_mode']=='full')]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
# resnet110
#  [[list([[1, 55]]) list([0]) 0.0 83.00719599343134]
ns_latency_exp_resnet110 = resnet110_exp_df.loc[(resnet110_exp_df['placement'].astype(str) == '[0]') & (resnet110_exp_df['partitioning'].astype(str) == '[[1, 55]]') & (resnet110_exp_df['threshold'] == 0) & (resnet110_exp_df['destination_tier']==destination_tier) & (resnet110_exp_df['source_wait']==270) & (resnet110_exp_df['model_mode']=='full') ]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
# effb0
#  [[list([[1, 4], [5, 8]]) list([0, 2]) 0.0 498.20140792656855]
ns_latency_exp_effnetb0 = effb0_exp_df.loc[(effb0_exp_df['placement'].astype(str) == '[0, 2]') & (effb0_exp_df['partitioning'].astype(str) == '[[1, 4], [5, 8]]') & (effb0_exp_df['threshold'] == 0) & (effb0_exp_df['destination_tier']==destination_tier) & (effb0_exp_df['source_wait']==707) & (effb0_exp_df['model_mode']=='full') ]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
# vggish, 
#  [[list([[1, 4]]) list([2]) 0.0 1880.1220841633308]
ns_latency_exp_vggish = vggish_exp_df.loc[(vggish_exp_df['placement'].astype(str) == '[2]') & (vggish_exp_df['partitioning'].astype(str) == '[[1, 4]]') & (vggish_exp_df['threshold'] == 0) & (vggish_exp_df['destination_tier']==destination_tier) & (vggish_exp_df['model_mode']=='full')]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])

# OUR
# resnet20
#  [list([[1, 9], [10, 10]]) list([0, 1]) 0.7
our_latency_exp_resnet20 = resnet20_exp_df.loc[(resnet20_exp_df['placement'].astype(str) == '[0, 1]') & (resnet20_exp_df['partitioning'].astype(str) == '[[1, 9], [10, 10]]') & (resnet20_exp_df['threshold'] == 0.7) & (resnet20_exp_df['destination_tier']==destination_tier) & (resnet20_exp_df['source_wait']!=270) & (resnet20_exp_df['model_mode']=='full')]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
# resnet110
#  [list([[1, 25], [26, 40]]) list([0, 1]) 0.1
our_latency_exp_resnet110 = resnet110_exp_df.loc[(resnet110_exp_df['placement'].astype(str) == '[0, 1]') & (resnet110_exp_df['partitioning'].astype(str) == '[[1, 25], [26, 40]]') & (resnet110_exp_df['threshold'] == 0.1) & (resnet110_exp_df['destination_tier']==destination_tier) & (resnet110_exp_df['source_wait']!=270) & (resnet110_exp_df['model_mode']=='full') ]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
# effb0
#  [[list([[1, 4], [5, 8]]) list([0, 1]) 0.3 
our_latency_exp_effnetb0 = effb0_exp_df.loc[(effb0_exp_df['placement'].astype(str) == '[0, 1]') & (effb0_exp_df['partitioning'].astype(str) == '[[1, 4], [5, 8]]') & (effb0_exp_df['threshold'] == 0.3) & (effb0_exp_df['destination_tier']==destination_tier) & (effb0_exp_df['source_wait']!=707) & (effb0_exp_df['model_mode']=='full') ]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
# vggish
#  [list([[1, 4]]) list([2]) 0.0 1880.1220841633308]
our_latency_exp_vggish = vggish_exp_df.loc[(vggish_exp_df['placement'].astype(str) == '[2]') & (vggish_exp_df['partitioning'].astype(str) == '[[1, 4]]') & (vggish_exp_df['threshold'] == 0) & (vggish_exp_df['destination_tier']==destination_tier) & (vggish_exp_df['source_wait']!=3007) & (vggish_exp_df['model_mode']=='full')]\
                            .sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])






print("\n\n\n\nNS: do they exist in exp?")
print(ns_latency_exp_resnet20[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'source_wait', 'model_mode', 'destination_tier']].values)
print(ns_latency_exp_resnet110[['partitioning', 'placement', 'threshold', 'e2e_tail_mean','source_wait', 'model_mode', 'destination_tier']].values)
print(ns_latency_exp_effnetb0[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'source_wait', 'model_mode', 'destination_tier']].values)
print(ns_latency_exp_vggish[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'source_wait', 'model_mode', 'destination_tier']].values)


print("OUR: do they exist in exp?")
print(our_latency_exp_resnet20[['partitioning', 'placement', 'threshold', 'e2e_tail_mean']].values)
print(our_latency_exp_resnet110[['partitioning', 'placement', 'threshold', 'e2e_tail_mean']].values)
print(our_latency_exp_effnetb0[['partitioning', 'placement', 'threshold', 'e2e_tail_mean']].values)
print(our_latency_exp_vggish[['partitioning', 'placement', 'threshold', 'e2e_tail_mean']].values)






edge_latency_exp_resnet20_0 = edge_latency_exp_resnet20['e2e_tail_mean'].values[0] if len(edge_latency_exp_resnet20.values) > 0 else 0
edge_latency_exp_resnet110_0 = edge_latency_exp_resnet110['e2e_tail_mean'].values[0] if len(edge_latency_exp_resnet110.values) > 0 else 0
edge_latency_exp_effnetb0_0 = edge_latency_exp_effnetb0['e2e_tail_mean'].values[0] if len(edge_latency_exp_effnetb0.values) > 0 else 0
edge_latency_exp_vggish_0 = edge_latency_exp_vggish['e2e_tail_mean'].values[0] if len(edge_latency_exp_vggish.values) > 0 else 0

cloud_latency_exp_resnet20_0 = cloud_latency_exp_resnet20['e2e_tail_mean'].values[0] if len(cloud_latency_exp_resnet20.values) > 0 else 0
cloud_latency_exp_resnet110_0 = cloud_latency_exp_resnet110['e2e_tail_mean'].values[0] if len(cloud_latency_exp_resnet110.values) > 0 else 0
cloud_latency_exp_effnetb0_0 = cloud_latency_exp_effnetb0['e2e_tail_mean'].values[0] if len(cloud_latency_exp_effnetb0.values) > 0 else 0
cloud_latency_exp_vggish_0 = cloud_latency_exp_vggish['e2e_tail_mean'].values[0] if len(cloud_latency_exp_vggish.values) > 0 else 0

ns_latency_exp_resnet20_0 = ns_latency_exp_resnet20['e2e_tail_mean'].values[0] if len(ns_latency_exp_resnet20.values) > 0 else 0
ns_latency_exp_resnet110_0 = ns_latency_exp_resnet110['e2e_tail_mean'].values[0] if len(ns_latency_exp_resnet110.values) > 0 else 0
ns_latency_exp_effnetb0_0 = ns_latency_exp_effnetb0['e2e_tail_mean'].values[0] if len(ns_latency_exp_effnetb0.values) > 0 else 0
ns_latency_exp_vggish_0 = ns_latency_exp_vggish['e2e_tail_mean'].values[0] if len(ns_latency_exp_vggish.values) > 0 else 0

our_latency_exp_resnet20_0 = our_latency_exp_resnet20['e2e_tail_mean'].values[0] if len(our_latency_exp_resnet20.values) > 0 else 0
our_latency_exp_resnet110_0 = our_latency_exp_resnet110['e2e_tail_mean'].values[0] if len(our_latency_exp_resnet110.values) > 0 else 0
our_latency_exp_effnetb0_0 = our_latency_exp_effnetb0['e2e_tail_mean'].values[0] if len(our_latency_exp_effnetb0.values) > 0 else 0
our_latency_exp_vggish_0 = our_latency_exp_vggish['e2e_tail_mean'].values[0] if len(our_latency_exp_vggish.values) > 0 else 0


bar1 = [edge_latency_exp_resnet20_0/edge_latency_exp_resnet20_0, edge_latency_exp_resnet110_0/edge_latency_exp_resnet110_0, edge_latency_exp_effnetb0_0/edge_latency_exp_effnetb0_0, edge_latency_exp_vggish_0/edge_latency_exp_vggish_0]
bar2 = [edge_latency_exp_resnet20_0/cloud_latency_exp_resnet20_0, edge_latency_exp_resnet110_0/cloud_latency_exp_resnet110_0, edge_latency_exp_effnetb0_0/cloud_latency_exp_effnetb0_0, edge_latency_exp_vggish_0/cloud_latency_exp_vggish_0]
bar3 = [edge_latency_exp_resnet20_0/ns_latency_exp_resnet20_0, edge_latency_exp_resnet110_0/ns_latency_exp_resnet110_0, edge_latency_exp_effnetb0_0/ns_latency_exp_effnetb0_0, edge_latency_exp_vggish_0/ns_latency_exp_vggish_0]
bar4 = [edge_latency_exp_resnet20_0/our_latency_exp_resnet20_0, edge_latency_exp_resnet110_0/our_latency_exp_resnet110_0, edge_latency_exp_effnetb0_0/our_latency_exp_effnetb0_0, edge_latency_exp_vggish_0/our_latency_exp_vggish_0]

labels = ['ResNet20', 'ResNet110', 'EfficientNetB0', 'VGGish']

# Set up the plot
r=1.5
fig, ax = plt.subplots(figsize=(r*5, r*1.5))  

# Set the bar width and positions
width = 0.2
x_pos1 = np.arange(len(bar1))
x_pos2 = [x + width for x in x_pos1]
x_pos3 = [x + width for x in x_pos2]
x_pos4 = [x + width for x in x_pos3]

# a line showing 1 speedup
plt.axhline(y=1, color='black', linestyle='--', zorder=-1)


# Plot the bars
rects1 = ax.bar(x_pos1, bar1, width, label='edge-only', color='green')
rects2 = ax.bar(x_pos2, bar2, width, label='cloud-only', color='blue')
rects3 = ax.bar(x_pos3, bar3, width, label='Neurosurgeon', color='firebrick')
rects4 = ax.bar(x_pos4, bar4, width, label='PORTEND', color='black')


print(bar1)
print(bar2)
print(bar3)
print(bar4)


# Add labels, title, and legend
ax.set_ylabel('Latency Speedup')
# ax.set_title('Problem1: Minimizing Latency')
ax.set_yticks(np.arange(0, 3, 1))
ax.set_xticks([x + 1.5*width for x in range(len(labels))])
ax.set_xticklabels(labels)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# Show the plot
plt.tight_layout()
plt.show()