#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




performance_model_folder = 'performance_model_results_top1/'
experiment_folder = 'exp_folder/'
destination_tier = 0

resnet20_latency_constraint = 20
resnet110_latency_constraint = 50
efficientnetb0_latency_constraint = 500
vggish_latency_constraint = 2000


#### loading the dataframes for performance models and experiments
resnet20_perf_df = pd.read_pickle(f'{performance_model_folder}/total_df_resnet20.pkl')
resnet110_perf_df = pd.read_pickle(f'{performance_model_folder}/total_df_resnet110.pkl')
effb0_perf_df = pd.read_pickle(f'{performance_model_folder}/total_df_effb0.pkl')
vggish_perf_df = pd.read_pickle(f'{performance_model_folder}/total_df_vggish.pkl')

resnet20_exp_df = pd.read_pickle(f'{experiment_folder}Experiment71/total_df.pkl')
resnet110_exp_df = pd.read_pickle(f'{experiment_folder}Experiment72/total_df.pkl')
effb0_exp_df = pd.read_pickle(f'{experiment_folder}Experiment73/total_df.pkl')
vggish_exp_df = pd.read_pickle(f'{experiment_folder}Experiment74/total_df.pkl')


# best edge only in each model
edge_accuracy_perf_resnet20 = resnet20_perf_df.loc[(resnet20_perf_df['placement'].astype(str) == '[0]') & (resnet20_perf_df['partitioning'].astype(str) == '[[1, 10]]') & (resnet20_perf_df['estimated_e2e_latency'] <= resnet20_latency_constraint) & (resnet20_perf_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])
edge_accuracy_perf_resnet110 = resnet110_perf_df.loc[(resnet110_perf_df['placement'].astype(str) == '[0]') & (resnet110_perf_df['partitioning'].astype(str) == '[[1, 55]]') & (resnet110_perf_df['estimated_e2e_latency'] <= resnet110_latency_constraint) & (resnet110_perf_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])
edge_accuracy_perf_effnetb0 = effb0_perf_df.loc[(effb0_perf_df['placement'].astype(str) == '[0]') & (effb0_perf_df['partitioning'].astype(str) == '[[1, 8]]') & (effb0_perf_df['estimated_e2e_latency'] <= efficientnetb0_latency_constraint)& (effb0_perf_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])
edge_accuracy_perf_vggish = vggish_perf_df.loc[(vggish_perf_df['placement'].astype(str) == '[0]') & (vggish_perf_df['partitioning'].astype(str) == '[[1, 4]]') & (vggish_perf_df['estimated_e2e_latency'] <= vggish_latency_constraint) & (vggish_perf_df['destination_tier']==destination_tier)]\
                             .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])


# best cloud only in each model
cloud_accuracy_perf_resnet20 = resnet20_perf_df.loc[(resnet20_perf_df['placement'].astype(str) == '[2]') & (resnet20_perf_df['partitioning'].astype(str) == '[[1, 10]]') & (resnet20_perf_df['estimated_e2e_latency'] <= resnet20_latency_constraint) & (resnet20_perf_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])
cloud_accuracy_perf_resnet110 = resnet110_perf_df.loc[(resnet110_perf_df['placement'].astype(str) == '[2]') & (resnet110_perf_df['partitioning'].astype(str) == '[[1, 55]]') & (resnet110_perf_df['estimated_e2e_latency'] <= resnet110_latency_constraint) & (resnet110_perf_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])
cloud_accuracy_perf_effnetb0 = effb0_perf_df.loc[(effb0_perf_df['placement'].astype(str) == '[2]') & (effb0_perf_df['partitioning'].astype(str) == '[[1, 8]]') & (effb0_perf_df['estimated_e2e_latency'] <= efficientnetb0_latency_constraint)& (effb0_perf_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])
cloud_accuracy_perf_vggish = vggish_perf_df.loc[(vggish_perf_df['placement'].astype(str) == '[2]') & (vggish_perf_df['partitioning'].astype(str) == '[[1, 4]]') & (vggish_perf_df['estimated_e2e_latency'] <= vggish_latency_constraint) & (vggish_perf_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])

# best ns in each model
ns_accuracy_perf_resnet20 = resnet20_perf_df.loc[(resnet20_perf_df['partial_flag'] == False) & (resnet20_perf_df['threshold'] == 0) & (resnet110_perf_df['placement'].astype(str)!='[0, 1]') & (resnet110_perf_df['placement'].astype(str)!='[1]') & (resnet20_perf_df['estimated_e2e_latency'] <= resnet20_latency_constraint)]\
                        .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])
                        
ns_accuracy_perf_resnet110 = resnet110_perf_df.loc[(resnet110_perf_df['partial_flag'] == False) & (resnet110_perf_df['threshold'] == 0) & (resnet110_perf_df['placement'].astype(str)!='[0, 1]') & (resnet110_perf_df['placement'].astype(str)!='[1]') & (resnet110_perf_df['estimated_e2e_latency'] <= resnet110_latency_constraint)]\
                        .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])
                        
ns_accuracy_perf_effnetb0 = effb0_perf_df.loc[(effb0_perf_df['partial_flag'] == False) & (effb0_perf_df['threshold'] == 0) & (effb0_perf_df['placement'].astype(str)!='[0, 1]') & (effb0_perf_df['placement'].astype(str)!='[1]') &  (effb0_perf_df['estimated_e2e_latency'] <= efficientnetb0_latency_constraint)]\
                        .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])
                        
ns_accuracy_perf_vggish = vggish_perf_df.loc[(vggish_perf_df['partial_flag'] == False) & (vggish_perf_df['threshold'] == 0) & (resnet110_perf_df['placement'].astype(str)!='[0, 1]') & (resnet110_perf_df['placement'].astype(str)!='[1]')& (vggish_perf_df['estimated_e2e_latency'] <= vggish_latency_constraint)]\
                        .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])

# best our method in each model
our_accuracy_perf_resnet20 = resnet20_perf_df.loc[(resnet20_perf_df['estimated_e2e_latency'] <= resnet20_latency_constraint)]\
                        .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])
our_accuracy_perf_resnet110 = resnet110_perf_df.loc[(resnet110_perf_df['estimated_e2e_latency'] <= resnet110_latency_constraint)]\
                        .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])
our_accuracy_perf_effnetb0 = effb0_perf_df.loc[(effb0_perf_df['estimated_e2e_latency'] <= efficientnetb0_latency_constraint)]\
                        .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])
our_accuracy_perf_vggish = vggish_perf_df.loc[(vggish_perf_df['estimated_e2e_latency'] <= vggish_latency_constraint)]\
                        .sort_values(by=['estimated_accuracy', 'estimated_e2e_latency'], ascending=[False, True])
                



# Define the data for the plot
edge_accuracy_perf_resnet20_0 = edge_accuracy_perf_resnet20['estimated_accuracy'].values[0] if len(edge_accuracy_perf_resnet20.values) > 0 else 0
edge_accuracy_perf_resnet110_0 = edge_accuracy_perf_resnet110['estimated_accuracy'].values[0] if len(edge_accuracy_perf_resnet110.values) > 0 else 0
edge_accuracy_perf_effnetb0_0 = edge_accuracy_perf_effnetb0['estimated_accuracy'].values[0] if len(edge_accuracy_perf_effnetb0.values) > 0 else 0
edge_accuracy_perf_vggish_0 = edge_accuracy_perf_vggish['estimated_accuracy'].values[0] if len(edge_accuracy_perf_vggish.values) > 0 else 0

cloud_accuracy_perf_resnet20_0 = cloud_accuracy_perf_resnet20['estimated_accuracy'].values[0] if len(cloud_accuracy_perf_resnet20.values) > 0 else 0
cloud_accuracy_perf_resnet110_0 = cloud_accuracy_perf_resnet110['estimated_accuracy'].values[0] if len(cloud_accuracy_perf_resnet110.values) > 0 else 0
cloud_accuracy_perf_effnetb0_0 = cloud_accuracy_perf_effnetb0['estimated_accuracy'].values[0] if len(cloud_accuracy_perf_effnetb0.values) > 0 else 0
cloud_accuracy_perf_vggish_0 = cloud_accuracy_perf_vggish['estimated_accuracy'].values[0] if len(cloud_accuracy_perf_vggish.values) > 0 else 0

ns_accuracy_perf_resnet20_0 = ns_accuracy_perf_resnet20['estimated_accuracy'].values[0] if len(ns_accuracy_perf_resnet20.values) > 0 else 0
ns_accuracy_perf_resnet110_0 = ns_accuracy_perf_resnet110['estimated_accuracy'].values[0] if len(ns_accuracy_perf_resnet110.values) > 0 else 0
ns_accuracy_perf_effnetb0_0 = ns_accuracy_perf_effnetb0['estimated_accuracy'].values[0] if len(ns_accuracy_perf_effnetb0.values) > 0 else 0
ns_accuracy_perf_vggish_0 = ns_accuracy_perf_vggish['estimated_accuracy'].values[0] if len(ns_accuracy_perf_vggish.values) > 0 else 0

our_accuracy_perf_resnet20_0 = our_accuracy_perf_resnet20['estimated_accuracy'].values[0] if len(our_accuracy_perf_resnet20.values) > 0 else 0
our_accuracy_perf_resnet110_0 = our_accuracy_perf_resnet110['estimated_accuracy'].values[0] if len(our_accuracy_perf_resnet110.values) > 0 else 0
our_accuracy_perf_effnetb0_0 = our_accuracy_perf_effnetb0['estimated_accuracy'].values[0] if len(our_accuracy_perf_effnetb0.values) > 0 else 0
our_accuracy_perf_vggish_0 = our_accuracy_perf_vggish['estimated_accuracy'].values[0] if len(our_accuracy_perf_vggish.values) > 0 else 0



bar1 = [cloud_accuracy_perf_resnet20_0, cloud_accuracy_perf_resnet110_0, cloud_accuracy_perf_effnetb0_0, cloud_accuracy_perf_vggish_0]
bar2 = [edge_accuracy_perf_resnet20_0, edge_accuracy_perf_resnet110_0, edge_accuracy_perf_effnetb0_0, edge_accuracy_perf_vggish_0]
bar3 = [ns_accuracy_perf_resnet20_0, ns_accuracy_perf_resnet110_0, ns_accuracy_perf_effnetb0_0, ns_accuracy_perf_vggish_0]
bar4 = [our_accuracy_perf_resnet20_0, our_accuracy_perf_resnet110_0, our_accuracy_perf_effnetb0_0, our_accuracy_perf_vggish_0]

labels = ['ResNet20+CIFAR10', 'ResNet110+CIFAR10', 'EfficientNetB0+ImageNet', 'VGGish+AudioSet']


# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Set the bar width and positions
width = 0.2
x_pos1 = np.arange(len(bar1))
x_pos2 = [x + width for x in x_pos1]
x_pos3 = [x + width for x in x_pos2]
x_pos4 = [x + width for x in x_pos3]

# Plot the bars
rects1 = ax.bar(x_pos1, bar1, width, label='cloud only', color='darkblue')
rects2 = ax.bar(x_pos2, bar2, width, label='edge only', color='darkgreen')
rects3 = ax.bar(x_pos3, bar3, width, label='neurosurgeon', color='darkred')
rects4 = ax.bar(x_pos4, bar4, width, label='our method', color='darkorange')

# Add labels, title, and legend
ax.set_ylabel('Maximized Accuracy')
ax.set_title('accuracy comparison (destination tier0)/performance model results')
# ax.set_xlabel('Parts')
ax.set_xticks([x + 1.5*width for x in range(len(labels))])
ax.set_xticklabels(labels)
ax.legend()

# Show the plot
plt.show()



# now that we did the performance model plot, lets do the experiment plot
# first we need to get the best config from perfomance model
# but edge only and cloud only approach don't need that
# best edge only in each model
edge_accuracy_exp_resnet20 = resnet20_exp_df.loc[(resnet20_exp_df['placement'].astype(str) == '[0]') & (resnet20_exp_df['partitioning'].astype(str) == '[[1, 10]]') & (resnet20_exp_df['e2e_tail_mean'] <= resnet20_latency_constraint) & (resnet20_exp_df['destination_tier']==destination_tier) & (resnet20_exp_df['source_wait']==270) ]\
                            .sort_values(by=['accuracy', 'e2e_tail_mean'], ascending=[True, False])
edge_accuracy_exp_resnet110 = resnet110_exp_df.loc[(resnet110_exp_df['placement'].astype(str) == '[0]') & (resnet110_exp_df['partitioning'].astype(str) == '[[1, 55]]') & (resnet110_exp_df['e2e_tail_mean'] <= resnet110_latency_constraint) & (resnet110_exp_df['destination_tier']==destination_tier) & (resnet110_exp_df['source_wait']==270)]\
                            .sort_values(by=['accuracy', 'e2e_tail_mean'], ascending=[True, False])
edge_accuracy_exp_effnetb0 = effb0_exp_df.loc[(effb0_exp_df['placement'].astype(str) == '[0]') & (effb0_exp_df['partitioning'].astype(str) == '[[1, 8]]') & (effb0_exp_df['e2e_tail_mean'] <= efficientnetb0_latency_constraint)& (effb0_exp_df['destination_tier']==destination_tier) & (effb0_exp_df['source_wait']==707) ]\
                            .sort_values(by=['accuracy', 'e2e_tail_mean'], ascending=[True, False])
edge_accuracy_exp_vggish = vggish_exp_df.loc[(vggish_exp_df['placement'].astype(str) == '[0]') & (vggish_exp_df['partitioning'].astype(str) == '[[1, 4]]') & (vggish_exp_df['e2e_tail_mean'] <= vggish_latency_constraint) & (vggish_exp_df['destination_tier']==destination_tier) ]\
                             .sort_values(by=['accuracy', 'e2e_tail_mean'], ascending=[True, False])


# best cloud only in each model
cloud_accuracy_exp_resnet20 = resnet20_exp_df.loc[(resnet20_exp_df['placement'].astype(str) == '[2]') & (resnet20_exp_df['partitioning'].astype(str) == '[[1, 10]]') & (resnet20_exp_df['e2e_tail_mean'] <= resnet20_latency_constraint) & (resnet20_exp_df['destination_tier']==destination_tier) & (resnet20_exp_df['source_wait']==270)]\
                            .sort_values(by=['accuracy', 'e2e_tail_mean'], ascending=[True, False])
cloud_accuracy_exp_resnet110 = resnet110_exp_df.loc[(resnet110_exp_df['placement'].astype(str) == '[2]') & (resnet110_exp_df['partitioning'].astype(str) == '[[1, 55]]') & (resnet110_exp_df['e2e_tail_mean'] <= resnet110_latency_constraint) & (resnet110_exp_df['destination_tier']==destination_tier) & (resnet110_exp_df['source_wait']==270)]\
                            .sort_values(by=['accuracy', 'e2e_tail_mean'], ascending=[True, False])
cloud_accuracy_exp_effnetb0 = effb0_exp_df.loc[(effb0_exp_df['placement'].astype(str) == '[2]') & (effb0_exp_df['partitioning'].astype(str) == '[[1, 8]]') & (effb0_exp_df['e2e_tail_mean'] <= efficientnetb0_latency_constraint)& (effb0_exp_df['destination_tier']==destination_tier) & (effb0_exp_df['source_wait']==707)]\
                            .sort_values(by=['accuracy', 'e2e_tail_mean'], ascending=[True, False])
cloud_accuracy_exp_vggish = vggish_exp_df.loc[(vggish_exp_df['placement'].astype(str) == '[2]') & (vggish_exp_df['partitioning'].astype(str) == '[[1, 4]]') & (vggish_exp_df['e2e_tail_mean'] <= vggish_latency_constraint) & (vggish_exp_df['destination_tier']==destination_tier)]\
                            .sort_values(by=['accuracy', 'e2e_tail_mean'], ascending=[True, False])

# for ns and our method, we need to get the best config from performance model
print("ns best config")
print("resnet20\n", ns_accuracy_perf_resnet20.head(10)[['partitioning', 'placement', 'threshold', 'estimated_accuracy']].values)
print("resnet110\n", ns_accuracy_perf_resnet110.head(10)[['partitioning', 'placement', 'threshold', 'estimated_accuracy']].values)
print("effb0\n", ns_accuracy_perf_effnetb0.head(10)[['partitioning', 'placement', 'threshold', 'estimated_accuracy']].values)
print("vggish\n", ns_accuracy_perf_vggish.head(10)[['partitioning', 'placement', 'threshold', 'estimated_accuracy']].values)

print("our best config")
print("resnet20\n", our_accuracy_perf_resnet20.head(10)[['partitioning', 'placement', 'threshold', 'estimated_accuracy']].values)
print("resnet110\n", our_accuracy_perf_resnet110.head(10)[['partitioning', 'placement', 'threshold', 'estimated_accuracy']].values)
print("effb0\n", our_accuracy_perf_effnetb0.head(10)[['partitioning', 'placement', 'threshold', 'estimated_accuracy']].values)
print("vggish\n", our_accuracy_perf_vggish.head(10)[['partitioning', 'placement', 'threshold', 'estimated_accuracy']].values)

# chosen ones:
# NS
# resnet20
#  [[list([[1, 10]]) list([0]) 0.0 ]]
ns_accuracy_exp_resnet20 = resnet20_exp_df.loc[(resnet20_exp_df['placement'].astype(str) == '[0]') & (resnet20_exp_df['partitioning'].astype(str) == '[[1, 10]]') & (resnet20_exp_df['threshold'] == 0) & (resnet20_exp_df['destination_tier']==destination_tier) & (resnet20_exp_df['source_wait']==270) & (resnet20_exp_df['model_mode']=='full')]\
                            .sort_values(by=['accuracy', 'e2e_tail_mean'], ascending=[True, False])

# resnet110
#  []
ns_accuracy_exp_resnet110 = resnet110_exp_df.loc[(resnet110_exp_df['placement'].astype(str) == 'EMPTY')]
# effb0
# []
ns_accuracy_exp_effnetb0 = effb0_exp_df.loc[(effb0_exp_df['placement'].astype(str) == 'EMPTY') ]
                            
# vggish
#  [[list([[1, 4]]) list([2]) 0.0 ]
ns_accuracy_exp_vggish = vggish_exp_df.loc[(vggish_exp_df['placement'].astype(str) == '[2]') & (vggish_exp_df['partitioning'].astype(str) == '[[1, 4]]') & (vggish_exp_df['threshold'] == 0) & (vggish_exp_df['destination_tier']==destination_tier) & (vggish_exp_df['model_mode']=='full')]\
                            .sort_values(by=['accuracy', 'e2e_tail_mean'], ascending=[True, False])


# OUR
# resnet20
#  [[list([[1, 9]]) list([0]) 0.0 0.8767999999998246]
our_accuracy_exp_resnet20 = resnet20_exp_df.loc[(resnet20_exp_df['placement'].astype(str) == '[0]') & (resnet20_exp_df['partitioning'].astype(str) == '[[1, 9]]') & (resnet20_exp_df['threshold'] == 0)]
# resnet110
#  [list([[1, 29], [30, 38]]) list([0, 2]) 0.7 0.88]
our_accuracy_exp_resnet110 = resnet110_exp_df.loc[(resnet110_exp_df['placement'].astype(str) == '[0, 2]') & (resnet110_exp_df['partitioning'].astype(str) == '[[1, 29], [30, 38]]') & (resnet110_exp_df['threshold'] == 0.7)]

#   [[list([[1, 4], [5, 8]]) list([0, 1]) 0.0 0.7385599999999705]
our_accuracy_exp_effnetb0 = effb0_exp_df.loc[(effb0_exp_df['placement'].astype(str) == '[0, 1]') & (effb0_exp_df['partitioning'].astype(str) == '[[1, 4], [5, 8]]') & (effb0_exp_df['threshold'] == 0)]
# vggish
#  [[list([[1, 4]]) list([2]) 0.0 0.7167999999998566]
our_accuracy_exp_vggish = vggish_exp_df.loc[(vggish_exp_df['placement'].astype(str) == '[2]') & (vggish_exp_df['partitioning'].astype(str) == '[[1, 4]]') & (vggish_exp_df['threshold'] == 0)]






print("\n\n\n\nNS: do they exist in exp?")
print(ns_accuracy_exp_resnet20[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']].values)
print(ns_accuracy_exp_resnet110[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']].values)
print(ns_accuracy_exp_effnetb0[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']].values)
print(ns_accuracy_exp_vggish[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']].values)


print("OUR: do they exist in exp?")
print(our_accuracy_exp_resnet20[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']].values)
print(our_accuracy_exp_resnet110[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']].values)
print(our_accuracy_exp_effnetb0[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']].values)
print(our_accuracy_exp_vggish[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']].values)






edge_accuracy_exp_resnet20_0 = edge_accuracy_exp_resnet20['accuracy'].values[0] if len(edge_accuracy_exp_resnet20.values) > 0 else 0
edge_accuracy_exp_resnet110_0 = edge_accuracy_exp_resnet110['accuracy'].values[0] if len(edge_accuracy_exp_resnet110.values) > 0 else 0
edge_accuracy_exp_effnetb0_0 = edge_accuracy_exp_effnetb0['accuracy'].values[0] if len(edge_accuracy_exp_effnetb0.values) > 0 else 0
edge_accuracy_exp_vggish_0 = edge_accuracy_exp_vggish['accuracy'].values[0] if len(edge_accuracy_exp_vggish.values) > 0 else 0

cloud_accuracy_exp_resnet20_0 = cloud_accuracy_exp_resnet20['accuracy'].values[0] if len(cloud_accuracy_exp_resnet20.values) > 0 else 0
cloud_accuracy_exp_resnet110_0 = cloud_accuracy_exp_resnet110['accuracy'].values[0] if len(cloud_accuracy_exp_resnet110.values) > 0 else 0
cloud_accuracy_exp_effnetb0_0 = cloud_accuracy_exp_effnetb0['accuracy'].values[0] if len(cloud_accuracy_exp_effnetb0.values) > 0 else 0
cloud_accuracy_exp_vggish_0 = cloud_accuracy_exp_vggish['accuracy'].values[0] if len(cloud_accuracy_exp_vggish.values) > 0 else 0

ns_accuracy_exp_resnet20_0 = ns_accuracy_exp_resnet20['accuracy'].values[0] if len(ns_accuracy_exp_resnet20.values) > 0 else 0
ns_accuracy_exp_resnet110_0 = ns_accuracy_exp_resnet110['accuracy'].values[0] if len(ns_accuracy_exp_resnet110.values) > 0 else 0
ns_accuracy_exp_effnetb0_0 = ns_accuracy_exp_effnetb0['accuracy'].values[0] if len(ns_accuracy_exp_effnetb0.values) > 0 else 0
ns_accuracy_exp_vggish_0 = ns_accuracy_exp_vggish['accuracy'].values[0] if len(ns_accuracy_exp_vggish.values) > 0 else 0

our_accuracy_exp_resnet20_0 = our_accuracy_exp_resnet20['accuracy'].values[0] if len(our_accuracy_exp_resnet20.values) > 0 else 0
our_accuracy_exp_resnet110_0 = our_accuracy_exp_resnet110['accuracy'].values[0] if len(our_accuracy_exp_resnet110.values) > 0 else 0
our_accuracy_exp_effnetb0_0 = our_accuracy_exp_effnetb0['accuracy'].values[0] if len(our_accuracy_exp_effnetb0.values) > 0 else 0
our_accuracy_exp_vggish_0 = our_accuracy_exp_vggish['accuracy'].values[0] if len(our_accuracy_exp_vggish.values) > 0 else 0




bar1 = [cloud_accuracy_exp_resnet20_0, cloud_accuracy_exp_resnet110_0, cloud_accuracy_exp_effnetb0_0, cloud_accuracy_exp_vggish_0+0.001]
bar2 = [edge_accuracy_exp_resnet20_0, edge_accuracy_exp_resnet110_0, edge_accuracy_exp_effnetb0_0, edge_accuracy_exp_vggish_0]
bar3 = [ns_accuracy_exp_resnet20_0, ns_accuracy_exp_resnet110_0, ns_accuracy_exp_effnetb0_0, ns_accuracy_exp_vggish_0+0.001]
bar4 = [our_accuracy_exp_resnet20_0, our_accuracy_exp_resnet110_0, our_accuracy_exp_effnetb0_0, our_accuracy_exp_vggish_0]

labels = ['ResNet20+CIFAR10', 'ResNet110+CIFAR10', 'EfficientNetB0+ImageNet', 'VGGish+AudioSet'] 

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Set the bar width and positions
width = 0.2
x_pos1 = np.arange(len(bar1))
x_pos2 = [x + width for x in x_pos1]
x_pos3 = [x + width for x in x_pos2]
x_pos4 = [x + width for x in x_pos3]

# Plot the bars
rects1 = ax.bar(x_pos1, bar1, width, label='cloud only', color='darkblue')
rects2 = ax.bar(x_pos2, bar2, width, label='edge only', color='darkgreen')
rects3 = ax.bar(x_pos3, bar3, width, label='neurosurgeon', color='darkred')
rects4 = ax.bar(x_pos4, bar4, width, label='our method', color='darkorange')


# Add text annotations to bars with zero height
for rects in [rects1, rects2, rects3, rects4]:
    for i, rect in enumerate(rects):
        if rect.get_height() == 0:
            ax.text(rect.get_x() + rect.get_width()/2, 0.005, 'X', weight='bold', ha='center', va='bottom')


# Add labels, title, and legend
ax.set_ylabel('Maximized Accuracy')
ax.set_title('Problem2: Maximizing Accuracy')
ax.set_xticks([x + 1.5*width for x in range(len(labels))])
ax.set_xticklabels(labels)
ax.legend()

# Show the plot
plt.show()