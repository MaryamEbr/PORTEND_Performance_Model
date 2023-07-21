#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



experiment_folder = 'exp_folder/'
destination_tier = 0
acc_const= 0.824
lat_const = 170

#### loading the dataframes for  experiments
resnet110_df = pd.read_pickle(f'{experiment_folder}Experiment72/total_df.pkl')

resnet110_df_our_final_all = resnet110_df.loc[ (resnet110_df['accuracy']>0.75)  & (resnet110_df['accuracy']<0.928) & (resnet110_df['model_mode']=='final' ) & (resnet110_df['e2e_tail_mean']<200) ].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])

#### to only show the points closer to pareto frontier
resnet110_df = resnet110_df.drop(resnet110_df[(resnet110_df['e2e_tail_mean']>65) & (resnet110_df['accuracy']<0.9225)].index)
resnet110_df = resnet110_df.drop(resnet110_df[(resnet110_df['e2e_tail_mean']>60) & (resnet110_df['accuracy']<0.92)].index)
resnet110_df = resnet110_df.drop(resnet110_df[(resnet110_df['e2e_tail_mean']>55) & (resnet110_df['accuracy']<0.885)].index)

resnet110_df = resnet110_df.drop(resnet110_df[(resnet110_df['e2e_tail_mean']>50) & (resnet110_df['accuracy']<0.864)].index)
resnet110_df = resnet110_df.drop(resnet110_df[(resnet110_df['e2e_tail_mean']>40) & (resnet110_df['accuracy']<0.842)].index)
resnet110_df = resnet110_df.drop(resnet110_df[(resnet110_df['e2e_tail_mean']>35) & (resnet110_df['accuracy']<0.845)].index)
resnet110_df = resnet110_df.drop(resnet110_df[(resnet110_df['e2e_tail_mean']>30) & (resnet110_df['accuracy']<0.834)].index)
resnet110_df = resnet110_df.drop(resnet110_df[(resnet110_df['e2e_tail_mean']>35) & (resnet110_df['accuracy']<0.833)].index)

# our method df

resnet110_df_our_final_1 = resnet110_df.loc[(resnet110_df['accuracy']>acc_const) & (resnet110_df['model_mode']=='final') & (resnet110_df['num_partitions']==1) & (resnet110_df['e2e_tail_mean']<lat_const)].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
resnet110_df_our_final_2 = resnet110_df.loc[(resnet110_df['accuracy']>acc_const) & (resnet110_df['accuracy']<0.928) & (resnet110_df['model_mode']=='final') & (resnet110_df['num_partitions']==2)& (resnet110_df['e2e_tail_mean']<lat_const)].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
resnet110_df_our_final_3 = resnet110_df.loc[(resnet110_df['accuracy']>acc_const) & (resnet110_df['accuracy']<0.928) & (resnet110_df['model_mode']=='final') & (resnet110_df['num_partitions']==3)& (resnet110_df['e2e_tail_mean']<lat_const)].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])

print("---------------------- 1p")
print(resnet110_df_our_final_1[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']])
print("---------------------- 2p")
print(resnet110_df_our_final_2[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']])
print("---------------------- 3p")
print(resnet110_df_our_final_3[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']])

# cloud method df
resnet110_df_cloud = resnet110_df.loc[(resnet110_df['accuracy']>acc_const) &(resnet110_df['model_mode']=='full') & (resnet110_df['source_wait']==270) & (resnet110_df['placement'].astype(str)=='[2]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False]).head(1)

# edge method df
resnet110_df_edge = resnet110_df.loc[(resnet110_df['accuracy']>acc_const) &(resnet110_df['model_mode']=='full') & (resnet110_df['source_wait']==270) & (resnet110_df['placement'].astype(str)=='[0]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False]).head(1)

# core
resnet110_df_core = resnet110_df.loc[(resnet110_df['accuracy']>acc_const) &(resnet110_df['model_mode']=='full') & (resnet110_df['source_wait']==270) & (resnet110_df['placement'].astype(str)=='[1]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False]).head(1)

# ns method df
resnet110_df_ns = resnet110_df.loc[(resnet110_df['accuracy']>acc_const) &(resnet110_df['model_mode']=='full') & (resnet110_df['source_wait']==270) & (resnet110_df['placement'].astype(str)!='[0, 1]') & (resnet110_df['placement'].astype(str)!='[1]') ].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False]).head(1)
 
 
our_all_latency = resnet110_df_our_final_all['e2e_tail_mean'].values
our_all_accuracy = resnet110_df_our_final_all['accuracy'].values

our_1p_latency = resnet110_df_our_final_1['e2e_tail_mean'].values
our_1p_accuracy = resnet110_df_our_final_1['accuracy'].values

our_2p_latency = resnet110_df_our_final_2['e2e_tail_mean'].values
our_2p_accuracy = resnet110_df_our_final_2['accuracy'].values

our_3p_latency = resnet110_df_our_final_3['e2e_tail_mean'].values
our_3p_accuracy = resnet110_df_our_final_3['accuracy'].values

cloud_latency = resnet110_df_cloud['e2e_tail_mean'].values[0]
cloud_accuracy = resnet110_df_cloud['accuracy'].values[0]

edge_latency = resnet110_df_edge['e2e_tail_mean'].values[0]
edge_accuracy = resnet110_df_edge['accuracy'].values[0]

core_latency = resnet110_df_core['e2e_tail_mean'].values[0]
core_accuracy = resnet110_df_core['accuracy'].values[0]

ns_latency = resnet110_df_ns['e2e_tail_mean'].values[0]
ns_accuracy = resnet110_df_ns['accuracy'].values[0]

# plotting the tradeoff
fig, ax = plt.subplots(figsize=(9, 7))

# plt.scatter(our_all_latency, our_all_accuracy, marker='o', color='silver', label='all partitions')

plt.scatter(our_3p_latency, our_3p_accuracy, marker='o', color='salmon', label='PORTEND - 3 partitions')
plt.scatter(our_2p_latency, our_2p_accuracy, marker='o', color='violet', label='PORTEND - 2 partitions')
plt.scatter(our_1p_latency, our_1p_accuracy, marker='o', color='dodgerblue', label='PORTEND - 1 partition')


# plt.scatter(our_3p_latency, our_3p_accuracy, marker='o', color='orange', label='our method')
# plt.scatter(our_2p_latency, our_2p_accuracy, marker='o', color='orange')
# plt.scatter(our_1p_latency, our_1p_accuracy, marker='o', color='orange')




plt.scatter(cloud_latency, cloud_accuracy, marker='o', s=200, color='darkblue', label='cloud-only')
plt.scatter(edge_latency, edge_accuracy, marker='o', s=200, color='darkgreen', label='edge-only')
# plt.scatter(core_latency, core_accuracy, marker='*', s=200, color='darkgreen', label='core only')

plt.scatter(ns_latency, ns_accuracy, marker='X', s=150, color='firebrick', label='Neurosurgeon')


plt.title(f"ResNet110(final fine-tuned models)+CIFAR10")
plt.legend(title='placement/partitioning', loc='lower right')
plt.grid()
plt.xlabel('End to End Latency (ms)')
plt.ylabel('Accuracy')
plt.show()
