#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['font.size'] = 11

experiment_folder = 'exp_folder/'
destination_tier = 0
acc_const= 0.40
lat_const = 1700

#### loading the dataframes for  experiments
resnet110_df = pd.read_pickle(f'{experiment_folder}Experiment72/total_df.pkl')
resnet110_df_before = pd.read_pickle(f'{experiment_folder}Experiment72/total_df.pkl')
resnet110_df = resnet110_df.loc[ (resnet110_df['accuracy']>0.75)  & (resnet110_df['accuracy']<0.928) & (resnet110_df['model_mode']=='final' ) & (resnet110_df['e2e_tail_mean']<200) ].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])


# filter the dataframe resnet110_df for 1p 2p 3p with only these point, which are showing partitioning, placement, threshold, e2e_tail_mean, accuracy
# 
#1p
#[[1, 41]]	[0]	0.0	69.731053	0.9210
#[[1, 35]]	[0]	0.0	61.872971	0.9088
#[[1, 30]]	[0]	0.0	53.946276	0.9052
#[[1, 25]]	[0]	0.0	45.890207	0.8954
#[[1, 20]]	[0]	0.0	38.297153	0.8600
#[[1, 15]]	[0]	0.0	31.358391	0.8368
#[[1, 10]]	[0]	0.0	21.235730	0.8234

resnet110_df_our_final_1 = resnet110_df.loc [
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 41]]') & (resnet110_df['placement'].astype(str) == '[0]') & (resnet110_df['threshold'] == 0.0)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 35]]') & (resnet110_df['placement'].astype(str) == '[0]') & (resnet110_df['threshold'] == 0.0)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 30]]') & (resnet110_df['placement'].astype(str) == '[0]') & (resnet110_df['threshold'] == 0.0)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 25]]') & (resnet110_df['placement'].astype(str) == '[0]') & (resnet110_df['threshold'] == 0.0)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 20]]') & (resnet110_df['placement'].astype(str) == '[0]') & (resnet110_df['threshold'] == 0.0)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 15]]') & (resnet110_df['placement'].astype(str) == '[0]') & (resnet110_df['threshold'] == 0.0)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 10]]') & (resnet110_df['placement'].astype(str) == '[0]') & (resnet110_df['threshold'] == 0.0))\
                                            ]


#2p
#[[1, 41], [42, 46]]	[0, 1]	0.0	135.437439	0.9270 
#[[1, 54], [55, 55]]	[1, 2]	0.9	110.005906	0.9274
#[[1, 41], [42, 46]]	[0, 1]	0.8	67.565562	0.9256
#[[1, 25], [26, 30]]	[0, 1]	1.0	44.543567	0.8982
#[[1, 20], [21, 25]]	[0, 1]	0.3	41.939554	0.8692
#[[1, 20], [21, 25]]	[0, 2]	0.8	36.974708	0.8636
#[[1, 15], [16, 20]]	[0, 1]	1.0	29.587835	0.8410

resnet110_df_our_final_2 = resnet110_df.loc [((resnet110_df['partitioning'].astype(str) == '[[1, 41], [42, 46]]') & (resnet110_df['placement'].astype(str) == '[0, 1]') & (resnet110_df['threshold'] == 0.0)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 54], [55, 55]]') & (resnet110_df['placement'].astype(str) == '[1, 2]') & (resnet110_df['threshold'] == 0.9)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 41], [42, 46]]') & (resnet110_df['placement'].astype(str) == '[0, 1]') & (resnet110_df['threshold'] == 0.8)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 25], [26, 30]]') & (resnet110_df['placement'].astype(str) == '[0, 1]') & (resnet110_df['threshold'] == 1.0)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 20], [21, 25]]') & (resnet110_df['placement'].astype(str) == '[0, 1]') & (resnet110_df['threshold'] == 0.3)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 20], [21, 25]]') & (resnet110_df['placement'].astype(str) == '[0, 2]') & (resnet110_df['threshold'] == 0.8)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 15], [16, 20]]') & (resnet110_df['placement'].astype(str) == '[0, 1]') & (resnet110_df['threshold'] == 1.0))\
                                            ]

#3p
#[[1, 48], [49, 49], [50, 51]]	[0, 1, 2]	0.2	101.407904	0.9254
#[[1, 30], [31, 35], [36, 40]]	[0, 1, 2]	0.6	51.701229	0.9122
#[[1, 10], [11, 15], [16, 20]]	[0, 1, 2]	1.0	19.801318	0.8286
#[[1, 10], [11, 15], [16, 20]]	[0, 1, 2]	0.4	27.820019	0.8336


resnet110_df_our_final_3 = resnet110_df.loc [((resnet110_df['partitioning'].astype(str) == '[[1, 48], [49, 49], [50, 51]]') & (resnet110_df['placement'].astype(str) == '[0, 1, 2]') & (resnet110_df['threshold'] == 0.2)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 30], [31, 35], [36, 40]]') & (resnet110_df['placement'].astype(str) == '[0, 1, 2]') & (resnet110_df['threshold'] == 0.6)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 10], [11, 15], [16, 20]]') & (resnet110_df['placement'].astype(str) == '[0, 1, 2]') & (resnet110_df['threshold'] == 1.0)) |\
                                            ((resnet110_df['partitioning'].astype(str) == '[[1, 10], [11, 15], [16, 20]]') & (resnet110_df['placement'].astype(str) == '[0, 1, 2]') & (resnet110_df['threshold'] == 0.4))\
                                            ]





with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    print("---------------------- 1p")
    print(resnet110_df_our_final_1[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']])
    print("---------------------- 2p")
    print(resnet110_df_our_final_2[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']])
    print("---------------------- 3p")
    print(resnet110_df_our_final_3[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']])




resnet110_df = resnet110_df_before
# with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
#     print("------------------------------------------------------------------------------------------")
#     print(resnet110_df[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy']])

# cloud method df
resnet110_df_cloud = resnet110_df.loc[(resnet110_df['accuracy']>acc_const) &(resnet110_df['model_mode']=='full') & (resnet110_df['source_wait']==270) & (resnet110_df['placement'].astype(str)=='[2]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False]).head(1)

# edge method df
resnet110_df_edge = resnet110_df.loc[(resnet110_df['accuracy']>acc_const) &(resnet110_df['model_mode']=='full') & (resnet110_df['source_wait']==270) & (resnet110_df['placement'].astype(str)=='[0]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False]).head(1)

# core
resnet110_df_core = resnet110_df.loc[(resnet110_df['accuracy']>acc_const) &(resnet110_df['model_mode']=='full') & (resnet110_df['source_wait']==270) & (resnet110_df['placement'].astype(str)=='[1]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False]).head(1)

# ns method df
resnet110_df_ns = resnet110_df.loc[(resnet110_df['accuracy']>acc_const) &(resnet110_df['model_mode']=='full') & (resnet110_df['source_wait']==270) & (resnet110_df['placement'].astype(str)!='[0, 1]') & (resnet110_df['placement'].astype(str)!='[1]') ].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False]).head(1)
 
 

cloud_latency = resnet110_df_cloud['e2e_tail_mean'].values[0]
cloud_accuracy = resnet110_df_cloud['accuracy'].values[0]

edge_latency = resnet110_df_edge['e2e_tail_mean'].values[0]
edge_accuracy = resnet110_df_edge['accuracy'].values[0]

core_latency = resnet110_df_core['e2e_tail_mean'].values[0]
core_accuracy = resnet110_df_core['accuracy'].values[0]

ns_latency = resnet110_df_ns['e2e_tail_mean'].values[0]
ns_accuracy = resnet110_df_ns['accuracy'].values[0]





our_1p_latency = resnet110_df_our_final_1['e2e_tail_mean'].values
our_1p_accuracy = resnet110_df_our_final_1['accuracy'].values

our_2p_latency = resnet110_df_our_final_2['e2e_tail_mean'].values
our_2p_accuracy = resnet110_df_our_final_2['accuracy'].values

our_3p_latency = resnet110_df_our_final_3['e2e_tail_mean'].values
our_3p_accuracy = resnet110_df_our_final_3['accuracy'].values


# plotting the tradeoff
r=1.7
fig, ax = plt.subplots(figsize=(r*3, r*2))  
ax.grid(color='lightgray')


plt.scatter(our_1p_latency, our_1p_accuracy, marker='o', s=60, color='olive', label='PORTEND - 1 partition')
plt.scatter(our_2p_latency, our_2p_accuracy, marker='^', s=60, color='violet', label='PORTEND - 2 partitions')
plt.scatter(our_3p_latency, our_3p_accuracy, marker='*', s=80, color='salmon', label='PORTEND - 3 partitions')




plt.scatter(cloud_latency, cloud_accuracy, marker='o', facecolor="none", edgecolors="blue", linewidths=2, s=150, color='darkblue', label='cloud-only')
plt.scatter(edge_latency, edge_accuracy, marker='s', facecolor="none", edgecolors="green", linewidths=2, s=150, color='darkgreen', label='edge-only')

plt.scatter(ns_latency, ns_accuracy, marker='X', s=100, color='firebrick', label='Neurosurgeon')


# plt.title(f"ResNet110(final fine-tuned models)+CIFAR10")
plt.legend(loc='lower right')

plt.xlabel('End to End Latency (ms)')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
