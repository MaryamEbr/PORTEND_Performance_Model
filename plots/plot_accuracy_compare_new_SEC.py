#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 11


experiment_folder = 'exp_folder/'
destination_tier = 0

resnet20_latency_constraint = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
resnet110_latency_constraint = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
effb0_latency_constraint = [350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 630, 640, 650, 660, 670, 680, 690, 700]
vggish_latency_constraint = [1500, 1600, 1800, 1850, 1900, 1950, 2000, 2500, 3000, 3500]


#### loading the dataframes for  experiments
resnet20_df = pd.read_pickle(f'{experiment_folder}Experiment71/total_df.pkl')
resnet110_df = pd.read_pickle(f'{experiment_folder}Experiment72/total_df.pkl')
effb0_df = pd.read_pickle(f'{experiment_folder}Experiment73/total_df.pkl')
vggish_df = pd.read_pickle(f'{experiment_folder}Experiment74/total_df.pkl')

# our method df
resnet20_df_our_full = resnet20_df.loc[(resnet20_df['model_mode']=='full') & (resnet20_df['source_wait']!=270) ].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
resnet110_df_our_full = resnet110_df.loc[(resnet110_df['model_mode']=='full') & (resnet110_df['num_partitions']==2) & (resnet110_df['source_wait']!=270)].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
effb0_df_our_full = effb0_df.loc[(effb0_df['model_mode']=='full') & (effb0_df['source_wait']!=707) ].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
vggish_df_our_full = vggish_df.loc[(vggish_df['model_mode']=='full') & (vggish_df['source_wait']!=3007) ].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])

# our method df
resnet20_df_our_final = resnet20_df.loc[(resnet20_df['model_mode']=='final')  ].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
resnet110_df_our_final = resnet110_df.loc[(resnet110_df['model_mode']=='final') ].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
effb0_df_our_final = effb0_df.loc[(effb0_df['model_mode']=='final')  ].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
vggish_df_our_final = vggish_df.loc[(vggish_df['model_mode']=='final') ].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])

# cloud method df
resnet20_df_cloud = resnet20_df.loc[(resnet20_df['model_mode']=='full') & (resnet20_df['source_wait']==270) & (resnet20_df['placement'].astype(str)=='[2]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
resnet110_df_cloud = resnet110_df.loc[(resnet110_df['model_mode']=='full') & (resnet110_df['source_wait']==270) & (resnet110_df['placement'].astype(str)=='[2]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
effb0_df_cloud = effb0_df.loc[(effb0_df['model_mode']=='full') & (effb0_df['source_wait']==707) & (effb0_df['placement'].astype(str)=='[2]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
vggish_df_cloud = vggish_df.loc[(vggish_df['model_mode']=='full') & (vggish_df['placement'].astype(str)=='[2]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])

# edge method df
resnet20_df_edge = resnet20_df.loc[(resnet20_df['model_mode']=='full') & (resnet20_df['source_wait']==270) & (resnet20_df['placement'].astype(str)=='[0]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
resnet110_df_edge = resnet110_df.loc[(resnet110_df['model_mode']=='full') & (resnet110_df['source_wait']==270) & (resnet110_df['placement'].astype(str)=='[0]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
effb0_df_edge = effb0_df.loc[(effb0_df['model_mode']=='full') & (effb0_df['source_wait']==707) & (effb0_df['placement'].astype(str)=='[0]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
vggish_df_edge = vggish_df.loc[(vggish_df['model_mode']=='full') & (vggish_df['placement'].astype(str)=='[0]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])


# ns method df
resnet20_df_ns = resnet20_df.loc[(resnet20_df['model_mode']=='full') & (resnet20_df['source_wait']==270) & (resnet20_df['placement'].astype(str)!='[0, 1]') & (resnet20_df['placement'].astype(str)!='[1]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
resnet110_df_ns = resnet110_df.loc[(resnet110_df['model_mode']=='full') & (resnet110_df['source_wait']==270) & (resnet110_df['placement'].astype(str)!='[0, 1]') & (resnet110_df['placement'].astype(str)!='[1]') ].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
effb0_df_ns = effb0_df.loc[(effb0_df['model_mode']=='full') & (effb0_df['source_wait']==707) & (effb0_df['placement'].astype(str)!='[0, 1]') & (effb0_df['placement'].astype(str)!='[1]') ].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
vggish_df_ns = vggish_df.loc[(vggish_df['model_mode']=='full') & (vggish_df['placement'].astype(str)!='[0, 1]') & (vggish_df['placement'].astype(str)!='[1]')].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     display(resnet20_df_our[['partitioning', 'placement', 'threshold', 'accuracy', 'e2e_tail_mean']])
    
    
    


latency_constraint_list = resnet110_latency_constraint
the_df_our_full = resnet110_df_our_full
the_df_our_final = resnet110_df_our_final
the_df_cloud = resnet110_df_cloud
the_df_edge = resnet110_df_edge
the_df_ns = resnet110_df_ns 


our_accuracy_list_full = []
our_accuracy_list_final = []
cloud_accuracy_list = []
edge_accuracy_list = []
ns_accuracy_list = []


for latency_constraint in latency_constraint_list:
    
    
    
    our_accuracy_full = the_df_our_full.loc[the_df_our_full['e2e_tail_mean']<=latency_constraint].sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values
    our_accuracy_final = the_df_our_final.loc[the_df_our_final['e2e_tail_mean']<=latency_constraint].sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values
    
    cloud_accuracy = the_df_cloud.loc[the_df_cloud['e2e_tail_mean']<=latency_constraint].sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values
    edge_accuracy = the_df_edge.loc[the_df_edge['e2e_tail_mean']<=latency_constraint].sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values
    ns_accuracy = the_df_ns.loc[the_df_ns['e2e_tail_mean']<=latency_constraint].sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values
    
    
    # # only for vggish
    # cloud_accuracy_list.append(cloud_accuracy[0]+0.001 if len(cloud_accuracy)>0 else 0)
    # our_accuracy_list.append(our_accuracy[0] if len(our_accuracy)>0 else 0)
    # edge_accuracy_list.append(edge_accuracy[0]+0.001 if len(edge_accuracy)>0 else 0)
    # ns_accuracy_list.append(ns_accuracy[0]+0.001 if len(ns_accuracy)>0 else 0)

    # for otheres
    cloud_accuracy_list.append(cloud_accuracy[0] if len(cloud_accuracy)>0 else 0)
    our_accuracy_list_full.append(our_accuracy_full[0] if len(our_accuracy_full)>0 else 0)
    our_accuracy_list_final.append(our_accuracy_final[0] if len(our_accuracy_final)>0 else 0)
    edge_accuracy_list.append(edge_accuracy[0] if len(edge_accuracy)>0 else 0)
    ns_accuracy_list.append(ns_accuracy[0] if len(ns_accuracy)>0 else 0)
            
            
            
    print(f'^^^^^ latency_constraint: {latency_constraint}')
    print("our method full", our_accuracy_full)
    print("our method final", our_accuracy_final)
    print("cloud method", cloud_accuracy)
    print("edge method", edge_accuracy)
    print("ns method", ns_accuracy)
    print()
    
    
    


# r=1.6
# fig, ax = plt.subplots(figsize=(r*3*1.25, r*1.4))  

r=1.7
fig, ax = plt.subplots(figsize=(r*3.8, r*2))  
# r=1.7
# fig, ax = plt.subplots(figsize=(r*3.5, r*2))  


plt.plot(latency_constraint_list[next((i for i, x in enumerate(our_accuracy_list_full) if x), None):], our_accuracy_list_full[next((i for i, x in enumerate(our_accuracy_list_full) if x), None):], label='PORTEND (no finetuning)', marker='o', color='black')
plt.plot(latency_constraint_list[next((i for i, x in enumerate(our_accuracy_list_final) if x), None):], our_accuracy_list_final[next((i for i, x in enumerate(our_accuracy_list_final) if x), None):], label='PORTEND (with finetuning)', marker='o', color='grey')



plt.scatter(latency_constraint_list[next((i for i, x in enumerate(edge_accuracy_list) if x), None)],  edge_accuracy_list[next((i for i, x in enumerate(cloud_accuracy_list) if x), None)], marker='s', facecolor="none", edgecolors="green", linewidths=2, s=150, color='darkgreen', label='edge-only', zorder=10)
plt.scatter(latency_constraint_list[next((i for i, x in enumerate(cloud_accuracy_list) if x), None)], cloud_accuracy_list[next((i for i, x in enumerate(cloud_accuracy_list) if x), None)]+0.0053, marker='o', facecolor="none", edgecolors="blue", linewidths=2, s=150, color='darkblue', label='cloud-only', zorder=10)
plt.scatter(latency_constraint_list[next((i for i, x in enumerate(ns_accuracy_list) if x), None)], ns_accuracy_list[next((i for i, x in enumerate(ns_accuracy_list) if x), None)], marker='X', s=100, color='firebrick', label='Neurosurgeon', zorder=10)


# plt.title('Problem2: Mazimizing Acuuracy (ResNet110+CIFAR10)')
plt.legend()
# plt.grid()
plt.xlabel('Latency Constraint (ms)')
plt.ylabel('Maximized Accuracy')
plt.tight_layout()
plt.show()