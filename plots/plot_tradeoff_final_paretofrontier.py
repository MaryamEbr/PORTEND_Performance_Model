#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 12



experiment_folder = 'exp_folder/'
destination_tier = 0

# # resnet20
# acc_const= 0.82
# lat_const = 100
# source_wait = 270

# resnet110
# acc_const= 0.75
# lat_const = 175
# source_wait = 270

# effnetb0
acc_const= 0.66
lat_const = 175000
source_wait = 707


# # vggish
# acc_const= 0.7
# lat_const = 3300
# source_wait = 3300000


#### loading the dataframes for  experiments
dnn_df = pd.read_pickle(f'{experiment_folder}Experiment73/total_df.pkl')

dnn_df = dnn_df.loc[(dnn_df['e2e_tail_mean']<lat_const) & (dnn_df['accuracy']>acc_const) & (dnn_df['model_mode']=='full' ) & (dnn_df['source_wait']!=source_wait) & (dnn_df['destination_tier']==destination_tier)].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])

#### to only show the points closer to pareto frontier

# # resnet20
# dnn_df = dnn_df.drop(dnn_df[(dnn_df['e2e_tail_mean']>20) & (dnn_df['accuracy']<0.882)].index)
# dnn_df = dnn_df.drop(dnn_df[(dnn_df['e2e_tail_mean']>18) & (dnn_df['accuracy']<0.865)].index)

# # resnet110
# dnn_df = dnn_df.drop(dnn_df[(dnn_df['e2e_tail_mean']>75) & (dnn_df['accuracy']<=0.89)].index)
# dnn_df = dnn_df.drop(dnn_df[(dnn_df['e2e_tail_mean']>55) & (dnn_df['accuracy']<=0.88)].index)
# dnn_df = dnn_df.drop(dnn_df[(dnn_df['e2e_tail_mean']>50) & (dnn_df['accuracy']<0.87)].index)
# dnn_df = dnn_df.drop(dnn_df[(dnn_df['e2e_tail_mean']>35) & (dnn_df['accuracy']<0.82)].index)

# effnetb0
dnn_df = dnn_df.drop(dnn_df[(dnn_df['e2e_tail_mean']>420) & (dnn_df['accuracy']<0.72)].index)
dnn_df = dnn_df.drop(dnn_df[(dnn_df['e2e_tail_mean']>475) & (dnn_df['accuracy']<0.727)].index)
dnn_df = dnn_df.drop(dnn_df[(dnn_df['e2e_tail_mean']>402) & (dnn_df['accuracy']<0.6781)].index)

# # vggish
# # dnn_df = dnn_df.drop(dnn_df[(dnn_df['e2e_tail_mean']>420) & (dnn_df['accuracy']<0.72)].index)
# dnn_df = dnn_df.drop(dnn_df[(dnn_df['e2e_tail_mean']>2000) & (dnn_df['accuracy']<0.716)].index)



# our method df
dnn_df_our_final_1 = dnn_df.loc[(dnn_df['num_partitions']==1)].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
dnn_df_our_final_2 = dnn_df.loc[(dnn_df['num_partitions']==2)].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])
# dnn_df_our_final_3 = dnn_df.loc[(dnn_df['num_partitions']==3)].sort_values(by=['e2e_tail_mean', 'accuracy'], ascending=[True, False])



with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(dnn_df_our_final_1[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy', 'exit_rate_all']])
    print("*************")
    print(dnn_df_our_final_2[['partitioning', 'placement', 'threshold', 'e2e_tail_mean', 'accuracy', 'exit_rate_all']])



our_1p_latency = dnn_df_our_final_1['e2e_tail_mean'].values
our_1p_accuracy = dnn_df_our_final_1['accuracy'].values

our_2p_latency = dnn_df_our_final_2['e2e_tail_mean'].values
our_2p_accuracy = dnn_df_our_final_2['accuracy'].values

# our_3p_latency = dnn_df_our_final_3['e2e_tail_mean'].values
# our_3p_accuracy = dnn_df_our_final_3['accuracy'].values




# Set up the plot
r=1.3
fig, ax = plt.subplots(figsize=(r*2.5, r*2))  


ax.set_axisbelow(True)
plt.grid(True, color='lightgray') 

plt.scatter(our_1p_latency, our_1p_accuracy, marker='o', s=60, color='olive', label='1 partition')
plt.scatter(our_2p_latency, our_2p_accuracy,  marker='^', s=60, color='violet', label='2 partitions')



# # add an arrow to the point , RESNET20
# xx = 16.83
# yy = 0.8812
# # plt.annotate('', xy=(xx-0.5, yy+0.00001), xytext=(xx-8.5, yy+0.005),
# #              arrowprops=dict('fancy', facecolor='orange', edgecolor='black', linewidth=1, mutation_scale=0.8))

# ax.annotate('',
#             xy=(xx-0.5, yy+0.00001),
#             xytext=(xx-15 + 0.5, yy+0.008),
#             arrowprops=dict(facecolor='orange', arrowstyle='fancy', mutation_scale=33))
# plt.xlim(5, 95)
# plt.ylim(0.825, 0.895)


# # add an arrow to the point , RESNET110
# xx = 58.59
# yy = 0.8878
# ax.annotate('',
#             xy=(xx-0.5, yy+0.0005),
#             xytext=(xx-25, yy+0.02),
#             arrowprops=dict(facecolor='orange', arrowstyle='fancy', mutation_scale=33))

# plt.xlim(5, 170)
# plt.ylim(0.76, 0.91)




# add an arrow to the point , EFF
xx = 421.30
yy = 0.7218

ax.annotate('',
            xy=(xx-0.2, yy+0.0001),
            xytext=(xx-45, yy+0.0075),
            arrowprops=dict(facecolor='orange', arrowstyle='fancy', mutation_scale=33))
plt.xlim(370, 637)
plt.ylim(0.665, 0.736)



# # add an arrow to the point ,vggish
# xx = 1844.43
# yy = 0.719
# plt.annotate('', xy=(xx-4, yy+0.00001), xytext=(xx-120, yy+0.005),
#              arrowprops=dict(facecolor='orange'))





# plt.title(f"ResNet20, CIFAR10", font size=12)
plt.legend(loc='lower right', handlelength=1, handletextpad=0.2)

plt.xlabel('End to End Latency (ms)')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
