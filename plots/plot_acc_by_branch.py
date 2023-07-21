#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 12

import pickle
### for accuracy by branch
# resnet20
#  [list([[1, 9], [10, 10]])]
# resnet110
#  [list([[1, 25], [26, 40]])] 
# effb0
#  [list([[1, 4], [5, 8]])]


model_name = 'EfficientNetB0'
dataset = 'ImageNet'
partitioning = [[1, 4], [5, 8]]
acc_list = []
exits_corrects_partitionwise_dict = pickle.load(open(f"all_profiling_results/accuracy_exitrate/profiling_exitrate_accuracy_partitionwise_model[{model_name}]_dataset[{dataset}]_test.txt", 'rb'))
for i, partition in enumerate(partitioning):
    # print("  threshold1  ", exitrate_accuracy_dict[str([1, partition[-1]])][1], f"    threhold{threshold}  ", exitrate_accuracy_dict[str([1, partition[-1]])][threshold])
    er_acc_list = exits_corrects_partitionwise_dict[str([1, partition[-1]])][1]
    branch_acc = er_acc_list[1]/(er_acc_list[0]+0.000000001)
    acc_list.append(branch_acc)
print(acc_list)





r=1.3
fig, ax = plt.subplots(figsize=(r*2.5, r*2.05))   
# create the list of two numbers
data = acc_list

# create a list of labels for the x-axis
labels = ['Branch 4', 'Branch 8']

# create evenly spaced x-coordinates for the bars
x = np.arange(len(labels))
w = 0.8
# create a bar plot with two bars and adjust their positions and width
plt.bar(x , data, width=w, color='darkblue')
# add labels and title
plt.xlabel('Exit Branches')
plt.ylabel('Accuracy')
# plt.title(f'{model_name}, {dataset}', fontsize=12)
plt.xticks(x , labels)
plt.ylim(0, 1.01)

# add the numbers above the bars
for i, v in enumerate(data):
    plt.text(i, v+0.01, f"{round(v*100,2)}%", color='black', ha='center', fontsize=12)


# plt.yticks([0.2, 0.4, 0.6, 0.8, 1])
# show the plot
plt.tight_layout()
plt.show()
