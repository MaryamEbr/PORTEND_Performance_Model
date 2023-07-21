import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import soundfile as sf
import time
import os
import csv
from functools import partial
import torchvision.transforms as transforms

from ResNet_CIFAR import ResNet_CIFAR_model_partial
from EfficientNet import EfficientNet_partial
from vggish import VGGish_final, VGGish_partial, AudioSetDataSet


### get model and dataset info
dataset_info_dict = {'CIFAR10': {'num_class': 10, 'input_size': 32, 'num_channels': 3, 'validation_size': 5000, 'test_size': 5000},
                    'CIFAR100': {'num_class': 100, 'input_size': 32, 'num_channels': 3, 'validation_size': 5000, 'test_size': 5000},
                    'ImageNet': {'num_class': 1000, 'input_size_B0': 224, 'input_size_B7': 600 ,'crop_size_B0': 256, 'crop_size_B7': 600, 'num_channels': 3, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'validation_size': 5000, 'test_size': 5000},
                    'AudioSet': {'num_class': 527, 'input_size': [96, 64], 'num_channels': 1},
                    }
    
model_info_dict = {'ResNet20': {'model_n': 3, 'branch_number': 10},
                    'ResNet110': {'model_n': 18, 'branch_number': 55},
                    'EfficientNetB0': {'branch_number': 8, 'dropout': 0.2, 'width_mult': 1.0, 'depth_mult': 1.0, 'norm_layer': nn.BatchNorm2d},
                    'EfficientNetB7': {'branch_number': 8, 'dropout': 0.5, 'width_mult': 2.0, 'depth_mult': 3.1, 'norm_layer': partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)},
                    'VGGish': {'branch_number': 4}, 
                    }


torch.manual_seed(43)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

### load test or validation set from the dataset (based on the path)
def load_dataset(dataset, model_name, path, batch_size=32):
    torch.manual_seed(43)

    if 'CIFAR' in dataset:

        transform_ds = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(dataset_info_dict[dataset]['input_size'])
        ])

        if dataset == 'CIFAR10':
            the_set = torchvision.datasets.CIFAR10(root=path, train=False, download=False, transform=transform_ds)
        if dataset == 'CIFAR100':
            the_set = torchvision.datasets.CIFAR100(root=path, train=False, download=False, transform=transform_ds)

        validation_set, test_set = torch.utils.data.random_split(the_set, [5000, 5000], generator=torch.Generator().manual_seed(43))
        
        if 'test' in path:
            the_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=False)
            the_size = len(test_set)
        if 'validation' in path:
            the_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=False)
            the_size = len(validation_set)


    if 'ImageNet' in dataset:
        
        transform_ds = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize(dataset_info_dict['ImageNet'][f'crop_size_{model_name[-2:]}']),
            transforms.CenterCrop(dataset_info_dict['ImageNet'][f'input_size_{model_name[-2:]}']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        the_set = torchvision.datasets.ImageFolder(path, transform=transform_ds)

        the_loader = torch.utils.data.DataLoader(the_set, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=False)
        the_size = len(the_set)


    if dataset == 'AudioSet':

        the_set = AudioSetDataSet(path)
        the_loader = DataLoader(the_set, batch_size=1, shuffle=False)
        the_size = len(the_set)
        
    return the_loader, the_size



### load the partition of model from full configuration weights or final trained weights
def load_partition(model_name, dataset, partition, partition_filename, full_or_final, device):

    if 'ResNet' in model_name:
        model = ResNet_CIFAR_model_partial(model_info_dict[model_name]['model_n'], model_info_dict[model_name]['branch_number'], dataset_info_dict[dataset]['num_class'], device, partition).to(device)
        
        if full_or_final == 'full':
            file_name = f"ins_folder/weights/ResNet{6*model_info_dict[model_name]['model_n']+2}_{dataset}_full_notfrozen.pt"
        if full_or_final == 'final':
            partition_filename = str([a[-1] for a in partition_filename]).replace(" ", "")
            file_name = f"ins_folder/weights/ResNet{6*model_info_dict[model_name]['model_n']+2}_{dataset}_final{partition_filename}.pt"
            
            # temp patch to use frozen model instead of final model with no early exit (threhsold=0)
            if len(partition_filename)>1:
                print("SHOULD BE HERE.....", flush=True)
                file_name = f"ins_folder/weights/ResNet{6*model_info_dict[model_name]['model_n']+2}_{dataset}_full_frozen.pt"

        # load weight into model
        model.load_state_dict(torch.load(file_name, map_location=device), strict=False)
    
        # loss function
        loss_fn = nn.CrossEntropyLoss()

    if 'EfficientNet' in model_name:
        model = EfficientNet_partial(model_info_dict[model_name]['dropout'], model_info_dict[model_name]['width_mult'], model_info_dict[model_name]['depth_mult'], model_info_dict[model_name]['norm_layer'], dataset_info_dict[dataset]['num_class'], partition).to(device)
        if full_or_final == 'full' and 'B0' in model_name:
            file_name = f"ins_folder/weights/{model_name}_{dataset}_full_notfrozen.pt"
        if full_or_final == 'full' and 'B7' in model_name:
            file_name = f"ins_folder/weights/{model_name}_{dataset}_full_frozen.pt"
        if full_or_final == 'final':
            partition_filename = str([a[-1] for a in partition_filename]).replace(" ", "")
            file_name = f"ins_folder/weights/{model_name}_{dataset}_final{partition_filename}.pt"

        # load weight into model
        model.load_state_dict(torch.load(file_name, map_location=device), strict=False)
    
        # loss function
        loss_fn = nn.CrossEntropyLoss()

    if model_name == 'VGGish':
        # no final here please!, just full model
        model = VGGish_partial(partition, device=device)
        model.load_state_dict(torch.load(f"ins_folder/weights/VGGish_AudioSet_full_notfrozen.pt", map_location=device), strict=False)
        model.to(device)
        
        loss_fn = nn.BCEWithLogitsLoss()


    return [model, loss_fn]
    


### test the partition of the model
def test_model (model, dataset, loss_fn, inputs, labels, device):
    
    # turn off autograd engine
    with torch.no_grad():
        
        # put model in evaluation mode
        model.eval()

        
        # have to seperate audioset+vggish from others, because it's different
        # VGGish case
        if dataset == 'AudioSet':
            
            # inputs = inputs.to(device)
            labels = labels.to(device)

            # forward pass
            [output, x] = model(inputs)
            
            # calculate entropy
            sm = torch.nn.Softmax(dim=1)
            entropy = -1*torch.sum(torch.log(sm(output)) * sm(output) / torch.log(torch.from_numpy(np.array(dataset_info_dict[dataset]['num_class']))), dim=1)
            entropy = torch.min(entropy)
            
            # no loss calculation
            test_loss = 0
            
            # calculate accuracy
            pred = torch.argmax(nn.Sigmoid()(output), dim=1)
            test_correct =  int( any(x in pred for x in labels) == True)

            
        # other cases: ResNet and EfficientNet
        else:
                
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward pass
            [output, x] = model(inputs)    
                
            # calculate entropy
            sm = torch.nn.Softmax(dim=1)
            entropy = -1*torch.sum(torch.log(sm(output)) * sm(output) / torch.log(torch.from_numpy(np.array(dataset_info_dict[dataset]['num_class']))))

            # calculate loss
            loss = loss_fn(output, labels)
            test_loss = loss.data.item()

            # calculate accuracy
            _, preds = torch.max(output, 1)
            test_correct = torch.sum(preds == labels.data).item()

    return [test_loss, test_correct, x, entropy]



