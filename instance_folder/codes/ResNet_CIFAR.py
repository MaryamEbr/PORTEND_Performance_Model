import torch
from torch import Tensor 
import torch.nn as nn

#### basic resnet blocks for following models
class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, down=False):
        super().__init__()
            
        self.conv1 = nn.Conv2d(inplanes, planes, stride=stride, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = None
        
        if down:
            self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        

    def forward(self, x: Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out 


#### model for partial inference. only one partition is returned. only the last exit branch in partition is considered.
### returns a list of output vector and intermediate data
class ResNet_CIFAR_model_partial(nn.Module):
    def __init__(self, model_n, branch_number, num_class, device, partition):
        super().__init__()

        self.num_classes = num_class
        self.model_n = model_n
        self.branch_number = branch_number
        self.layers_dict = nn.ModuleDict({})
        self.partition_begin = partition[0]
        self.partition_end = partition[1]

        ### begining layers
        if self.partition_begin <= 1 <= self.partition_end:
            self.layers_dict['conv1_br1'] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding='same')
            self.layers_dict['bn1_br1'] = nn.BatchNorm2d(16)
            self.layers_dict['relu1_br1'] = nn.ReLU(inplace=True)
        
        
        ### ResNet blocks [16, 32, 64] - CIFAR architecture
        # first block, 16 channels
        for i in range(self.model_n):
            
            if self.partition_begin <= i+2 <= self.partition_end:
                self.layers_dict['block_br'+str(i+2)] = BasicBlock(16, 16).to(device)
            
        
        # second block, 32 channels
        for i in range(self.model_n):
            
            if i == 0 and self.partition_begin <= i+2+self.model_n <= self.partition_end:
                self.layers_dict['block_br'+str(i+2+self.model_n)] = BasicBlock(16, 32, stride=2, down=True).to(device)
                
            elif self.partition_begin <= i+2+self.model_n <= self.partition_end:
                self.layers_dict['block_br'+str(i+2+self.model_n)] = BasicBlock(32, 32).to(device)
                
                
        # third block, 64 channels
        for i in range(self.model_n):
            if i == 0 and self.partition_begin <= i+2+(2*self.model_n) <= self.partition_end:
                self.layers_dict['block_br'+str(i+2+(2*self.model_n))] = BasicBlock(32, 64, stride=2, down=True).to(device)
            elif self.partition_begin <= i+2+(2*self.model_n) <= self.partition_end:
                self.layers_dict['block_br'+str(i+2+(2*self.model_n))] = BasicBlock(64, 64).to(device)
        
    
        ### output layers
        for i in range(self.branch_number):
            
            if i+1 != self.partition_end:
                continue
            
            if (i <= self.model_n):
                insize = 16
            if (self.model_n < i <= 2*self.model_n):
                insize = 32
            if (i > 2*self.model_n):
                insize = 64
                
                
            self.layers_dict['pool_br'+str(i+1)] = nn.AdaptiveAvgPool2d((1, 1))
            self.layers_dict['flatten_br'+str(i+1)] = nn.Flatten(1)
            self.layers_dict['dense_br'+str(i+1)] = nn.Linear(insize, self.num_classes)


    def forward(self, x: Tensor) -> Tensor:

        ### begining layers
        if self.partition_begin <= 1 <= self.partition_end:
            x = self.layers_dict['conv1_br1'](x)
            x = self.layers_dict['bn1_br1'](x)
            x = self.layers_dict['relu1_br1'](x)
        
        ## exit branch 1
        if 1 == self.partition_end:
            temp = self.layers_dict['pool_br1'](x)
            temp = self.layers_dict['flatten_br1'](temp)
            out_vector = self.layers_dict['dense_br1'](temp)
        
        ### ResNet blocks
        for i in range(self.branch_number-1):
            
            if self.partition_begin <= i+2 <= self.partition_end:
                x = self.layers_dict['block_br'+str(i+2)](x)
            
            ## exit branch after each resnet block
            if i+2 == self.partition_end:
                temp = self.layers_dict['pool_br'+str(i+2)](x)
                temp = self.layers_dict['flatten_br'+str(i+2)](temp)
                out_vector = self.layers_dict['dense_br'+str(i+2)](temp)
            

        return [out_vector, x]


#### model for final training and traversing. with just a selected set of exit branches
#### call with all possible exits and use the intermediate data list as a traversal function
class ResNet_CIFAR_model_final(nn.Module):
    def __init__(self, model_n, branch_number, num_class, selected_exits, device):
        super().__init__()

        self.num_classes = num_class
        self.model_n = model_n
        self.branch_number = branch_number
        self.selected_exits = selected_exits-1
        self.layers_dict = nn.ModuleDict({})

        ### begining layers, always there
        self.layers_dict['conv1_br1'] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding='same')
        self.layers_dict['bn1_br1'] = nn.BatchNorm2d(16)
        self.layers_dict['relu1_br1'] = nn.ReLU(inplace=True)
        
        
        ### ResNet blocks [16, 32, 64] - CIFAR architecture
        # first block, 16 channels
        for i in range(self.model_n):
            if i+1 <= self.selected_exits[-1]:
                self.layers_dict['block_br'+str(i+2)] = BasicBlock(16, 16).to(device)
            
        
        # second block, 32 channels
        for i in range(self.model_n):
            if i+1+self.model_n <=  self.selected_exits[-1]:
                if i == 0:
                    self.layers_dict['block_br'+str(i+2+self.model_n)] = BasicBlock(16, 32, stride=2, down=True).to(device)
                else:
                    self.layers_dict['block_br'+str(i+2+self.model_n)] = BasicBlock(32, 32).to(device)
                
                
        # third block, 64 channels
        for i in range(self.model_n):
            if i+1+(2*self.model_n) <=  self.selected_exits[-1]:
                if i == 0:
                    self.layers_dict['block_br'+str(i+2+(2*self.model_n))] = BasicBlock(32, 64, stride=2, down=True).to(device)
                else:
                    self.layers_dict['block_br'+str(i+2+(2*self.model_n))] = BasicBlock(64, 64).to(device)
        
    
        ### output layers
        for i in range(self.branch_number):
            if i not in self.selected_exits:
                continue
            
            if (i <= self.model_n):
                insize = 16
            if (self.model_n < i <= 2*self.model_n):
                insize = 32
            if (i > 2*self.model_n):
                insize = 64
                
            self.layers_dict['pool_br'+str(i+1)] = nn.AdaptiveAvgPool2d((1, 1))
            self.layers_dict['flatten_br'+str(i+1)] = nn.Flatten(1)
            self.layers_dict['dense_br'+str(i+1)] = nn.Linear(insize, self.num_classes)


    def forward(self, x: Tensor) -> Tensor:
        out_vector_list = []
        intermediate_list = []
        
        

        ### begining layers
        # first block will always be a part of model
        intermediate_list.append(x)
        x = self.layers_dict['conv1_br1'](x)
        x =  self.layers_dict['bn1_br1'](x)
        x = self.layers_dict['relu1_br1'](x)
        intermediate_list.append(x)
        
        
        ## exit branch 1
        if 0 in self.selected_exits:
            temp = self.layers_dict['pool_br1'](x)
            temp = self.layers_dict['flatten_br1'](temp)
            out_vector_list.append(self.layers_dict['dense_br1'](temp))
        
        
        
        ### ResNet blocks
        for i in range(self.branch_number-1):
            # for partial models
            if i+1 <= self.selected_exits[-1]:
                x = self.layers_dict['block_br'+str(i+2)](x)
                intermediate_list.append(x)
            
            if i+1 in self.selected_exits:
                ## exit branch after each resnet block
                temp = self.layers_dict['pool_br'+str(i+2)](x)
                temp = self.layers_dict['flatten_br'+str(i+2)](temp)
                out_vector_list.append(self.layers_dict['dense_br'+str(i+2)](temp))
            

        return [out_vector_list, intermediate_list]