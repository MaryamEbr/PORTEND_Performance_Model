import copy
import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import collections
from itertools import repeat
import torch.optim as optim
import numpy as np
import time

import torch
from torch import nn, Tensor
import torchvision
from torchvision.ops import StochasticDepth

### some used functions and classes
def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:

    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))

class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input

class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )

class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )

class MBConvConfig():
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels_in: int,
        out_channels_in: int,
        num_layers_in: int,
        width_mult: float,
        depth_mult: float):



        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.width_mult = width_mult
        self.depth_mult = depth_mult


        self.input_channels = self.adjust_channels(input_channels_in, width_mult)
        self.out_channels = self.adjust_channels(out_channels_in, width_mult)
        self.num_layers = self.adjust_depth(num_layers_in, depth_mult)

        # print("in", self.input_channels, "out", self.out_channels, "num_lay", self.num_layers)
    
    def adjust_depth(self, num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))

    
    def adjust_channels(self, channels: int, width_mult: float) -> int:
        ### make divisible
        v = channels * width_mult
        divisor = 8
        min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


#### model for partial inference. only one partition is returned. only the last exit branch in partition is considered.
### returns a list of output vector and intermediate data
class EfficientNet_partial(nn.Module):
    def __init__(self, dropout, width_mult, depth_mult, norm_layer, num_classes, partition):
        super().__init__()

        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1, width_mult, depth_mult),
            MBConvConfig(6, 3, 2, 16, 24, 2, width_mult, depth_mult),
            MBConvConfig(6, 5, 2, 24, 40, 2, width_mult, depth_mult),
            MBConvConfig(6, 3, 2, 40, 80, 3, width_mult, depth_mult),
            MBConvConfig(6, 5, 1, 80, 112, 3, width_mult, depth_mult),
            MBConvConfig(6, 5, 2, 112, 192, 4, width_mult, depth_mult),
            MBConvConfig(6, 3, 1, 192, 320, 1, width_mult, depth_mult),
        ]

        stochastic_depth_prob = 0.2
        self.partition_begin = partition[0]
        self.partition_end = partition[1]
        self.inverted_residual_setting = inverted_residual_setting
        self.layers_dict = nn.ModuleDict({})
        

        # building first layer
        if self.partition_begin <= 1 <= self.partition_end:
            firstconv_output_channels = inverted_residual_setting[0].input_channels
            self.layers_dict['block_br1'] = Conv2dNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU)
        
    
        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for ind, cnf in enumerate(inverted_residual_setting):
            if self.partition_begin <= ind+2 <= self.partition_end:
                
                # print("in init block ", ind+2)
                stage: List[nn.Module] = []
                for _ in range(cnf.num_layers):
                    # copy to avoid modifications. shallow copy is enough
                    block_cnf = copy.copy(cnf)

                    # overwrite info if not the first conv in the stage
                    if stage:
                        block_cnf.input_channels = block_cnf.out_channels
                        block_cnf.stride = 1

                    # adjust stochastic depth probability based on the depth of the stage block
                    sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                    stage.append(MBConv(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1


                if ind == len(inverted_residual_setting)-1:
                    lastconv_input_channels = inverted_residual_setting[-1].out_channels
                    lastconv_output_channels = 4 * lastconv_input_channels
                    
                    self.layers_dict[f'block_br{ind+2}'] = nn.Sequential(*stage, Conv2dNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.SiLU))
            
                else:
                    self.layers_dict[f'block_br{ind+2}'] = nn.Sequential(*stage)


        
        # building exit branches
        # first exit branch
        if 1 == self.partition_end:
            # print("in init exit ", 1)
            self.layers_dict[f'exit_br1'] = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                        nn.Flatten(1),
                        nn.Dropout(p=dropout, inplace=True),
                        nn.Linear(inverted_residual_setting[0].input_channels, num_classes))
        
        # second to last exit braches
        for ind, cnf in enumerate(inverted_residual_setting):
            if ind+2 == self.partition_end:
                # print("in init exit ", ind+2)
                if ind == len(inverted_residual_setting)-1:
                    linear_out_channels = 4 * cnf.out_channels
                else:
                    linear_out_channels = cnf.out_channels
                
                self.layers_dict[f'exit_br{ind+2}'] = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                        nn.Flatten(1),
                        nn.Dropout(p=dropout, inplace=True),
                        nn.Linear(linear_out_channels, num_classes))
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:

        for ind in range(len(self.inverted_residual_setting)+1):
            if self.partition_begin <= ind+1 <= self.partition_end:
                # print("in forward block ", ind+1)
                x = self.layers_dict[f'block_br{ind+1}'](x)

            if ind+1 == self.partition_end:
                # print("in forward exit ", ind+1)
                out_vector = self.layers_dict[f'exit_br{ind+1}'](x)
            
        return [out_vector, x]


#### model for final training and traversing. with just a selected set of exit branches
#### call with all possible exits and use the intermediate data list as a traversal function
class EfficientNet_final(nn.Module):
    def __init__(self, dropout, width_mult, depth_mult, norm_layer, num_classes, selected_exits):
        super().__init__()

        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1, width_mult, depth_mult),
            MBConvConfig(6, 3, 2, 16, 24, 2, width_mult, depth_mult),
            MBConvConfig(6, 5, 2, 24, 40, 2, width_mult, depth_mult),
            MBConvConfig(6, 3, 2, 40, 80, 3, width_mult, depth_mult),
            MBConvConfig(6, 5, 1, 80, 112, 3, width_mult, depth_mult),
            MBConvConfig(6, 5, 2, 112, 192, 4, width_mult, depth_mult),
            MBConvConfig(6, 3, 1, 192, 320, 1, width_mult, depth_mult),
        ]
        stochastic_depth_prob = 0.2

        self.selected_exits = selected_exits
        self.inverted_residual_setting = inverted_residual_setting
        self.layers_dict = nn.ModuleDict({})
 

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.layers_dict['block_br1'] = Conv2dNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU)
        
    
        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for ind, cnf in enumerate(inverted_residual_setting):
            if ind+1 < self.selected_exits[-1]:
                # print("in init block ", ind+2)
                stage: List[nn.Module] = []
                for _ in range(cnf.num_layers):
                    # copy to avoid modifications. shallow copy is enough
                    block_cnf = copy.copy(cnf)

                    # overwrite info if not the first conv in the stage
                    if stage:
                        block_cnf.input_channels = block_cnf.out_channels
                        block_cnf.stride = 1

                    # adjust stochastic depth probability based on the depth of the stage block
                    sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                    stage.append(MBConv(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1


                if ind == len(inverted_residual_setting)-1:
                    lastconv_input_channels = inverted_residual_setting[-1].out_channels
                    lastconv_output_channels = 4 * lastconv_input_channels
                    
                    self.layers_dict[f'block_br{ind+2}'] = nn.Sequential(*stage, Conv2dNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.SiLU))
            
                else:
                    self.layers_dict[f'block_br{ind+2}'] = nn.Sequential(*stage)


        
        # building exit branches
        # first exit branch
        if 1 in self.selected_exits:
            # print("in init exit ", 1)
            self.layers_dict[f'exit_br1'] = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                        nn.Flatten(1),
                        nn.Dropout(p=dropout, inplace=True),
                        nn.Linear(inverted_residual_setting[0].input_channels, num_classes))
        
        # second to last exit braches
        for ind, cnf in enumerate(inverted_residual_setting):
            if ind+2 in self.selected_exits:
                # print("in init exit ", ind+2)
                if ind == len(inverted_residual_setting)-1:
                    linear_out_channels = 4 * cnf.out_channels
                else:
                    linear_out_channels = cnf.out_channels
                
                self.layers_dict[f'exit_br{ind+2}'] = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                        nn.Flatten(1),
                        nn.Dropout(p=dropout, inplace=True),
                        nn.Linear(linear_out_channels, num_classes))
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:

        out_vector_list = []
        intermediate_list = []
        intermediate_list.append(x)

        for ind in range(len(self.inverted_residual_setting)+1):
            if ind < self.selected_exits[-1]:
                # print("in forward block ", ind+1)
                x = self.layers_dict[f'block_br{ind+1}'](x)
                intermediate_list.append(x)
            if ind+1 in self.selected_exits:
                # print("in forward exit ", ind+1)
                out_vector_list.append(self.layers_dict[f'exit_br{ind+1}'](x))
            
        return [out_vector_list, intermediate_list]


