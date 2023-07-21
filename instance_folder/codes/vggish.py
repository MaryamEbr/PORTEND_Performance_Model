import numpy as np
import torch
import torch.nn as nn
from torch import hub
from torch.utils.data import Dataset, DataLoader
import csv
import os
import soundfile as sf

import vggish_input, vggish_params


class Postprocessor(nn.Module):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """

    def __init__(self):
        """Constructs a postprocessor."""
        super(Postprocessor, self).__init__()
        # Create empty matrix, for user's state_dict to load
        self.pca_eigen_vectors = torch.empty((vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE,),dtype=torch.float,)
        self.pca_means = torch.empty((vggish_params.EMBEDDING_SIZE, 1), dtype=torch.float)

        self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)
        
        state_dict = torch.load('ins_folder/weights/vggish_pca_params-970ea276.pth')
        state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME] = torch.as_tensor(state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME], dtype=torch.float)
        state_dict[vggish_params.PCA_MEANS_NAME] = torch.as_tensor(state_dict[vggish_params.PCA_MEANS_NAME].reshape(-1, 1), dtype=torch.float)

        self.load_state_dict(state_dict)

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (embeddings_batch.shape,)
        assert (embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE), "Bad batch shape: %r" % (embeddings_batch.shape,)

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)).t()
        
        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL)
        
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round((clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL)* (255.0/ (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)))


        return quantized_embeddings

    def forward(self, x):
        x = self.postprocess(x)
        return x
    


#### model for partial inference. only one partition is returned. only the last exit branch in partition is considered.
### returns a list of output vector and intermediate data
class VGGish_partial(nn.Module):
    def __init__(self, partition, device):
        super().__init__()
        self.device = device
        self.partition_begin = partition[0]
        self.partition_end = partition[1]

        self.relu = nn.ReLU(inplace=True)

        if self.partition_begin <= 1 <= self.partition_end:
            self.conv1_block_br1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.pool1_block_br1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if self.partition_begin <= 2 <= self.partition_end:
            self.conv2_block_br2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool2_block_br2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if self.partition_begin <= 3 <= self.partition_end:
            self.conv3_block_br3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.conv4_block_br3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.pool4_block_br3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if self.partition_begin <= 4 <= self.partition_end:
            self.conv5_block_br4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.conv6_block_br4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.pool6_block_br4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        if 1 == self.partition_end:
            self.embeddings_exit_br1 = nn.Sequential(
                nn.Linear(64*48*32, 128),
                nn.ReLU(True),
                Postprocessor(),
                torch.nn.Linear(128, 527))
        
        if 2 == self.partition_end:
            self.embeddings_exit_br2 = nn.Sequential(
                nn.Linear(128*24*16, 128),
                nn.ReLU(True),
                Postprocessor(),
                torch.nn.Linear(128, 527))
        
        if 3 == self.partition_end:
            self.embeddings_exit_br3 = nn.Sequential(
                nn.Linear(256*12*8, 128),
                nn.ReLU(True),
                Postprocessor(),
                torch.nn.Linear(128, 527))
        
        if 4 == self.partition_end:
            self.embeddings_exit_br4 = nn.Sequential(
                nn.Linear(512*4*6, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 128),
                nn.ReLU(True), 
                Postprocessor(),
                torch.nn.Linear(128, 527))
            


    def forward(self, x, pre=True):
        
        ### block1
        if self.partition_begin <= 1 <= self.partition_end:
            ################# preprocess
            if pre:
                x = self._preprocess(x.numpy(), 44100)
                x = x.to(self.device)
                
                
            x = self.conv1_block_br1(x)
            x = self.relu(x)
            x = self.pool1_block_br1(x)

        x = x.to(self.device)
        # exit1
        if 1 == self.partition_end:
            temp = torch.transpose(x, 1, 3)
            temp = torch.transpose(temp, 1, 2)
            temp = temp.contiguous().view(temp.size(0), -1)
            out_vector = self.embeddings_exit_br1(temp)
        
        
        ### block2
        if self.partition_begin <= 2 <= self.partition_end:
            x = self.conv2_block_br2(x) 
            x = self.relu(x)
            x = self.pool2_block_br2(x) 
            
        # exit2
        if 2 == self.partition_end:
            temp = torch.transpose(x, 1, 3)
            temp = torch.transpose(temp, 1, 2)
            temp = temp.contiguous().view(temp.size(0), -1)
            out_vector = self.embeddings_exit_br2(temp)
        
        ### block3
        if self.partition_begin <= 3 <= self.partition_end:
            x = self.conv3_block_br3(x)
            x = self.relu(x)
            x = self.conv4_block_br3(x)
            x = self.relu(x)
            x = self.pool4_block_br3(x)
            
        # exit3
        if 3 == self.partition_end:
            temp = torch.transpose(x, 1, 3)
            temp = torch.transpose(temp, 1, 2)
            temp = temp.contiguous().view(temp.size(0), -1)
            out_vector = self.embeddings_exit_br3(temp)
        
        ### block4
        if self.partition_begin <= 4 <= self.partition_end:
            x = self.conv5_block_br4(x)
            x = self.relu(x)
            x = self.conv6_block_br4(x)
            x = self.relu(x)
            x = self.pool6_block_br4(x)
            
            
        # exit4
        if 4 == self.partition_end:
            temp = torch.transpose(x, 1, 3)
            temp = torch.transpose(temp, 1, 2)
            temp = temp.contiguous().view(temp.size(0), -1)
            out_vector = self.embeddings_exit_br4(temp)
        
        return [out_vector, x]


    def _preprocess (self, wav_data, sr):
        wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        mel_data = vggish_input.waveform_to_examples(wav_data, sr)
        return mel_data




#### model for final training and traversing. with just a selected set of exit branches
#### call with all possible exits and use the intermediate data list as a traversal function
class VGGish_final(nn.Module):
    def __init__(self, selected_exits, device):
        super().__init__()

        self.device = device
        self.selected_exits = selected_exits

        self.relu = nn.ReLU(inplace=True)

        if self.selected_exits[-1] >=1:
            self.conv1_block_br1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.pool1_block_br1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if self.selected_exits[-1] >=2:
            self.conv2_block_br2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool2_block_br2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if self.selected_exits[-1] >=3:
            self.conv3_block_br3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.conv4_block_br3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.pool4_block_br3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if self.selected_exits[-1] >=4:
            self.conv5_block_br4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.conv6_block_br4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.pool6_block_br4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        if 1 in self.selected_exits:
            self.embeddings_exit_br1 = nn.Sequential(
                nn.Linear(64*48*32, 128),
                nn.ReLU(True),
                Postprocessor(),
                torch.nn.Linear(128, 527))
        
        if 2 in self.selected_exits:
            self.embeddings_exit_br2 = nn.Sequential(
                nn.Linear(128*24*16, 128),
                nn.ReLU(True),
                Postprocessor(),
                torch.nn.Linear(128, 527))
        
        if 3 in self.selected_exits:
            self.embeddings_exit_br3 = nn.Sequential(
                nn.Linear(256*12*8, 128),
                nn.ReLU(True),
                Postprocessor(),
                torch.nn.Linear(128, 527))
        
        if 4 in self.selected_exits:
            self.embeddings_exit_br4 = nn.Sequential(
                nn.Linear(512*4*6, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 128),
                nn.ReLU(True), 
                Postprocessor(),
                torch.nn.Linear(128, 527))
            


    def forward(self, x, pre=True):
        out_vector_list = []
        intermediate_list = []
        
        ### block1
        if self.selected_exits[-1] >=1:
            intermediate_list.append(x)
            ################# preprocess
            if pre:
                x = self._preprocess(x.numpy(), 44100)
                x = x.to(self.device)
            
            x = self.conv1_block_br1(x)
            x = self.relu(x)
            x = self.pool1_block_br1(x)
            intermediate_list.append(x)
            
        x = x.to(self.device)
        # exit1
        if 1 in self.selected_exits:
            temp = torch.transpose(x, 1, 3)
            temp = torch.transpose(temp, 1, 2)
            temp = temp.contiguous().view(temp.size(0), -1)
            out_vector_list.append(self.embeddings_exit_br1(temp))
        
        
        ### block2
        if self.selected_exits[-1] >=2:
            x = self.conv2_block_br2(x) 
            x = self.relu(x)
            x = self.pool2_block_br2(x) 
            intermediate_list.append(x) 
            
        # exit2
        if 2 in self.selected_exits:
            temp = torch.transpose(x, 1, 3)
            temp = torch.transpose(temp, 1, 2)
            temp = temp.contiguous().view(temp.size(0), -1)
            out_vector_list.append(self.embeddings_exit_br2(temp))
        
        ### block3
        if self.selected_exits[-1] >=3:
            x = self.conv3_block_br3(x)
            x = self.relu(x)
            x = self.conv4_block_br3(x)
            x = self.relu(x)
            x = self.pool4_block_br3(x)
            intermediate_list.append(x) 
            
        # exit3
        if 3 in self.selected_exits:
            temp = torch.transpose(x, 1, 3)
            temp = torch.transpose(temp, 1, 2)
            temp = temp.contiguous().view(temp.size(0), -1)
            out_vector_list.append(self.embeddings_exit_br3(temp))
        
        ### block4
        if self.selected_exits[-1] >=4:
            x = self.conv5_block_br4(x)
            x = self.relu(x)
            x = self.conv6_block_br4(x)
            x = self.relu(x)
            x = self.pool6_block_br4(x)
            intermediate_list.append(x) 
            
            
        # exit4
        if 4 in self.selected_exits:
            temp = torch.transpose(x, 1, 3)
            temp = torch.transpose(temp, 1, 2)
            temp = temp.contiguous().view(temp.size(0), -1)
            out_vector_list.append(self.embeddings_exit_br4(temp))
        
        return [out_vector_list, intermediate_list]
    
    
    def _preprocess (self, wav_data, sr):
        wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        mel_data = vggish_input.waveform_to_examples(wav_data, sr)
        return mel_data





#### dataset class to handle audioset 
class AudioSetDataSet(Dataset):
    def __init__(self, path):

        self.path = path
        self.label_displayname_dict = {}
        for i, key in enumerate(csv.DictReader(open(os.path.join(f'{self.path}class_labels_indices.csv')))):
            self.label_displayname_dict[key['mid']] = key['display_name']

        self.sample_label_dict_valid = {}
        for key in csv.DictReader(open(os.path.join(f'{self.path}labels.csv'))):
            self.sample_label_dict_valid[key['YTID']] = key['positive_labels']

        self.wav_names = [a for a in os.listdir(path) if a.endswith('.wav')]
    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        file = self.wav_names[idx]
        wav_data, _ = sf.read(os.path.join(self.path, file), dtype='int16') #sr is always 44100

        # preprocessing labels to onehot version
        wav_labels = self.sample_label_dict_valid[file.split('.wav')[0]].split(',')
        label_data = torch.Tensor([list(self.label_displayname_dict.keys()).index(a) for a in  wav_labels])
        return wav_data, label_data