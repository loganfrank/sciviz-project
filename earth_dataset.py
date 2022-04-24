import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets

import numpy as np

class EarthMantleDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, nchannels):
        self.root = root
        self.transform = transform
        self.nchannels = nchannels
        self.instances = [f for f in os.listdir(self.root) if 'raw' in f]
        
    def __len__(self):
        return len(self.instances)
        
    def __getitem__(self, index):
        path = f'{self.root}{self.instances[index]}'
        data = np.fromfile(path, dtype=np.float32)
        data = data.reshape(256, 256, 256, 4).transpose(3, 0, 1, 2)
        
        if self.nchannels == 1:
            data = data[:1, :, :, :]
        else:
            data = data[1:, :, :, :]
        
        # Convert to PyTorch tensor
        data = torch.from_numpy(data)
        
        # Clone for ground truth then create the mask and Voronoi inputs
        ground_truth = data.clone()
        data = self.transform(data)
        return data, ground_truth