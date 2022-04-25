import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets

import numpy as np

class VectorFieldSim2D(torch.utils.data.Dataset):
    def __init__(self, u_equation, v_equation, noise_equation, num_examples, size, transform):
        # All points in the example
        x = torch.linspace(0, size - 1, steps=size)
        y = torch.linspace(0, size - 1, steps=size)
        points = torch.stack(torch.meshgrid(y, x))

        u = u_equation(points[-2], points[-1])
        v = v_equation(points[-2], points[-1])
        self.instances = torch.stack([u, v])
        self.instances = torch.repeat_interleave(self.instances[None, :, :, :], num_examples, dim=0)
        self.instances = self.instances + noise_equation((torch.rand_like(self.instances[:, 0, :, :]).unsqueeze(1) * 10) - 5)

        self.transform = transform
        
    def __len__(self):
        return len(self.instances)
        
    def __getitem__(self, index):
        # Retrieve the instance
        data = self.instances[index]

        # Clone for ground truth then create the mask and Voronoi inputs
        ground_truth = data.clone()
        data = self.transform(data)
        return data, ground_truth