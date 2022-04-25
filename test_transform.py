import numpy as np
import torch
from transforms import MaskAndVoronoi3D

if __name__ == '__main__':
    
    np.random.seed(1)
    
    n_channels = 1
    device = torch.device('cuda')
    
    data = torch.rand((n_channels, 128, 128, 128), dtype=torch.float32)
    t = MaskAndVoronoi3D(150, device=device)
    res = t(data)
    
    print('stop')