import numpy as np
import torch
from transforms import MaskAndVoronoi3D

if __name__ == '__main__':
    
    np.random.seed(1)
    
    n_channels = 1
    
    data = torch.rand((n_channels, 256, 256, 256), dtype=torch.float32)
    t = MaskAndVoronoi3D(20)
    res = t(data)
    
    print('stop')