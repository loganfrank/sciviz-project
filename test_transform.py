import numpy as np
import torch
from transforms import MaskAndVoronoi2D

if __name__ == '__main__':
    
    np.random.seed(1)
    
    n_channels = 1
    device = torch.device('cuda')
    
    data = torch.rand((2, 96, 96), dtype=torch.float32)
    t = MaskAndVoronoi2D(200, 10, device=device)
    t.generate_masks(data=data)
    res = t(data)
    
    print('stop')