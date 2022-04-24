import random
import torch
import numpy as np

class MaskAndVoronoi3D(object):
    def __init__(self, nsensors, device=torch.device('cpu')):
        # The number of sensors to be in the mask image
        self.nsensors = nsensors
        
        # Compute device, should be GPU if possible to speed up
        self.device = device
        
    def __call__(self, data):
        
        # Move data onto device
        data = data.to(self.device)
        
        # All points in the example
        x = torch.linspace(0, data.shape[-1] - 1, steps=data.shape[-1])
        y = torch.linspace(0, data.shape[-2] - 1, steps=data.shape[-2])
        z = torch.linspace(0, data.shape[-3] - 1, steps=data.shape[-3])
        points = torch.vstack(torch.meshgrid(z, y, x)).reshape(3, -1).T
        
        # Determine the sensor locations and their associated values
        sensor_points = torch.randperm(len(points))[:self.nsensors]
        sensor_points = (points[sensor_points].long()).to(self.device)
        sensor_values = (data[:, sensor_points[:, 2], sensor_points[:, 1], sensor_points[:, 0]].T).to(self.device)
        
        # Calculate the distance of every point in sample to every sensor point, get the argmin distance
        points = (torch.repeat_interleave(points[None, :, :], self.nsensors, dim=0)).to(self.device)
        distances = torch.sqrt(torch.sum(((points - torch.unsqueeze(sensor_points, dim=1)) ** 2), dim=2))
        closest_sensor = torch.argmin(distances, dim=0)
        
        # Create the Voronoi tessellation
        voronoi = (sensor_values[closest_sensor].T).reshape(data.shape)
        
        # Create the sensor location mask
        mask = torch.zeros((data.shape[-3], data.shape[-2], data.shape[-1]))
        mask[sensor_points[:, 0], sensor_points[:, 1], sensor_points[:, 2]] = 1
        mask = torch.unsqueeze(mask, dim=0)
        
        # Concatenate the mask and Voronoi inputs
        # TODO check device and if this works on GPU
        output = torch.cat((mask.clone(), voronoi.clone()), dim=0)
        
        # Free up memory
        del data, sensor_points, sensor_values, points
        
        # Return mask and Voronoi network input
        return output
    
class ZFlip(object):
    def __init__(self, p):
        # Probability of flipping
        self.p = p
        
    def __call__(self, data):
        i = random.random()
        if i < self.p:
            return torch.flip(data, [-3])
        else:
            return data
        
class YFlip(object):
    def __init__(self, p):
        # Probability of flipping
        self.p = p
        
    def __call__(self, data):
        i = random.random()
        if i < self.p:
            return torch.flip(data, [-2])
        else:
            return data
        
class XFlip(object):
    def __init__(self, p):
        # Probability of flipping
        self.p = p
        
    def __call__(self, data):
        i = random.random()
        if i < self.p:
            return torch.flip(data, [-1])
        else:
            return data