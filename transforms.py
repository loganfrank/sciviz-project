import random
import torch
import numpy as np
import matplotlib.pyplot as plt

class MaskAndVoronoi2D(object):
    def __init__(self, nsensors, nmasks, device=torch.device('cpu')):
        # The number of sensors to be in the mask image
        self.nsensors = nsensors
        
        # The number of pre-computed masks to use
        self.nmasks = nmasks
        self.masks = None
        
        # Compute device, should be GPU if possible to speed up
        self.device = device
        
    def generate_masks(self, data):
        # This function will precompute several masks
        # All points in the example
        x = torch.linspace(0, data.shape[-1] - 1, steps=data.shape[-1])
        y = torch.linspace(0, data.shape[-2] - 1, steps=data.shape[-2])
        points = torch.vstack(torch.meshgrid(y, x)).reshape(2, -1).T

        # Save all (x, y) points
        self.points = points
        
        # Determine the sensor locations and their associated values
        sensor_points = [torch.randperm(len(points))[:self.nsensors] for _ in range(self.nmasks)]
        sensor_points = torch.stack([points[i] for i in sensor_points]).long()

        # Assign the masks to a class variable
        self.masks = sensor_points
        
        
    def __call__(self, data):
        
        # Move data onto device
        data = data.to(self.device)
        
        # Get a random mask
        mask_points = self.masks[torch.randperm(self.nmasks)[0].item()].to(self.device)
        mask_values = (data[:, mask_points[:, 0], mask_points[:, 1]].T).to(self.device)
        
        # Calculate the distance of every point in sample to every sensor point, get the argmin distance
        points = (torch.repeat_interleave(self.points[None, :, :], self.nsensors, dim=0)).to(self.device)
        distances = torch.sqrt(torch.sum(((points - torch.unsqueeze(mask_points, dim=1)) ** 2), dim=2))
        closest_sensor = torch.argmin(distances, dim=0)
        
        # Create the Voronoi tessellation
        voronoi = (mask_values[closest_sensor].T).reshape(data.shape)
        
        # Create the sensor location mask
        mask = torch.zeros((data.shape[-2], data.shape[-1]), device=self.device)
        mask[mask_points[:, 0], mask_points[:, 1]] = 1
        mask = torch.unsqueeze(mask, dim=0)
        
        # Concatenate the mask and Voronoi inputs
        output = torch.cat((mask.clone(), voronoi.clone()), dim=0)
        
        # Free up memory
        del mask_points, mask_values, points, distances, closest_sensor, voronoi, mask
        
        # Visualize them (uncomment below)
        # mask = output[0, :, :].cpu().numpy()
        # voronoi_v = output[1, :, :].cpu().numpy()
        # voronoi_u = output[2, :, :].cpu().numpy()
        # v = data[0, :, :].cpu().numpy()
        # u = data[1, :, :].cpu().numpy()

        # pts = np.array(self.points).reshape(96, 96, 2).transpose(2, 0, 1)

        # fig, axes = plt.subplots(1, 4)
        # axes[0].quiver(pts[0, ::2, ::2], pts[1, ::2, ::2], u[::2, ::2], v[::2, ::2], angles='xy', scale_units='xy', scale=10, headwidth=2)
        # axes[1].imshow(mask, cmap='gray')
        # axes[2].imshow(voronoi_u, cmap='gray')
        # axes[3].imshow(voronoi_v, cmap='gray')
        
        # axes[0].axis('equal')
        # axes[0].axis('off')
        # axes[1].axis('off')
        # axes[2].axis('off')
        # axes[3].axis('off')
        
        # plt.savefig('inputs.png', dpi=800)
        
        # fig = plt.Figure()
        # plt.quiver(pts[0, ::3, ::3], pts[1, ::3, ::3], u[::3, ::3], v[::3, ::3], angles='xy', scale_units='xy', scale=9, headwidth=3)
        # plt.axis('equal')
        # plt.axis('off')
        # plt.savefig('vector_field.png', dpi=1000)

        # Return mask and Voronoi network input
        return output

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
        mask = torch.zeros((data.shape[-3], data.shape[-2], data.shape[-1]), device=self.device)
        mask[sensor_points[:, 0], sensor_points[:, 1], sensor_points[:, 2]] = 1
        mask = torch.unsqueeze(mask, dim=0)
        
        # Concatenate the mask and Voronoi inputs
        output = torch.cat((mask.clone(), voronoi.clone()), dim=0)
        
        # Free up memory
        del data, sensor_points, sensor_values, points, closest_sensor, distances, mask, voronoi, x, y, z
        
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