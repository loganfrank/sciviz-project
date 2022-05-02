# Basic Python
import sys
import os
sys.path.append(os.getcwd() + '/')
import argparse
import time
import copy
import random
import hashlib
import math
from math import pi, cos
from functools import partial

# NumPy
import numpy as np 

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms 

# Other imports
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Inter-project imports
from networks import ReconstructionCNN2D
from utils import make_complex, make_deterministic
from transforms import MaskAndVoronoi2D
from dataset import VectorFieldSim2D

######################################
##### Get command line arguments #####
######################################

def arguments():
    parser = argparse.ArgumentParser(description='CSE5194: Scientific Visualization Final Project!')

    # Normal parameters
    parser.add_argument('--name', default='train_vector_field', type=str, metavar='NAME', help='name of experiment')
    parser.add_argument('--batch_size', default=32, type=int, metavar='BS', help='batch size')
    parser.add_argument('--learning_rate', default=0.0005, type=float, metavar='LR', help='learning rate')
    parser.add_argument('--num_epochs', default=500, type=int, metavar='NE', help='number of epochs to train for')
    parser.add_argument('--scheduler', default='True', type=str, metavar='SCH', help='should we use a lr scheduler')

    # Parameters for this project
    parser.add_argument('--nsensors', default=150, type=int, metavar='NS', help='number of sensors in our data')
    parser.add_argument('--nmasks', default=20, type=int, metavar='NM', help='number of pre-generated masks to use')
    parser.add_argument('--nexamples', default='500,100,100', type=str, metavar='NEX', help='number of train, validation, and test examples')
    parser.add_argument('--size', default=96, type=int, metavar='SIZE', help='size of one training example, h=w=size (square)')

    # Parameters for reproducibility and how to train
    parser.add_argument('--seed', default='1', type=str, metavar='S', help='set a seed for reproducability')
    parser.add_argument('--device', default='cpu', type=str, metavar='DEV', help='device id (e.g. \'cpu\', \'cuda:0\'')

    # Put parameters into a dictionary
    args = vars(parser.parse_args())

    # Make sure a seed is specified
    assert args['seed'] is not None, 'Must specify a seed value'

    # Create the experiment name
    args['name'] = f'train' if args['name'] == '' else args['name']

    # Parse the lr scheduler option
    args['scheduler'] = (args['scheduler'] == 'True')

    # Parse the num train/val/test examples
    args['nexamples'] = [int(n) for n in args['nexamples'].split(',')]

    # Set the device
    if 'cuda' in args['device']:
        assert torch.cuda.is_available(), 'Device set to GPU but CUDA not available'
    args['device'] = torch.device(args['device'])

    return args
             
#########################
##### Main function #####
#########################

def main(args):
    # Set the determinism
    make_deterministic(args['seed'])

    # Create the mask and Voronoi transformations
    # 150 and 20 are hard-coded to ensure they are the same ones that were used 
    # in training and we don't repeat masks for the test set
    train_transform = MaskAndVoronoi2D(150, 20)
    train_transform.generate_masks(data=torch.empty((2, args['size'], args['size']), dtype=torch.float32))
    val_transform = MaskAndVoronoi2D(150, 20)
    val_transform.generate_masks(data=torch.empty((2, args['size'], args['size']), dtype=torch.float32))

    # Create the train and val dataset objects
    u_equation = lambda x, y: 20 * np.sin(0.05*x + 0.05*y)
    v_equation = lambda x, y: 20 * np.cos(0.05*x - 0.05*y)
    noise_equation = lambda x: 0.25 * np.cos(x)
    train_dataset = VectorFieldSim2D(u_equation=u_equation, v_equation=v_equation, noise_equation=noise_equation, num_examples=args['nexamples'][0], size=args['size'], transform=train_transform)
    val_dataset = VectorFieldSim2D(u_equation=u_equation, v_equation=v_equation, noise_equation=noise_equation, num_examples=args['nexamples'][1], size=args['size'], transform=val_transform)
    
    test_transform = MaskAndVoronoi2D(args['nsensors'], args['nmasks'])
    test_transform.generate_masks(data=torch.empty((2, args['size'], args['size']), dtype=torch.float32))
    test_dataset = VectorFieldSim2D(u_equation=u_equation, v_equation=v_equation, noise_equation=noise_equation, num_examples=args['nexamples'][1], size=args['size'], transform=test_transform)

    # We won't need these, only created so we don't accidentally
    # do data leakage between the train and test set
    del train_transform, val_transform, train_dataset, val_dataset

    # Create the network
    in_channels = 3 # U and V Voronoi tesselation and mask input = 3 input channels
    out_channels = 2 # U and V reconstruction
    network = ReconstructionCNN2D(in_channels, out_channels)
    network_path = './train_vector_field-best.pt'
    network_dict = torch.load(network_path, map_location='cpu')
    network.load_state_dict(network_dict)
    print(network)

    # Call the train helper function
    test(args, test_dataset, network)

def query_point(points):
    temp = ((0 <= points[:, 0]) & (points[:, 0] <= 95))
    temp2= ((0 <= points[:, 1]) & (points[:, 1] <= 95))
    return np.all(np.stack([temp, temp2]), axis=0).reshape(-1, 1)
        
def get_cell(points: np.ndarray) -> list:
    # Transform the point from physical space to the data space
    transformedx, transformedy = points[:, 1], points[:, 0]
    
    # Determine the bounds of the cell
    xfloor = np.floor(transformedx)
    xceil = np.floor(transformedx) + 1
    yfloor = np.floor(transformedy)
    yceil = np.floor(transformedy) + 1
    
    # Handle upper edge case
    xfloor = np.where(xfloor == 95, 94, xfloor).astype(np.uint16)
    xceil = np.where(xceil == 96, 95, xceil).astype(np.uint16)
    yfloor = np.where(yfloor == 95, 94, yfloor).astype(np.uint16)
    yceil = np.where(yceil == 96, 95, yceil).astype(np.uint16)
    
    # Compute the interpolation weights
    xweight = (transformedx - xfloor) / (xceil - xfloor)
    yweight = (transformedy - yfloor) / (yceil - yfloor)
    
    bottom_left = [yfloor, xfloor]
    bottom_right = [yfloor, xceil]
    upper_left = [yceil, xfloor]
    upper_right = [yceil, xceil]
    
    return np.transpose(np.array([bottom_left, bottom_right, upper_left, upper_right], dtype=np.uint8), axes=(0, 2, 1)), np.array([xweight, yweight]).T

def get_value(field, cells, interpolation_weights):
    # Each cell is [x, y, z]
    bottom_left = cells[0, :, :]
    bottom_right = cells[1, :, :]
    upper_left = cells[2, :, :]
    upper_right = cells[3, :, :]
    
    # Get the values at the indexes
    bottom_left = field[..., bottom_left[:, 0], bottom_left[:, 1]]
    bottom_right = field[..., bottom_right[:, 0], bottom_right[:, 1]]
    upper_left = field[..., upper_left[:, 0], upper_left[:, 1]]
    upper_right = field[..., upper_right[:, 0], upper_right[:, 1]]

    
    # Get the interpolation weights
    wx = interpolation_weights[:, 0]
    wy = interpolation_weights[:, 1]
    
    lerp1 = (1 - wx) * upper_left + wx * upper_right
    lerp2 = (1 - wx) * bottom_left + wx * bottom_right
    
    # Perform bilinear interpolation
    return ((1 - wy) * lerp2 + wy * lerp1).T

def query_value(field, points):
    assert np.all(query_point(points)), 'must provide points in grid bounds'
    cells, interpolation_weights = get_cell(points)
    return get_value(field, cells, interpolation_weights)
    
### Euler Particle Tracing Method, slightly altered from lab 3
def trace_euler(field: np.ndarray, starting_positions: np.ndarray, num_steps: int = 200, step_size: float = 0.5, verbose: bool = False) -> np.ndarray:
    # Create the array to keep track of all positions of the particle
    points = np.zeros((num_steps + 1, *starting_positions.shape))
    points[0, :, :] = starting_positions
    
    # Create the variable for the current position of the particle and initialize it to be the starting positions
    current_positions = starting_positions
    
    # Perform particle tracing
    for step in range(1, num_steps + 1):
        # Find the vector values at the current position
        vectors = query_value(field, current_positions)
        vectors = vectors / (np.sqrt((vectors ** 2).sum(axis=1)).reshape(-1, 1) + 1e-8)
        
        # Update the position of the particle (depending on if it will be in bounds or not)
        next_step = current_positions + (vectors * step_size)
        next_step_in_bounds = query_point(next_step)
        current_positions = np.where(next_step_in_bounds, next_step, current_positions)
        
        # Add the path of the particle
        points[step, :, :] = current_positions
        
        # Print status
        if verbose:
            print(' ' * 100, end='\r', flush=True) 
            print(f'Tracing: {100. * step / num_steps}%', end='\r', flush=True)
    
    # Clear the status update message
    if verbose:
        print(' ' * 100, end='\r', flush=True) 
    
    return np.hstack(points)

#################################
##### Helper train function #####
#################################

def test(args, test_dataset, network):
    # Send network to GPU
    network = network.to(args['device'])

    # Create the dataloaders, seed their workers, etc.
    test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Get the number of examples in the train, val, and test datasets
    num_test_instances = len(test_dataset)

    # Get the number of batches in the train, val, and test dataset
    num_test_batches = len(test_dataloader)

    # Arrays for keeping track of metrics for each example
        
    ### TEST
    # Disable computing gradients
    network.eval()
    torch.set_grad_enabled(False)

    # The starting points for particle tracing
    # All points in the example
    size = 96
    x = np.linspace(0, size - 1, num=size)
    y = np.linspace(0, size - 1, num=size)
    starting_points = np.stack(np.meshgrid(y, x, indexing='ij')).reshape(2, -1).T
    pts = starting_points.reshape(96, 96, 2).transpose(2, 0, 1)
    
    # Iterate over the TEST batches
    for batch_num, (inputs, ground_truths) in enumerate(test_dataloader):
        
        # Send images and labels to compute device
        inputs = inputs.to(args['device'])
        ground_truths = ground_truths.to(args['device'])
        
        # Forward propagation
        reconstructions = network(inputs)
        reconstructions = reconstructions.squeeze().detach().cpu().numpy()
        ground_truths = ground_truths.squeeze().detach().cpu().numpy()
        
        # Perform particle tracing
        reconstruction_paths = trace_euler(reconstructions, starting_points, 200, 0.25)
        reconstruction_paths = reconstruction_paths[[1716, 2505, 7050, 8018], :]
        reconstruction_paths = reconstruction_paths.reshape(4, -1, 2)
        ground_truth_paths = trace_euler(ground_truths, starting_points, 200, 0.25)
        ground_truth_paths = ground_truth_paths[[1716, 2505, 7050, 8018], :]
        ground_truth_paths = ground_truth_paths.reshape(4, -1, 2)
        
        ur = reconstructions[0, :, :]
        vr = reconstructions[1, :, :]
        ugt = ground_truths[0, :, :]
        vgt = ground_truths[1, :, :]
        
        fig, axes = plt.subplots(1, 2)
        axes[0].quiver(pts[0, ::3, ::3], pts[1, ::3, ::3], ur[::3, ::3], vr[::3, ::3], angles='xy', scale_units='xy', scale=9, headwidth=3)
        axes[0].axis('equal')
        axes[0].axis('off')
        axes[0].plot(reconstruction_paths[0, :, 0], reconstruction_paths[0, :, 1], label='path1')
        axes[0].plot(reconstruction_paths[1, :, 0], reconstruction_paths[1, :, 1], label='path2')
        axes[0].plot(reconstruction_paths[2, :, 0], reconstruction_paths[2, :, 1], label='path3')
        axes[0].plot(reconstruction_paths[3, :, 0], reconstruction_paths[3, :, 1], label='path4')
        
        axes[1].quiver(pts[0, ::3, ::3], pts[1, ::3, ::3], ur[::3, ::3], vr[::3, ::3], angles='xy', scale_units='xy', scale=9, headwidth=3)
        axes[1].axis('equal')
        axes[1].axis('off')
        axes[1].plot(ground_truth_paths[0, :, 0], ground_truth_paths[0, :, 1], label='path1')
        axes[1].plot(ground_truth_paths[1, :, 0], ground_truth_paths[1, :, 1], label='path2')
        axes[1].plot(ground_truth_paths[2, :, 0], ground_truth_paths[2, :, 1], label='path3')
        axes[1].plot(ground_truth_paths[3, :, 0], ground_truth_paths[3, :, 1], label='path4')
        plt.show()

        # Give epoch status update
        print(' ' * 100, end='\r', flush=True) 
        print(f'Testing: {100. * (batch_num + 1) / num_test_batches : 0.1f}% ({batch_num + 1}/{num_test_batches})', end='\r', flush=True)
    
    # Clear the status update message
    print(' ' * 100, end='\r', flush=True) 


##############################################################
##### Entry point to script, retrieves command line args #####
##############################################################

if __name__ == '__main__':
    ### Get command line arguments
    args = arguments()

    ### Call main function
    main(args)
