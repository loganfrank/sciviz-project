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
    parser.add_argument('--device', default='cuda', type=str, metavar='DEV', help='device id (e.g. \'cpu\', \'cuda:0\'')

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
    test_transform = MaskAndVoronoi2D(args['nsensors'], args['nmasks'])
    test_transform.generate_masks(data=torch.empty((2, args['size'], args['size']), dtype=torch.float32))

    # Create the train and val dataset objects
    u_equation = lambda x, y: 20 * np.sin(0.05*x + 0.05*y)
    v_equation = lambda x, y: 20 * np.cos(0.05*x - 0.05*y)
    noise_equation = lambda x: 0.25 * np.cos(x)
    train_dataset = VectorFieldSim2D(u_equation=u_equation, v_equation=v_equation, noise_equation=noise_equation, num_examples=args['nexamples'][0], size=args['size'], transform=train_transform)
    val_dataset = VectorFieldSim2D(u_equation=u_equation, v_equation=v_equation, noise_equation=noise_equation, num_examples=args['nexamples'][1], size=args['size'], transform=val_transform)
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

#################################
##### Helper train function #####
#################################

def test(args, test_dataset, network):
    # Send network to GPU
    network = network.to(args['device'])

    # Create the dataloaders, seed their workers, etc.
    test_dataloader = data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    # Get the number of examples in the train, val, and test datasets
    num_test_instances = len(test_dataset)

    # Get the number of batches in the train, val, and test dataset
    num_test_batches = len(test_dataloader)

    # Arrays for keeping track of metrics for each example
        
    ### TEST
    # Disable computing gradients
    network.eval()
    torch.set_grad_enabled(False)

    # Instantiate array for keeping track of all L_2 errors
    all_errors = np.zeros(num_test_instances)
    
    # Iterate over the TEST batches
    for batch_num, (inputs, ground_truths) in enumerate(test_dataloader):
        
        # Send images and labels to compute device
        inputs = inputs.to(args['device'])
        ground_truths = ground_truths.to(args['device'])
        
        # Forward propagation
        reconstructions = network(inputs)
        
        # Threshold for flat prediction - gets the average L2 error for each example
        errors = F.mse_loss(reconstructions, ground_truths, reduction='none').mean(dim=[1, 2, 3])
        
        # Record the actual and predicted labels for the instance
        all_errors[ batch_num * args['batch_size'] : min( (batch_num + 1) * args['batch_size'], num_test_instances) ] = errors.detach().cpu().numpy()

        # Give epoch status update
        print(' ' * 100, end='\r', flush=True) 
        print(f'Testing: {100. * (batch_num + 1) / num_test_batches : 0.1f}% ({batch_num + 1}/{num_test_batches})', end='\r', flush=True)
    
    # Clear the status update message
    print(' ' * 100, end='\r', flush=True) 

    # TODO: Compute whatever metrics
    print(f'Average L2 Error: {all_errors.mean().item() : 0.5f}')


##############################################################
##### Entry point to script, retrieves command line args #####
##############################################################

if __name__ == '__main__':
    ### Get command line arguments
    args = arguments()

    ### Call main function
    main(args)
