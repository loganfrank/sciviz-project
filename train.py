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
from networks import ReconstructionCNN3D
from utils import worker_init_fn, make_complex, make_deterministic, adjust_weight_decay_and_learning_rate
from transforms import ZFlip, YFlip, XFlip, MaskAndVoronoi3D
from earth_dataset import EarthMantleDataset

######################################
##### Get command line arguments #####
######################################

def arguments():
    parser = argparse.ArgumentParser(description='CSE5194: Scientific Visualization Final Project!')

    # Normal parameters
    parser.add_argument('--image_dir', default='', type=str, metavar='IMG', help='prefix path to images')
    parser.add_argument('--networks_dir', default='', type=str, metavar='NET', help='prefix path to network weights')
    parser.add_argument('--results_dir', default='', type=str, metavar='RES', help='prefix path to results')
    parser.add_argument('--name', default='', type=str, metavar='NAME', help='name of experiment')
    parser.add_argument('--batch_size', default=32, type=int, metavar='BS', help='batch size')
    parser.add_argument('--learning_rate', default=0.001, type=float, metavar='LR', help='learning rate')
    parser.add_argument('--num_epochs', default=1000, type=int, metavar='NE', help='number of epochs to train for')
    parser.add_argument('--scheduler', default='True', type=str, metavar='SCH', help='should we use a lr scheduler')

    # Parameters for this project
    parser.add_argument('--nsensors', default=150, type=int, metavar='NS', help='number of sensors in our data')
    parser.add_argument('--nchannels', default=1, type=int, metavar='NC', help='1 for scalar field, 3 for vector field')

    # Parameters for reproducibility and how to train
    parser.add_argument('--seed', default=None, type=str, metavar='S', help='set a seed for reproducability')
    parser.add_argument('--device', default='cuda', type=str, metavar='DEV', help='device id (e.g. \'cpu\', \'cuda:0\'')

    # Put parameters into a dictionary
    args = vars(parser.parse_args())

    # Make sure a seed is specified
    assert args['seed'] is not None, 'Must specify a seed value'

    # Create the experiment name
    args['name'] = f'train_{args["dataset"]}' if args['name'] == '' else args['name']

    # Parse the lr scheduler option
    args['scheduler'] = (args['scheduler'] == 'True')

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
    train_transform = transforms.Compose([ZFlip(0.5), YFlip(0.5), XFlip(0.5), MaskAndVoronoi3D(args['nsensors'])])
    val_transform = transforms.Compose([MaskAndVoronoi3D(args['nsensors'])])

    # Create the train and val dataset objects
    train_dataset = EarthMantleDataset(root=f'{args["image_dir"]}train/', transform=train_transform, nchannels=args['nchannels'])
    val_dataset = EarthMantleDataset(root=f'{args["image_dir"]}val/', transform=val_transform, nchannels=args['nchannels'])
    
    # Create the network
    in_channels = args['nchannels'] + 1
    out_channels = args['nchannels'] 
    network = ReconstructionCNN3D(in_channels, out_channels)
    network = nn.DataParallel(network)
    print(network)

    # Create the loss function
    loss_function = nn.MSELoss()

    # Call the train helper function
    train(args, train_dataset, val_dataset, network, loss_function)

#################################
##### Helper train function #####
#################################

def train(args, train_dataset, val_dataset, network, loss_function):
    # Send network to GPU
    network = network.to(args['device'])

    # Create the dataloaders, seed their workers, etc.
    train_dataloader = data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    # Create the adam optimizer
    optimizer = optim.Adam(network.parameters(), lr=args['learning_rate'], weight_decay=0.0)

    # Create the learning rate scheduler
    if args['scheduler']:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['num_epochs'], eta_min=1e-8)
    else:
        scheduler = None

    # Get the number of examples in the train, val, and test datasets
    num_train_instances = len(train_dataset)
    num_val_instances = len(val_dataset)

    # Get the number of batches in the train, val, and test dataset
    num_train_batches = len(train_dataloader)
    num_val_batches = len(val_dataloader)

    # Arrays for keeping track of epoch results
    # Train
    training_losses = np.zeros(args['num_epochs'])
    
    # Test
    val_errors = np.zeros(args['num_epochs'])

    ### Training
    for epoch in range(args['num_epochs']):
        
        # Print out the epoch number
        print(f'Epoch {epoch}:')
        
        ### TRAINING
        # Prepare for training by enabling gradients
        network.train()
        torch.set_grad_enabled(True)
        
        # Instantiate the running training loss
        training_loss = 0.0

        # Keep track of best L2 error
        best_L2_error = np.inf
        
        # Iterate over the TRAINING batches
        for batch_num, (inputs, ground_truths) in enumerate(train_dataloader):
            
            # Send images and labels to compute device
            inputs = inputs.to(args['device'])
            ground_truths = ground_truths.to(args['device'])
            
            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)
                    
            # Forward propagation
            reconstructions = network(inputs)
            
            # Compute loss
            loss = loss_function(reconstructions, ground_truths)
            
            # Backward propagation
            loss.backward()
            
            # Adjust weights
            optimizer.step()
            
            # Accumulate average loss
            training_loss += loss.item()
            
            # Give epoch status update
            print(' ' * 100, end='\r', flush=True) 
            print(f'Epoch {epoch}: {100. * (batch_num + 1) / num_train_batches : 0.1f}% ({batch_num + 1}/{num_train_batches}) - Loss = {loss.item()}', end='\r', flush=True)
        
        # Clear the status update message
        print(' ' * 100, end='\r', flush=True) 
        
        # Get the average training loss
        training_loss /= num_train_batches
        print(f'Training Error: {training_loss : 0.6f}')

        # Take a LR scheduler step for step and cos only
        if scheduler is not None:
            scheduler.step()
        
        ### TEST
        # Disable computing gradients
        network.eval()
        torch.set_grad_enabled(False)

        # Instantiate array for keeping track of all L_2 errors
        all_errors = np.zeros(num_val_instances)
        
        # Iterate over the TEST batches
        for batch_num, (inputs, ground_truths) in enumerate(val_dataloader):
            
            # Send images and labels to compute device
            inputs = inputs.to(args['device'])
            ground_truths = ground_truths.to(args['device'])
            
            # Forward propagation
            reconstructions = network(inputs)
            
            # Threshold for flat prediction
            errors = F.mse_loss(reconstructions, ground_truths, reduction='none').mean(dim=[1, 2, 3, 4])
            
            # Record the actual and predicted labels for the instance
            all_errors[ batch_num * args['batch_size'] : min( (batch_num + 1) * args['batch_size'], num_val_instances) ] = errors.detach().cpu().numpy()

            # Give epoch status update
            print(' ' * 100, end='\r', flush=True) 
            print(f'Testing: {100. * (batch_num + 1) / num_val_batches : 0.1f}% ({batch_num + 1}/{num_val_batches})', end='\r', flush=True)
        
        # Clear the status update message
        print(' ' * 100, end='\r', flush=True) 
        
        # Compute test set accuracy
        average_L2_error = all_errors.mean()
        print(f'Val Error: {average_L2_error : 0.5f}')

        # Keep track of best validation model
        if average_L2_error < best_L2_error:
            print('Found improved network')
            best_L2_error = average_L2_error

            torch.save(network.state_dict(), f'{args["networks_dir"]}/{args["name"]}-best.pt')

        # Save epoch results
        # Train loss
        training_losses[epoch] = training_loss

        # Test accuracy
        val_errors[epoch] = average_L2_error

    # Save training results
    try:
        # Output training to file
        with open(f'{args["results_dir"]}/{args["name"]}.txt', 'w') as f:
            # Normal args
            f.write(f'Name: {args["name"]},, \n')
            f.write(f'Dataset: {args["dataset"]},, \n')
            f.write(f'Batch Size: {args["batch_size"]},, \n')
            f.write(f'Learning Rate: {args["learning_rate"]},, \n')
            f.write(f'Num Epochs: {args["num_epochs"]},, \n')
            f.write(f'Weight Decay: {args["weight_decay"]},, \n')
            f.write(f'LR scheduler: {args["scheduler"]},, \n')
            
            # Project args
            f.write(f'N Sensors: {args["nsensors"]},, \n')
            f.write(f'N Channels: {args["nchannels"]},, \n')

            # Reproducibility args
            f.write(f'Seed: {args["seed"]},, \n')
            f.write(f'Device: {args["device"]},, \n')

            # Print column headers
            f.write('epoch,train_error,val_error \n')

            # Zip everything into an object
            zip_object = zip(
                training_losses,
                val_errors
            )

            # Output everything
            for epoch, (train_loss, val_error) in enumerate(zip_object):
                f.write(f'{epoch},{train_loss: 0.8f},{val_error: 0.8f} \n')
    except:
        print('Error when saving file')

##############################################################
##### Entry point to script, retrieves command line args #####
##############################################################

if __name__ == '__main__':
    ### Get command line arguments
    args = arguments()

    ### Call main function
    main(args)
