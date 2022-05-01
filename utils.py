import os
import random
import hashlib

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from skimage.io import imread, imshow

########################################################
##### Init function to seed each dataloader worker #####
########################################################

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

#####################################################
##### Generate a complex seed from a simple one #####
#####################################################

def make_complex(simple_seed, verbose=False):
    # Hash the simple seed to make high complexity representation: Good Practice in (Pseudo) Random Number Generation for Bioinformatics Applications, by David Jones (http://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf)
    m = hashlib.md5(str(simple_seed).encode('UTF-8'))

    # Convert to hex string
    hex_md5 = m.hexdigest()

    # Seed must be in range [0, 2**32 - 1], take last 32 bits (8 values * 4 bits per value = 32 bits)
    least_significant_32bits = hex_md5[-8:]

    # Convert hex to base 10
    complex_seed = int(least_significant_32bits, base=16)

    # If we want verbosity, display the value
    if verbose:
        print(f'Original seed: { simple_seed }, Complex seed: { complex_seed }, Binary value: { bin(complex_seed)[2:].zfill(32) }')

    # Confirmed: simple_seed in range [0, 88265] yield unique complex_seed values
    return complex_seed

###############################################################################
##### Make the program deterministic by setting the seeds and other flags #####
###############################################################################

def make_deterministic(seed):
    seed = make_complex(seed)
    
    # NumPy
    np.random.seed(seed)
    np.random.default_rng(seed=seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Python / OS
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set other flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#################################################################################################################
##### Properly apply weight decay to all parameters except BN parameters (scale & shift) and network biases #####
#################################################################################################################

def adjust_weight_decay_and_learning_rate(network, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []

    for name, param in network.named_parameters():
        # Skip any parameters that do not require gradients
        if not param.requires_grad:
            continue

        # Do not apply weight decay to bias parameters
        if len(param.squeeze().shape) == 1 or name in skip_list:
            no_decay.append(param)
            print(f'No weight decay for param: {name}')
        # Only apply weight decay to weight parameters
        else:
            decay.append(param)

    # Create the list to separate the parameters
    return [{ 'params': no_decay, 'weight_decay': 0.0 }, { 'params': decay, 'weight_decay': weight_decay }]
 
##### Plotting spectrograms for analysis #####



##### Fourier image difference #####

def compute_reconstruction_difference(reconstruction, ground_truth):



    norm_reconstruction = np.linalg.norm(reconstruction.cpu(), ord=2, axis=0)

    norm_ground_truth = np.linalg.norm(ground_truth.cpu(), ord=2, axis=0)

    plt.imshow(norm_reconstruction, cmap='gray')
    plt.axis('off')

    plt.imshow(norm_ground_truth, cmap='gray')
    plt.axis('off')

    norm_fft_reconstruction = np.fft.fftshift(np.fft.fft2(norm_reconstruction))
    norm_fft_ground_truth =  np.fft.fftshift(np.fft.fft2(norm_ground_truth))
    diff_image = norm_fft_ground_truth - norm_fft_reconstruction

    # Plot the differences in the frequency domain?
    
    # Inverse the 2d fft and take the abs to make the differences clear
    plt.imshow(abs(np.fft.ifft2(diff_image)), cmap='gray')
    plt.axis('off')

    # May not be necessary to convert this back.
    # It can be used for other analysis.
    # diff_image_inv  = np.fft.ifftshift(diff_image)

    return np.sum(abs(np.fft.ifft2(diff_image)))




##### Data to wave function #####

def data_to_waves(data):
    # Compute the wave f(x,y) for x and y dimensions
    
    # First: get the magnitude for direcitons i.e. collapse u v channels

    normed_data = np.linalg.norm(data.squeeze(0), ord=2, axis=0)

    # Get the wave for all x values and then all y values
    


    pass


# def 



