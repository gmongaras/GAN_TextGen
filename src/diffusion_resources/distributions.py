import torch
import numpy as np





# Given a t value, get the a_bar value at that timestep t
# Inputs:
#   variance_scheduler - Scheduler used to get the variance/beta
#                        value at the given t value
#   t - Batch of t values to get the a_bar values at
def get_a_bars(variance_scheduler, t):
    # The a_bar values
    a_bar = torch.ones(t.shape)
    
    # Iterate over all t values to get their
    # a_bar value
    for i in range(0, len(t)):
        # Create the array of values to multiply together
        prods = [1-variance_scheduler(s) for s in range(1, int(t[i].item()))]
        
        # Get the a_bar value and save it
        a_bar[i] = np.prod(prods)
        
    # Return the a bar values
    return a_bar






# Given some data data, add noise to it through diffusion
# to get a noisy sample. This data can be real or fake
# Inputs:
#   data - Batch of data to transform
#   sigma - Standard deviation of the noise to add to the data
#   variance_scheduler - Scheduler used to get the variance/beta
#                        value at the given t value
#   t - Batch of t values corresponding to the batch of
#       data to retreive the variance at
def y_sample(data, sigma, variance_scheduler, t):
    # Get the a_bar values at the current timestep.
    # The a_bar value weights the data higher when
    # t is small and the noise higher when t is large,
    # thus corrupting the data more when t is higher.
    a_bars = get_a_bars(variance_scheduler, t)
    a_bars = a_bars.unsqueeze(-1).unsqueeze(-1).repeat(1, data.shape[1], data.shape[2])
    
    # Term 1: Scaled data
    scaled_data_gen = torch.sqrt(a_bars)*data
    
    # Term 2: Additive noise
    epsilon = torch.randn_like(data)
    additive_noise = torch.sqrt(1-a_bars)*sigma*epsilon
    
    # Return the final values
    return scaled_data_gen + additive_noise




# Sample from the discrete p_pie distribution to get a new
# sequence of t values. The distributuion has values from 1
# to T and each value is weighted by v/(sum from 1 to T)
# Inputs:
#   sample_size - The number of values to sample from the distribution
#   T - Current T value
def p_pie_sample(sample_size, T):
    weights = np.arange(T)/np.arange(T).sum()
    sample = np.random.choice(np.arange(1, T+1), size=sample_size, p=weights)
    return sample