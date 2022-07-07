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
    
    # Iterate from s=1 to s=t
    for s in range(1, t+1):
        # Get the beta values
        betas = variance_scheduler.get_betas(t)
        
        # Multiply the current a_bar values by
        # 1-betas
        a_bar *= 1-betas
        
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
    # Get the a_bar values at the current timestep
    a_bars = get_a_bars(variance_scheduler, t)
    
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
    sum = torch.sum([i for i in range(1, T+1)])
    weights = torch.tensor([(i/sum) for i in range(1, T+1)])
    sample = np.random.choice([i for i in range(0, T)], sample_size, p=weights)
    return torch.tensor(sample)