import torch




# Given a value of t, return the Beta value
# at that given timestep using a linear trend
# Inputs:
#   B_0 - Lowest possible beta value
#   B_T - Highest possible beta value
#   T - Highest possible number of Beta timesteps
#   t - The current timestep
def linear_beta_scheduler(B_0, B_T, T, t):
    return ((B_T-B_0)/T)*t + B_0


# Given the current T value and the discriminator
# performance on real data
def T_scheduler(T, d_target, C, D_real):
    r_d = torch.mean(torch.sign(D_real - 0.5))
    return T + torch.sign(r_d - d_target)*C