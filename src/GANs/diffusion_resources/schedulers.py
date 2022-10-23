import torch




# Given a value of t, return the Beta value
# at that given timestep using a linear trend
class linear_variance_scheduler():
    # Inputs:
    #   B_0 - Lowest possible beta value
    #   B_T - Highest possible beta value
    #   T_max - Highest possible number of Beta timesteps
    def __init__(self, B_0, B_T, T_max):
        self.B_0 = B_0
        self.B_T = B_T
        self.T_max = T_max
    
    
    # Get the variance/beta value at the given timesetps
    # Inputs:
    #   t - The current timesteps to get the Beta value at
    def __call__(self, t):
        return ((self.B_T-self.B_0)/self.T_max)*t + self.B_0


# Given the current T value and the discriminator
# performance on real data, get the updated value of T
# Inputs:
#   T - The current value of T
#   d_target - The value we want the average discriminator out
#              on real data to be
#   C - A constant multiplying the effect of the change in T
#   D_real - The discriminator output on real data
def T_scheduler(T, d_target, C, D_real):
    r_d = torch.mean(torch.sign(D_real - 0.5))
    return T + torch.sign(r_d - d_target)*C