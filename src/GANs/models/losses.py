import torch



# Loss function for the normal model which directly evaluates the output
# of the generator based on what the actual output should be
# Inputs:
#   Y_fake - The generated output from the model to evaluate
#   Y - The real data we want Y_fake to be
def binary_cross_entropy_loss(Y_fake, Y):
    # Clamp the input between 0.00001 and infinity to avoid
    # nan or -inf loss values
    Y_fake = torch.clamp(Y_fake, 0.00001, torch.inf)

    # Return the binary cross entropy loss
    return torch.nn.BCELoss(reduction="mean")(Y_fake, Y)


# Loss function for the extra loss function added to the WGAN loss.
# This directly evaluates the generator on real data to kind of push
# it in the direction of the real distribution as if it were
# a RNN generating a sequence.
# Inputs:
#   Y_fake - The generated output from the model to evaluate
#   Y - The real data we want Y_fake to be
def categorical_cross_entropy_loss(Y_fake, Y):
    # Clamp the input between 0.000001 and infinity to
    # avoid NaN or -inf loss values
    Y_fake = torch.clamp(Y_fake, 0.00001, torch.inf)
    
    # Get the CCE loss across the batch
    return torch.sum(-Y*torch.log(torch.clamp(Y_fake, 0.00001, torch.inf)), dim=-1).mean()




# In this loss function, we want the critic to minimize the
# difference between the fake and real classifications.
# For the critic loss, predictions are not between 0 and 1. So,
# minimizing this loss makes the critic give a smaller score to
# fake data and a larger score for real data.
# Inputs:
#   disc_real - Discriminator output on real data
#   disc_fake - Discriminator output on fake data
def wasserstein_disc(disc_real, disc_fake):
    return torch.mean(disc_fake) - torch.mean(disc_real)

def wasserstein_disc_split(disc_real, disc_fake):
    return torch.mean(disc_fake), -torch.mean(disc_real)

# In this loss function, we want to maximize the
# same exact thing the discriminator wants to minimize.
# So, we are essentially maximizing the discriminator's
# score on fake data since the discriminator wants to
# minimize its score on fake data
# Inputs:
#   y_pred_gen - The discriminator prediction of how real it
#                thinks the generator output is (D(G(z)))
def wasserstein_gen(y_pred_gen):
    return -torch.mean(y_pred_gen)






### Loss functions for GLS-GAN

# Inputs:
#   disc_real - Discriminator output on real data
#   disc_fake - Discriminator output on fake data
#   cost_slope - Slope of the Leaky ReLU in the cost function
def GLS_disc(disc_real, disc_fake, cost_slope, dist_funct="l1"):
    # The distance between the real and fake distributions
    if dist_funct == "l2":
        dist = torch.nn.PairwiseDistance(1)(disc_real, disc_fake)
    else:
        dist = torch.nn.PairwiseDistance(2)(disc_real, disc_fake)
    
    # Return the loss
    return torch.mean(torch.nn.LeakyReLU(cost_slope)(
        disc_real - disc_fake + dist))
    

def GLS_disc_split(disc_real, disc_fake, cost_slope, dist_funct="l1"):
    # The distance between the real and fake distributions
    if dist_funct == "l2":
        dist = torch.nn.PairwiseDistance(1)(disc_real, disc_fake)
    else:
        dist = torch.nn.PairwiseDistance(2)(disc_real, disc_fake)
        
    relu = torch.nn.LeakyReLU(cost_slope)
    
    # Return the loss values
    return torch.mean(relu(disc_real)), \
        -torch.mean(relu(disc_fake)), \
        relu(dist)


# Inputs:
#   disc_fake - Discriminator output on fake data
def GLS_gen(disc_fake):
    return torch.mean(disc_fake)






# Loss functions for the diffusion model
def diff_disc(disc_real, disc_fake):
    return torch.mean(torch.log(disc_real)) + \
        torch.mean(torch.log(1-disc_fake))
        
def diff_disc_split(disc_real, disc_fake):
    return torch.mean(torch.log(disc_real)), \
        torch.mean(torch.log(1-disc_fake))

def diff_gen(disc_fake):
    return torch.mean(torch.log(1-disc_fake))







# Normal BCE loss for the discriminator
def minimax_disc(disc_real, disc_fake):
    return -(torch.mean(torch.log(disc_real) + \
        torch.log(1-disc_fake)))

# Non-saturating GAN loss for the generator.
# The goal is to maximize how real the model thinks
# the sentence is
# Input:
#   y_pred_gen - The discriminator prediction of how real it
#                thinks the generator output is (D(G(z)))
def minimax_gen(y_pred_gen):
    return -torch.mean(torch.log(y_pred_gen))


# Minimax loss from the original GAN where the
# generator wants to minimize the loss and the
# discriminator wants to maximize the loss.
# Inputs:
#   disc_real - Discriminator output on real data
#   disc_fake - Discriminator output on fake data
def minimax_loss(disc_real, disc_fake):
    return torch.mean(torch.log(disc_real) + \
        torch.log(1-disc_fake))
