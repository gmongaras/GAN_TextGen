import torch





# Using the Wasserstein Loss Function
# Input
def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)







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