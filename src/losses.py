import torch





# In this loss function, we want the critic to maximize the
# difference between the fake and real classifications.
# For the critic loss, predictions are not between 0 and 1. So,
# maximizing this loss makes the critic give a larger score to
# real data and a smaller score for fake data.
# Inputs:
#   disc_real - Discriminator output on real data
#   disc_fake - Discriminator output on fake data
def wasserstein_disc(disc_real, disc_fake):
    return -(torch.mean(disc_real) - torch.mean(disc_fake))

# In this loss function, we want to maximize the score of the
# critic when the critic is given fake data. Since the critic
# classifies real data with a larger score, we want the generator
# to make data that's more real, so it'll maximize the
# value the discriminator gives.
# Inputs:
#   y_pred_gen - The discriminator prediction of how real it
#                thinks the generator output is (D(G(z)))
def wasserstein_gen(y_pred_gen):
    return -torch.mean(y_pred_gen)







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