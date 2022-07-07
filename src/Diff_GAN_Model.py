from helpers.helpers import encode_sentences
from helpers.helpers import encode_sentences_one_hot
from helpers.helpers import addPadding
from helpers.helpers import addPadding_one_hot


from models.losses import wasserstein_disc
from models.losses import wasserstein_disc_split
from models.losses import wasserstein_gen
from models.losses import minimax_disc
from models.losses import minimax_gen
from models.losses import minimax_loss


from models.Generator import Generator
from models.Discriminator import Discriminator
import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt



cpu = torch.device('cpu')
gpu = torch.device('cuda:0')




class Diff_GAN_Model(nn.Module):
    # Inputs:
    #   vocab - A dictionary of vocab where the keys are integers and the
    #           values are words
    #   M_gen - The number of blocks in the encoder part of the generator model
    #   N_gen - The number of blocks in the decoder part of the generator model
    #   N_disc - The number of blocks in the discriminator model
    #   batchSize - Size to bach data into
    #   embedding_size - The size of the vector to embed each word
    #   sequence_length - The max length of the sentence to train the model on
    #   num_heads - Number of heads for the MHA modules
    #   trainingRatio - 2-D array representing the number of epochs to 
    #                   train the generator (0) vs the discriminator (1)
    #   decRatRate - Decrease the ratio after every decRatRate steps. Use -1 to
    #                never decrease the ratio
    #   pooling - What pooling mode should be used? ("avg", "max", or "none")
    #   embed_mode - What embedding mode should be used for the
    #                generator? ("norm" or "custom")
    #   alpha - Learning rate of the model
    #   Lambda - Lambda value used for gradient penalty in disc loss
    #   Beta1 - Adam beta 1 term
    #   Beta2 - Adam beta 2 term
    #   device - Device to put the model on
    #   saveSteps - Number of steps until the model is saved
    #   saveDir - Directory to save the model to
    #   genSaveFile - Name of the file to save the generator model to
    #   discSaveFile - Name of the file to save the discriminator model to
    #   trainGraphFile - File to save training graph during training
    #   Beta_0 - Lowest possible Beta value, when t is 0
    #   Beta_T - Highest possible Beta value, when t is T
    #   T_min - Min diffusion steps when corrupting the data
    #   T_max - Max diffusion steps when corrupting the data
    #   sigma - Addative noise weighting term
    #   d_target - Term used for the T scheduler denoting if the 
    #               T change should be positive of negative depending 
    #               on the disc output
    #   C - Constant for the T scheduler multiplying the change of T
    def __init__(self, vocab, M_gen, N_gen, N_disc, batchSize, embedding_size, sequence_length, num_heads, trainingRatio, decRatRate, pooling, embed_mode, alpha, Lambda, Beta1, Beta2, device, saveSteps, saveDir, genSaveFile, discSaveFile, trainGraphFile, Beta_0, Beta_T, T_min, T_max, sigma, d_target, C):
        super(Diff_GAN_Model, self).__init__()
        
        # The ratio must not have a lower value for the discriminator (1)
        assert trainingRatio[0]<=trainingRatio[1], "The training ratio must have a grater number in the zeroth index"
        
        # Save the needed variables
        self.vocab = vocab
        self.vocab_inv = {vocab[i]:i for i in vocab.keys()}
        self.sequence_length = sequence_length
        self.batchSize = batchSize
        self.trainingRatio = trainingRatio
        self.decRatRate = decRatRate
        self.Lambda = Lambda
        self.embed_mode = embed_mode
        
        # Saving paramters
        self.saveSteps = saveSteps
        self.saveDir = saveDir
        self.genSaveFile = genSaveFile
        self.discSaveFile = discSaveFile
        self.trainGraphFile = trainGraphFile
        
        # Diffusion parameters
        self.Beta_0 = Beta_0
        self.Beta_T = Beta_T
        self.T_min = T_min
        self.T_max = T_max
        self.sigma = sigma
        self.d_target = d_target
        self.C = C

        # Convert the device to a torch device
        if device.lower() == "fullgpu":
            if torch.cuda.is_available():
                dev = device.lower()
                device = torch.device('cuda:0')
            else:
                dev = "cpu"
                print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
                device = torch.device('cpu')
        else:
            dev = device.lower()
            device = torch.device('cpu')
        self.device = device
        self.dev = dev
        
        # The generator and discriminator models
        if self.dev != "cpu":
            self.generator = Generator(vocab, M_gen, N_gen, batchSize, embedding_size, sequence_length, num_heads, embed_mode, gpu)
            self.discriminator = Discriminator(N_disc, batchSize, len(vocab), embedding_size, sequence_length, num_heads, pooling, gpu)
        else:
            self.generator = Generator(vocab, M_gen, N_gen, batchSize, embedding_size, sequence_length, num_heads, embed_mode, device)
            self.discriminator = Discriminator(N_disc, batchSize, len(vocab), embedding_size, sequence_length, num_heads, pooling, device)
        
        # The optimizer for the model
        self.optim_gen = torch.optim.Adam(self.generator.parameters(), alpha, betas=[Beta1, Beta2])
        self.optim_disc = torch.optim.Adam(self.discriminator.parameters(), alpha, betas=[Beta1, Beta2])