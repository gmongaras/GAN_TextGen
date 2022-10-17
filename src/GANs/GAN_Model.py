from ..helpers.helpers import encode_sentences
from ..helpers.helpers import encode_sentences_one_hot
from ..helpers.helpers import addPadding
from ..helpers.helpers import addPadding_one_hot


from .models.losses import wasserstein_disc
from .models.losses import wasserstein_disc_split
from .models.losses import wasserstein_gen
from .models.losses import GLS_disc
from .models.losses import GLS_disc_split
from .models.losses import GLS_gen
from .models.losses import minimax_disc
from .models.losses import minimax_gen
from .models.losses import minimax_loss
from .models.losses import categorical_cross_entropy_loss


from .models.Generator import Generator
from .models.Discriminator import Discriminator

import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt
import os



cpu = torch.device('cpu')
try:
    if torch.has_mps:
        gpu = torch.device("mps")
    else:
        gpu = torch.device('cuda:0')
except:
    gpu = torch.device('cuda:0')





class GAN_Model(nn.Module):
    # Inputs:
    #   vocab - A dictionary of vocab where the keys are integers and the
    #           values are words
    #   M_gen - Number of noise encoding blocks in the generator
    #   B_gen - Number of transformer blocks to encode the input sequence
    #   O_gen - Number of transformer blocks to get the output sequence
    #   L_gen - Number of transformer blocks to encode the lengths
    #   T_disc - Number of transformer blocks in each discriminator block
    #   B_disc - Number of discriminator blocks in the discriminator
    #   O_disc - Number of output MHA blocks in the discrimiantor
    #   hiddenSize - Hidden linear size in the transformer blocks
    #   noiseDist - Distribution to sample noise from. Can be one of 
    #               (\"norm\", \"unif\", \"trunc\" (for truncated normal))
    #   costSlope - The slope of the GLS-GAN cost function
    #   batchSize - Size to bach data into
    #   embedding_size_gen - Embedding size of the generator
    #   embedding_size_disc - Embedding size of the discriminator
    #   sequence_length - The max length of the sentence to train the model on
    #   num_heads - Number of heads for the MHA modules
    #   dynamic_n - True to dynamically change the number of times to train the models. False otherwise
    #   Lambda_n - (Only used if dynamic_n is True) Amount to scale the discriminator over 
    #               the generator to give the discriminator a higher weight (when >1) or the 
    #               generator a higher weight (when <1)
    #   HideAfterEnd - True to hide any tokens after the <END> token in the 
    #                  discriminator MHA with a mask, False to keep these tokens visibile
    #   n_D - Number of times to train the discriminator more than the generator for each epoch
    #   pooling - What pooling mode should be used? ("avg", "max", or "none")
    #   gen_outEnc_mode - How should the generator encode its output? ("norm" or "gumb")
    #   embed_mode_gen - What embedding mode should be used for the
    #                    generator? ("norm" or "custom")
    #   embed_mode_disc - What embedding mode should be used for the
    #                     discriminator? ("fc" or "pca")
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
    #   loadInEpoch - Should the data be loaded in as needed instead of
    #                 before training (True if so, False to load before training)
    #   delWhenLoaded - Delete the data as it's loaded in to save space?
    #                   Note: This is automatically False if loadInEpoch is True
    def __init__(self, vocab, M_gen, B_gen, O_gen, L_gen, T_disc, B_disc, O_disc, hiddenSize, noiseDist, costSlope, batchSize, embedding_size_gen, embedding_size_disc, sequence_length, num_heads, dynamic_n, Lambda_n, HideAfterEnd, n_D, pooling, gen_outEnc_mode, embed_mode_gen, embed_mode_disc, alpha, Lambda, Beta1, Beta2, device, saveSteps, saveDir, genSaveFile, discSaveFile, trainGraphFile, loadInEpoch, delWhenLoaded):
        super(GAN_Model, self).__init__()
        
        # Save the needed variables
        self.vocab = vocab
        self.vocab_inv = {vocab[i]:i for i in vocab.keys()}
        self.sequence_length = sequence_length
        self.batchSize = batchSize
        self.n_D = n_D
        self.Lambda = Lambda
        self.loadInEpoch = loadInEpoch
        self.dynamic_n = dynamic_n
        self.Lambda_n = Lambda_n
        self.HideAfterEnd = HideAfterEnd
        self.delWhenLoaded = delWhenLoaded if self.loadInEpoch == False else False
        self.embedding_size_disc = embedding_size_disc
        self.costSlope = costSlope
        self.num_heads = num_heads
        
        # Saving paramters
        self.saveSteps = saveSteps
        self.saveDir = saveDir
        self.genSaveFile = genSaveFile
        self.discSaveFile = discSaveFile
        self.trainGraphFile = trainGraphFile
        
        # Parameter for changing n_G
        self.Beta = 1 # Starting Beta value

        # Convert the device to a torch device
        if device.lower() == "fullgpu":
            if torch.cuda.is_available():
                dev = device.lower()
                device = torch.device('cuda:0')
            elif torch.has_mps == True:
                dev = "mps"
                device = torch.device('mps')
            else:
                dev = "cpu"
                print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
                device = torch.device('cpu')
        else:
            dev = device.lower()
            device = torch.device('cpu')
        self.device = device
        self.dev = dev
        
        # The generator, discriminator, and teacher models
        if self.dev != "cpu":
            self.generator = Generator(vocab, M_gen, B_gen, O_gen, L_gen, noiseDist, hiddenSize, batchSize, embedding_size_gen, sequence_length, num_heads, embed_mode_gen, gen_outEnc_mode, gpu)
            self.discriminator = Discriminator(T_disc, B_disc, O_disc, "none", hiddenSize, batchSize, len(vocab), embedding_size_disc, sequence_length, num_heads, pooling, embed_mode_disc, gpu)
        else:
            self.generator = Generator(vocab, M_gen, B_gen, O_gen, L_gen, noiseDist, hiddenSize, batchSize, embedding_size_gen, sequence_length, num_heads, embed_mode_gen, gen_outEnc_mode, device)
            self.discriminator = Discriminator(T_disc, B_disc, O_disc, "none", hiddenSize, batchSize, len(vocab), embedding_size_disc, sequence_length, num_heads, pooling, embed_mode_disc, device)
        
        # The optimizer for the models
        self.optim_gen = torch.optim.Adam(self.generator.parameters(), alpha, betas=[Beta1, Beta2])
        self.optim_disc = torch.optim.Adam(self.discriminator.parameters(), alpha, betas=[Beta1, Beta2])

        # Uniform distribution for the gradient penalty
        self.unif = torch.distributions.uniform.Uniform(0, 1)

        # Dynamic generator training estimation
        self.G_n = 1
        self.ROC = 1
        
        
    def one_hot(a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
    
    
    # For WGAN-GP loss, we need a gradient penalty to add
    # to the discriminator loss. This function gets that
    # penalty
    # Inputs:
    #   x - A sample of real data
    #   x_tilde - A sample of data from the generator
    def get_gradient_penalty(self, x, lens_real, x_tilde, lens_fake, \
        masks_real, masks_fake):
        # Get the device
        device = self.device
        if self.dev == "partgpu":
            device = gpu

        # Sample a uniform distribution to get a random number, epsilon
        epsilon = self.unif.sample((self.batchSize, 1, 1)).to(device)
        
        # Create a new tensor fo the same shape as the real and fake data
        lens_hat = epsilon.squeeze(-1)*lens_real + (1-epsilon.squeeze(-1))*lens_fake.squeeze()
        epsilon = epsilon.expand(x.shape)
        x_hat = epsilon*x + (1-epsilon)*x_tilde

        # Masks are the combination of the two masks
        masks_hat = torch.logical_and(masks_real, masks_fake).clone().detach()

        # We only want the gradients of the discriminator
        x_hat = x_hat.clone().detach()
        x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
        lens_hat = lens_hat.clone().detach()
        lens_hat = torch.autograd.Variable(lens_hat, requires_grad=True)
        
        # Send the transformed and combined data through the
        # discriminator
        disc_hat, disc_lens_hat = self.discriminator(x_hat, lens_hat, masks_hat)
        
        # Get the gradients of the discriminator output on the sentences
        gradients = torch.autograd.grad(outputs=(disc_hat), inputs=(x_hat),
                              grad_outputs=(torch.ones(disc_hat.size(), device=device)),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Gradient penalty of the sentences
        GP_sent = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.Lambda

        # Get the gradients of the discriminator output on the sentences
        gradients = torch.autograd.grad(outputs=(disc_lens_hat), inputs=(lens_hat),
                              grad_outputs=(torch.ones(disc_lens_hat.size(), device=device)),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Gradient penalty of the sentences
        GP_lens = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.Lambda
        
        # Get the final penalty and return it
        return GP_sent, GP_lens
    
    
    
    
    # Given some encoded sentence data, return the
    # lengths of each sentence
    # Input:
    #   data - Data of the shape (N, S, V)
    # Output:
    #   Positions of the shape (N)
    def getLens(self, data):
        # Position of any <END> tokens in the outut. Defaults
        # to the last position.
        end_pos = data.shape[1]*torch.ones((data.shape[0]), requires_grad=False, dtype=torch.int16)
        
        # Get the position of the first <END> token for each generated sequence
        whereEnd = torch.where(torch.argmax(data, dim=-1).cpu() == self.vocab_inv["<END>"])
        uniq = torch.unique(whereEnd[0], dim=0, sorted=False, return_inverse=True, return_counts=True)
        vals = torch.zeros(uniq[0].shape, dtype=torch.int16, requires_grad=False)
        i = 0
        for j in range(0, uniq[0].shape[0]):
            vals[j] = whereEnd[1][i]
            i += uniq[2][j]
        end_pos[uniq[0]] = vals.to(torch.int16)
        end_pos = end_pos.long()
            
        # Return the output and the masks
        return end_pos.to(self.device if self.dev != "partgpu" else gpu)
    

    # Given lengths for a sentence, return the MHA mask
    # for that sentence
    # Input:
    #   Tensor of shape (N, S) where each item is the softmax
    #   encoded length for that sentence
    # Output:
    #   Tensor of shape (num_heads*N, L, L) where teh second L stores
    #   the positions in the sentence to ignore
    def getMask(self, lens_soft):
        # Get the argmax of the lengths
        lens = torch.argmax(lens_soft, dim=-1)

        # # Masks
        # masks = torch.zeros((self.batchSize, self.sequence_length), dtype=torch.bool)

        # # For each length, create a binary tensor with True
        # # after the length and False before
        # for i in range(0, self.batchSize):
        #     l = lens[i]
        #     m = masks[i]

        #     m[l+1:] = True
        #     masks[i] = m


        """
        What do the masks look like?
        Sequence: 'word word word <END> <PAD> <PAD>'
        Size: (1, 6, E)
        Mask:
        [[0, 0, 0, 0, -inf, -inf]
         [0, 0, 0, 0, -inf, -inf]
         [0, 0, 0, 0, -inf, -inf]
         [0, 0, 0, 0, -inf, -inf]
         [0, 0, 0, 0, -inf, -inf]
         [0, 0, 0, 0, -inf, -inf]]

        So the mask is of shape (N, L, L) where the mask has
        -inf after the length and 0 before the length in the Nth
        sequence

        Why are the -infs in the columns and not the rows?
        When the matrix S = soft(QK) is created, the columns will be multiplied
        by the rows in the V matrix, as matrix multiplication goes. So, the columns
        with -inf will be multiplied by the row with the same index, which will be
        the row with the <PAD> encoding in it.
        """


        # Create a 2-d mask of each sequene
        masks = torch.zeros((self.batchSize, self.sequence_length, self.sequence_length), dtype=torch.float)

        # For each length, create a binary tensor with True
        # after the length and False before
        for i in range(0, self.batchSize):
            l = lens[i]
            masks[i, :, min(l+1, self.sequence_length):] = -np.inf

        # Broadcast the masks over the number of heads
        masks = torch.repeat_interleave(masks, self.num_heads, dim=0)

        return masks.to(self.device if self.dev != "partgpu" else gpu)
    
    
    
    # Train the models
    # Input:
    #   X - A list of sentences to train the models on
    #   epochs - Number of epochs to train the models for
    def train_model(self, X, epochs):
        # Encode the all the data if self.loadInEpoch is false
        if self.loadInEpoch == False:
            X_orig = encode_sentences(X, self.vocab_inv, self.sequence_length, lambda x: x, self.delWhenLoaded, self.device)
            s = len(X_orig)
            del X
        else:
            X = np.array(X, dtype=object)
            s = X.shape[0]
        
        # Save loss values over training for the loss plot
        self.genLoss = []
        self.discLoss = []
        self.discLoss_real = []
        self.discLoss_fake = []
        self.MDs = []
        self.genLoss_lens = []
        self.discLoss_lens = []
        self.discLoss_real_lens = []
        self.discLoss_fake_lens = []
        
        # Initial variable states
        masks = None
        r_d = 1 # Initial states for dynamic_n
        r_g = 0
        L_p_g = None
        L_p_d = None
        
        # Train the model for epochs number of epochs
        for epoch in range(1, epochs+1):
            # Model saving
            if epoch % self.saveSteps == 0:
                self.saveModels(self.saveDir, self.genSaveFile, self.discSaveFile, epoch)
            
            # Create a list of indices which the Discriminator
            # has left to see and the Generator has left to see
            disc_nums = torch.randperm(s, device=self.device)
            
            # Train the discriminator first or if dynamic_n is used and
            # r_d > r_g
            self.optim_disc.zero_grad()
            disc_loss_avg = 0
            disc_fake_avg = 0
            disc_real_avg = 0
            disc_loss_lens_avg = 0
            disc_fake_lens_avg = 0
            disc_real_lens_avg = 0
            for i in range(0, max(self.n_D, 1) if self.dynamic_n == False else 1):
                # Sample data for the discriminator
                disc_sub = disc_nums[:self.batchSize]
                disc_nums = disc_nums[self.batchSize:]

                # Put the data on the correct device
                if self.dev == "partgpu":
                    disc_sub = disc_sub.to(gpu)
                else:
                    disc_sub = disc_sub.to(self.device)
                
                # Generate some data from the generator
                with torch.no_grad():
                    fake_X = self.generator.forward(training=False)
                
                # Get the real sentences as one-hot sequences
                real_X = [torch.nn.functional.one_hot(X_orig[i], len(self.vocab)) for i in disc_sub.cpu().numpy()]
                
                # Add padding to the real data
                real_X = addPadding_one_hot(real_X, self.vocab_inv, self.sequence_length)
                if self.dev == "partgpu":
                    real_X = real_X.to(gpu)
                else:
                    real_X = real_X.to(self.device)
                
                # Get the variance
                # sampleSize = 100000
                # maxVariance = len(self.vocab)/100
                # minVariance = 0.1
                # variance = torch.linspace(maxVariance, minVariance, epochs)[epoch]
                
                # # Iterate over all non-<PAD> tokens in the real data
                # # and spread the data distribution
                # for i in range(0, len(real_X)):
                #     for j in range(0, lens_real[i].cpu().item()):
                #         # Get the current distribution to spread
                #         curDist = real_X[i, j]
                        
                #         # Get the argmax of the distribution
                #         mean = torch.argmax(curDist).float()
                        
                #         # Get a normal distribution with a mean
                #         # of the current value and a variance of
                #         # the calculated variance
                #         N = torch.distributions.normal.Normal(mean.cpu(), variance.cpu())
                        
                #         # Sample the distribution
                #         S = N.sample([sampleSize]) # Sample
                #         S = S[torch.logical_and(S < len(self.vocab)-1, S > 0)] # Only get values in the desired range
                #         S = torch.round(S).long() # Discretize the sample
                        
                #         # Get the distribution
                #         N_disc = torch.bincount(S, minlength=len(self.vocab)).float()
                        
                #         # Make the values sum up to 1
                #         N_disc /= N_disc.sum()
                        
                #         # The discrete distribution will now
                #         # replace the one-hot vector
                #         real_X[i, j] = N_disc.to(self.device)




                # Sample half a batch from the real and fake generated data
                # to generate a batch of half real and half fake sentences
                fake_X_samp = fake_X[np.random.permutation(np.arange(self.batchSize))[:self.batchSize//2]]
                real_X_samp = real_X[np.random.permutation(np.arange(self.batchSize))[:self.batchSize//2]]

                # Create the batch of data to generate lengths for
                X_samp = torch.cat((fake_X_samp, real_X_samp), dim=0)

                # Get the fake lengths from the generator
                with torch.no_grad():
                    lens_fake = self.generator.forward_lens(X_samp)




                # Get the real data lengths
                lens_real = self.getLens(real_X)

                # One hot encode the lengths and masks
                lens_real = torch.nn.functional.one_hot(lens_real, self.sequence_length).float()
                masks_real = self.getMask(lens_real)

                # Get the masks for the generator
                masks_fake = self.getMask(lens_fake)
                
                # Send the generated output through the discriminator
                # to get a batch of predictions on the fake sentences
                disc_fake, disc_lens_fake = self.discriminator(fake_X, lens_fake, masks_fake)

                # Make the inputs into the discriminator loss variables
                # for differentiation
                real_X = torch.autograd.Variable(real_X, requires_grad=True)
                fake_X = torch.autograd.Variable(fake_X, requires_grad=True)
                lens_real = torch.autograd.Variable(lens_real, requires_grad=True)
                lens_fake = torch.autograd.Variable(lens_fake, requires_grad=True)

                # Get the gradient penalty
                gradient_penalty, gradient_penalty_lens = \
                    self.get_gradient_penalty(real_X, lens_real, fake_X, lens_fake, \
                    masks_real, masks_fake)

                # Calculate mean difference for debugging
                MD = torch.abs(real_X-fake_X).sum(-1).sum(-1).mean().detach().cpu()
                
                # Send the generated output through the discriminator
                # to get a batch of predictions on the real sentences
                disc_real, disc_lens_real = self.discriminator(real_X, lens_real, masks_real)
                
                # Get the discriminator loss
                # discLoss = GLS_disc(disc_real, disc_fake, real_X, fake_X, self.costSlope, "l1")
                discLoss = wasserstein_disc(disc_real, disc_fake)
                discLoss_lens = wasserstein_disc(disc_lens_real, disc_lens_fake)
                
                # discLoss_real, discLoss_fake, dist = GLS_disc_split(disc_real, disc_fake, real_X, fake_X, self.costSlope, "l1")
                discLoss_fake, discLoss_real = wasserstein_disc_split(disc_real, disc_fake)
                discLoss_lens_fake, discLoss_lens_real = wasserstein_disc_split(disc_lens_real, disc_lens_fake)

                # The cost of the discriminator is the loss + the penalty
                discCost = discLoss + gradient_penalty
                discCost_lens = discLoss_lens + gradient_penalty_lens
                
                if self.dynamic_n == False or (self.dynamic_n == True and r_d > self.Lambda_n*r_g):
                    # Backpropogate the cost
                    (discCost + discCost_lens).backward()
                    
                    # Step the optimizer
                    self.optim_disc.step()
                self.optim_disc.zero_grad()

                # Loss cumulation
                disc_loss_avg += discLoss.cpu().detach().item()
                disc_fake_avg += discLoss_fake.cpu().detach().item()
                disc_real_avg += discLoss_real.cpu().detach().item()
                disc_loss_lens_avg += discLoss_lens.cpu().detach().item()
                disc_fake_lens_avg += discLoss_lens_fake.cpu().detach().item()
                disc_real_lens_avg += discLoss_lens_real.cpu().detach().item()

                # Delete all discriminator stuff as its no longer needed
                discCost = discCost.detach().cpu()
                discCost_lens = discCost_lens.detach().cpu()
                discLoss = discLoss.detach().cpu()
                discLoss_lens = discLoss_lens.detach().cpu()
                discLoss_fake = discLoss_fake.detach().cpu()
                discLoss_lens_fake = discLoss_lens_fake.detach().cpu()
                discLoss_real = discLoss_real.detach().cpu()
                discLoss_lens_real = discLoss_lens_real.detach().cpu()
                gradient_penalty = gradient_penalty.cpu().detach().item()
                gradient_penalty_lens = gradient_penalty_lens.cpu().detach().item()
                del disc_sub, real_X, X_samp, real_X_samp, fake_X, masks_fake, masks_real,\
                    disc_real, disc_lens_real, disc_fake, disc_lens_fake, fake_X_samp, \
                    lens_fake, lens_real


            # Train the generator next
            self.optim_gen.zero_grad()
            for i in range(0, 1 if self.dynamic_n == False else 1):
                # Sample real data for the second loss function
                disc_sub = disc_nums[:self.batchSize//2]
                disc_nums = disc_nums[self.batchSize//2:]

                # Put the data on the correct device
                if self.dev == "partgpu":
                    disc_sub = disc_sub.to(gpu)
                    disc_nums = disc_nums.to(gpu)
                else:
                    disc_sub = disc_sub.to(self.device)
                    disc_nums = disc_nums.to(self.device)



                # Generate some data from the generator
                fake_X = self.generator.forward(training=True)
                
                # Get the real sentences as one-hot sequences
                real_X = [torch.nn.functional.one_hot(X_orig[i], len(self.vocab)) for i in disc_sub.cpu().numpy()]
                
                # Add padding to the real data
                real_X = addPadding_one_hot(real_X, self.vocab_inv, self.sequence_length)
                if self.dev == "partgpu":
                    real_X = real_X.to(gpu)
                else:
                    real_X = real_X.to(self.device)


                # Sample half a batch from the real and fake generated data
                # to generate a batch of half real and half fake sentences
                fake_X_samp = fake_X[np.random.permutation(np.arange(self.batchSize))[:self.batchSize//2]]

                # Create the batch of data to generate lengths for
                X_samp = torch.cat((fake_X_samp, real_X), dim=0)

                # Get the fake lengths from the generator
                lens_fake = self.generator.forward_lens(X_samp)

                # Get the masks for the generator
                masks_fake = self.getMask(lens_fake)
                    

                    
                
                # Send the generated output through the discriminator
                # to get a batch of predictions on the fake sentences
                disc_fake, disc_fake_lens = self.discriminator(fake_X, lens_fake, masks_fake)
                
                # Get the generator loss
                #genLoss = GLS_gen(disc_fake)
                genLoss = wasserstein_gen(disc_fake)
                genLoss_lens = wasserstein_gen(disc_fake_lens)

                if self.dynamic_n == False or (self.dynamic_n == True and r_d <= self.Lambda_n*r_g):
                    # Backpropogate the loss
                    (genLoss + genLoss_lens).backward()
                    
                    # Step the optimizer
                    self.optim_gen.step()
                self.optim_gen.zero_grad()
                genLoss = genLoss.detach()
                genLoss_lens = genLoss_lens.detach()
            
                # Delete all generator stuff as its no longer needed
                del fake_X, real_X, fake_X_samp, X_samp, lens_fake, masks_fake, \
                    disc_fake, disc_fake_lens

            
            # Dynamic n updates:
            if self.dynamic_n == True:
                if L_p_g == None:
                    L_p_g = -genLoss.item() # flip loss as the paper uses L_g = fake, we use L_d = -fake
                    L_p_d = -discLoss.item() # flip loss as the paper uses L_d = real - fake, we use L_d = fake - real
                L_c_g, L_c_d = -genLoss.item(), -discLoss.item()
                r_g, r_d = np.abs((L_c_g - L_p_g)/L_p_g), np.abs((L_c_d - L_p_d)/L_p_d)
                L_p_g, L_p_d = -genLoss.item(), -discLoss.item()
            
            
            # Flip the maximizing values to represent the actual value
            disc_real_avg *= -1
            genLoss *= -1
            disc_real_lens_avg *= -1
            genLoss_lens *= -1

            # Average the losses
            disc_loss_avg /= self.n_D
            disc_real_avg /= self.n_D
            disc_fake_avg /= self.n_D
            disc_loss_lens_avg /= self.n_D
            disc_real_lens_avg /= self.n_D
            disc_fake_lens_avg /= self.n_D
            
            
            # Save the loss values
            self.genLoss.append(genLoss.item())
            self.genLoss_lens.append(genLoss_lens.item())
            self.discLoss.append(disc_loss_avg)
            self.discLoss_real.append(disc_real_avg)
            self.discLoss_fake.append(disc_fake_avg)
            self.discLoss_lens.append(disc_loss_lens_avg)
            self.discLoss_real_lens.append(disc_real_lens_avg)
            self.discLoss_fake_lens.append(disc_fake_lens_avg)
            self.MDs.append(MD.item())
            
            print(f"Epoch: {epoch}   Generator Loss: {round(genLoss.item(), 4)}    Generator Loss L: {round(genLoss_lens.item(), 4)}\n"+\
                f"Discriminator Real: {round(disc_real_avg, 4)}     Discriminator Fake: {round(disc_fake_avg, 4)}    Discriminator Loss: {round(disc_loss_avg, 4)}    GP: {round(gradient_penalty, 4)}    MD: {round(MD.item(), 4)}\n"+\
                f"Discriminator Real L: {round(disc_real_lens_avg, 4)}     Discriminator Fake L: {round(disc_fake_lens_avg, 4)}    Discriminator Loss L: {round(disc_loss_lens_avg, 4)}    GP L: {round(gradient_penalty_lens, 4)}\n", end="")
            if self.dynamic_n:
                print(f"    r_g: {round(r_g, 4)}    r_d: {round(r_d, 4)}", end="")
            print()
    
    
    
    
    # Save the models and a training graph
    def saveModels(self, saveDir, genFile, discFile, epoch=None):
        if epoch == None:
            self.generator.saveModel(saveDir, genFile)
            self.discriminator.saveModel(saveDir, discFile)
        else:
            l = len(genFile.split(".")[-1])+1
            genFile = genFile[:-l] + f" - {epoch}.pkl"
            l = len(discFile.split(".")[-1])+1
            discFile = discFile[:-l] + f" - {epoch}.pkl"
            
            self.generator.saveModel(saveDir, genFile)
            self.discriminator.saveModel(saveDir, discFile)
            
            if self.trainGraphFile:
                fig, ax = plt.subplots()
                y = [i for i in range(1, len(self.genLoss)+1)]
                ax.plot(y, self.genLoss, label="Gen loss")
                ax.plot(y, self.discLoss_real, label="Disc loss real")
                ax.plot(y, self.discLoss_fake, label="Disc loss fake")
                ax.plot(y, self.discLoss, label="Disc loss combined")
                ax.set_title("Gen and disc loss over epochs")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Loss")
                ax.legend()
                plt.savefig(self.saveDir + os.sep + self.trainGraphFile)
                plt.close()

                fig, ax = plt.subplots()
                ax.plot(y, self.MDs, label="MD")
                ax.set_title("MD over epochs")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("MD")
                ax.legend()
                plt.savefig(self.saveDir + os.sep + "MDGraph.png")
                plt.close()

                
                fig, ax = plt.subplots()
                y = [i for i in range(1, len(self.genLoss)+1)]
                ax.plot(y, self.genLoss_lens, label="Gen loss")
                ax.plot(y, self.discLoss_real_lens, label="Disc loss real")
                ax.plot(y, self.discLoss_fake_lens, label="Disc loss fake")
                ax.plot(y, self.discLoss_lens, label="Disc loss combined")
                ax.set_title("Gen and disc loss on lengths over epochs")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Loss")
                ax.legend()
                plt.savefig(self.saveDir + os.sep + "L_" + self.trainGraphFile)
                plt.close()
    
    # Load the models
    def loadModels(self, loadDir, genFile, discFile):
        self.generator.loadModel(loadDir, genFile)
        self.discriminator.loadModel(loadDir, discFile)
