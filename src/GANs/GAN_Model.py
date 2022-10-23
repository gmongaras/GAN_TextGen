from ..helpers.helpers import encode_sentences
from ..helpers.helpers import addPadding_one_hot


from .models.losses import wasserstein_disc
from .models.losses import wasserstein_disc_split
from .models.losses import wasserstein_gen


from .models.Generator import Generator
from .models.Discriminator import Discriminator

import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt
import os
import json



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
    #   alpha - Learning rate of the model
    #   Lambda - Lambda value used for gradient penalty in disc loss
    #   Beta1 - Adam beta 1 term
    #   Beta2 - Adam beta 2 term
    #   device - Device to put the model on
    #   saveSteps - Number of steps until the model is saved
    #   saveDir - Directory to save the model to
    #   saveDefFile - Name of the file to save GAN defaults to so it can be easily loaded in
    #   genSaveFile - Name of the file to save the generator model to
    #   genSaveDefFile - Name of the file to save generator defaults to so it can be easily loaded in
    #   discSaveFile - Name of the file to save the discriminator model to
    #   trainGraphFile - File to save training graph during training
    #   loadInEpoch - Should the data be loaded in as needed instead of
    #                 before training (True if so, False to load before training)
    #   delWhenLoaded - Delete the data as it's loaded in to save space?
    #                   Note: This is automatically False if loadInEpoch is True
    def __init__(self, vocab, M_gen, B_gen, O_gen, T_disc, B_disc, O_disc, hiddenSize, 
        noiseDist, costSlope, batchSize, embedding_size_gen, embedding_size_disc, 
        sequence_length, num_heads, dynamic_n, Lambda_n, HideAfterEnd, n_D, pooling, 
        alpha, Lambda, Beta1, Beta2, device, saveSteps, saveDir, saveDefFile, genSaveFile, 
        genSaveDefFile, discSaveFile, trainGraphFile, loadInEpoch, delWhenLoaded):
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

        # Dictionary of important default parameters for later loading
        self.defaults = {
            "vocab": vocab,
            "M_gen": M_gen,
            "B_gen": B_gen,
            "O_gen": O_gen,
            "T_disc": T_disc,
            "B_disc": B_disc,
            "O_disc": O_disc,
            "hiddenSize": hiddenSize,
            "noiseDist": noiseDist,
            "embedding_size_gen": embedding_size_gen,
            "embedding_size_disc": embedding_size_disc,
            "sequence_length": sequence_length,
            "num_heads": num_heads,
            "HideAfterEnd": HideAfterEnd,
            "n_D": n_D,
            "pooling": pooling,
        }
        
        # Saving paramters
        self.saveSteps = saveSteps
        self.saveDir = saveDir
        self.saveDefFile = saveDefFile
        self.genSaveFile = genSaveFile
        self.genSaveDefFile = genSaveDefFile
        self.discSaveFile = discSaveFile
        self.trainGraphFile = trainGraphFile

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
        
        # The generator and discriminator models on the correct device
        if self.dev != "cpu":
            self.generator = Generator(vocab, M_gen, B_gen, O_gen, noiseDist, hiddenSize, batchSize, embedding_size_gen, sequence_length, num_heads, gpu)
            self.discriminator = Discriminator(T_disc, B_disc, O_disc, "none", hiddenSize, len(vocab), embedding_size_disc, num_heads, pooling, gpu)
        else:
            self.generator = Generator(vocab, M_gen, B_gen, O_gen, noiseDist, hiddenSize, batchSize, embedding_size_gen, sequence_length, num_heads, device)
            self.discriminator = Discriminator(T_disc, B_disc, O_disc, "none", hiddenSize, len(vocab), embedding_size_disc, num_heads, pooling, device)
        
        # The optimizer for the models
        self.optim_gen = torch.optim.Adam(self.generator.parameters(), alpha, betas=[Beta1, Beta2])
        self.optim_disc = torch.optim.Adam(self.discriminator.parameters(), alpha, betas=[Beta1, Beta2])

        # Uniform distribution for the gradient penalty
        self.unif = torch.distributions.uniform.Uniform(0, 1)
    
    
    # For WGAN-GP loss, we need a gradient penalty to add
    # to the discriminator loss. This function gets that
    # penalty
    # Inputs:
    #   x - A sample of real data
    #   x_tilde - A sample of data from the generator
    def get_gradient_penalty(self, x, x_tilde, masks_real=None, masks_fake=None):
        # Get the device
        device = self.device
        if self.dev == "partgpu":
            device = gpu

        # Sample a uniform distribution to get a random number, epsilon
        epsilon = self.unif.sample((self.batchSize, 1, 1)).to(device)
        
        # Create a new tensor fo the same shape as the real and fake data
        epsilon = epsilon.expand(x.shape)
        x_hat = epsilon*x + (1-epsilon)*x_tilde

        # Masks are the combination of the two masks
        if masks_real != None and masks_fake != None:
            masks_hat = torch.logical_and(masks_real, masks_fake).clone().detach()

        # We only want the gradients of the discriminator
        x_hat = x_hat.clone().detach()
        x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
        
        # Send the transformed and combined data through the discriminator
        if masks_real != None and masks_fake != None:
            disc_hat = self.discriminator(x_hat, masks_hat)
        else:
            disc_hat = self.discriminator(x_hat)
        
        # Get the gradients of the discriminator output on the sentences
        gradients = torch.autograd.grad(outputs=(disc_hat), inputs=(x_hat),
                              grad_outputs=(torch.ones(disc_hat.size(), device=device)),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Get the final penalty and return it
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.Lambda
    
    
    
    
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
        
        # Initial variable states for dynamic_n
        r_d = 1
        r_g = 0
        L_p_g = None
        L_p_d = None
        
        # Train the model for `epochs` number of epochs
        for epoch in range(1, epochs+1):
            # Model saving
            if epoch % self.saveSteps == 0:
                self.saveModels(self.saveDir, self.genSaveFile, self.genSaveDefFile, self.discSaveFile, epoch)
            
            # Create a list of indices which the Discriminator
            # has left to see and the Generator has left to see
            # in this epoch so that sampling doesn't resample
            disc_nums = torch.randperm(s, device=self.device)

            # Reset averages
            disc_loss_avg = 0
            disc_fake_avg = 0
            disc_real_avg = 0
            
            # Train the discriminator
            self.optim_disc.zero_grad()
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



                # Get the masks if HideAfterEnd is true
                if self.HideAfterEnd:
                    # Get the data lengths
                    lens_real = self.getLens(real_X)
                    lens_fake = self.getLens(fake_X)

                    # One hot encode the lengths
                    try:
                        lens_real = torch.nn.functional.one_hot(lens_real, self.sequence_length).float()
                    except:
                        lens_real = torch.zeros((self.batchSize, self.sequence_length))
                    try:
                        lens_fake = torch.nn.functional.one_hot(lens_fake, self.sequence_length).float()
                    except:
                        lens_fake = torch.zeros((self.batchSize, self.sequence_length))

                    # Get the masks
                    masks_real = self.getMask(lens_real)
                    masks_fake = self.getMask(lens_fake)
                
                # Send the generated output through the discriminator
                # to get a batch of predictions on the fake sentences
                disc_fake = self.discriminator(fake_X, masks_fake if self.HideAfterEnd else None)

                # Make the inputs into the discriminator loss variables
                # for differentiation
                real_X = torch.autograd.Variable(real_X, requires_grad=True)
                fake_X = torch.autograd.Variable(fake_X, requires_grad=True)

                # Get the gradient penalty
                gradient_penalty = \
                    self.get_gradient_penalty(real_X, fake_X, masks_real if self.HideAfterEnd else None, masks_fake if self.HideAfterEnd else None)

                # Calculate mean difference for debugging
                MD = torch.abs(real_X-fake_X).sum(-1).sum(-1).mean().detach().cpu()
                
                # Send the generated output through the discriminator
                # to get a batch of predictions on the real sentences
                disc_real = self.discriminator(real_X, masks_real if self.HideAfterEnd else None)
                
                # Get the discriminator loss
                discLoss = wasserstein_disc(disc_real, disc_fake)
                discLoss_fake, discLoss_real = wasserstein_disc_split(disc_real, disc_fake)

                # The discriminator cost is the loss + the penalty
                discCost = discLoss + gradient_penalty
                
                if self.dynamic_n == False or (self.dynamic_n == True and r_d > self.Lambda_n*r_g):
                    # Backpropogate the cost
                    discCost.backward()
                    
                    # Step the optimizer
                    self.optim_disc.step()
                self.optim_disc.zero_grad()

                # Loss cumulation
                disc_loss_avg += discLoss.cpu().detach().item()
                disc_fake_avg += discLoss_fake.cpu().detach().item()
                disc_real_avg += discLoss_real.cpu().detach().item()

                # Delete all discriminator stuff
                # to conserve memory
                discCost = discCost.detach().cpu()
                discLoss = discLoss.detach().cpu()
                discLoss_fake = discLoss_fake.detach().cpu()
                discLoss_real = discLoss_real.detach().cpu()
                gradient_penalty = gradient_penalty.cpu().detach().item()
                del disc_sub, real_X, fake_X, disc_real, disc_fake


            # Train the generator next
            self.optim_gen.zero_grad()
            for i in range(0, 1 if self.dynamic_n == False else 1):
                # Sample real data for the second loss function
                disc_sub = disc_nums[:self.batchSize]
                disc_nums = disc_nums[self.batchSize:]

                # Put the data on the correct device
                if self.dev == "partgpu":
                    disc_sub = disc_sub.to(gpu)
                    disc_nums = disc_nums.to(gpu)
                else:
                    disc_sub = disc_sub.to(self.device)
                    disc_nums = disc_nums.to(self.device)


                # Generate some data from the generator
                fake_X = self.generator.forward(training=True)

                # Get the masks if HideAfterEnd is true
                if self.HideAfterEnd:
                    # Get the data lengths
                    lens_fake = self.getLens(fake_X)

                    # One hot encode the lengths
                    try:
                        lens_fake = torch.nn.functional.one_hot(lens_fake, self.sequence_length).float()
                    except:
                        lens_fake = torch.zeros((self.batchSize, self.sequence_length))

                    # Get the masks
                    masks_fake = self.getMask(lens_fake)
                    

                    
                
                # Send the generated output through the discriminator
                # to get a batch of predictions on the fake sentences
                disc_fake  = self.discriminator(fake_X, masks_fake if self.HideAfterEnd else None)
                
                # Get the generator loss
                genLoss = wasserstein_gen(disc_fake)

                if self.dynamic_n == False or (self.dynamic_n == True and r_d <= self.Lambda_n*r_g):
                    # Backpropogate the loss
                    genLoss.backward()
                    
                    # Step the optimizer
                    self.optim_gen.step()
                self.optim_gen.zero_grad()
            
                # Delete all generator stuff as its no longer needed
                genLoss = genLoss.detach()
                del fake_X, disc_fake

            
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

            # Average the losses
            disc_loss_avg /= self.n_D
            disc_real_avg /= self.n_D
            disc_fake_avg /= self.n_D
            
            
            # Save the loss values
            self.genLoss.append(genLoss.item())
            self.discLoss.append(disc_loss_avg)
            self.discLoss_real.append(disc_real_avg)
            self.discLoss_fake.append(disc_fake_avg)
            self.MDs.append(MD.item())
            
            print(f"Epoch: {epoch}   Generator Loss: {round(genLoss.item(), 4)}\n"+\
                f"Discriminator Real: {round(disc_real_avg, 4)}     Discriminator Fake: {round(disc_fake_avg, 4)}    Discriminator Loss: {round(disc_loss_avg, 4)}    GP: {round(gradient_penalty, 4)}    MD: {round(MD.item(), 4)}\n", end="")
            if self.dynamic_n:
                print(f"    r_g: {round(r_g, 4)}    r_d: {round(r_d, 4)}", end="")
            print()
    
    
    
    
    # Save the models and a training graph
    def saveModels(self, saveDir, genFile, genSaveDefFile, discFile, epoch=None):
        if epoch == None:
            self.generator.saveModel(saveDir, genFile)
            self.discriminator.saveModel(saveDir, discFile)
        else:
            l = len(genFile.split(".")[-1])+1
            genFile = genFile[:-l] + f" - {epoch}.pkl"
            l = len(discFile.split(".")[-1])+1
            discFile = discFile[:-l] + f" - {epoch}.pkl"
            
            self.generator.saveModel(saveDir, genFile, genSaveDefFile)
            self.discriminator.saveModel(saveDir, discFile)
            with open(saveDir + os.sep + self.saveDefFile, "w") as f:
                json.dump(self.defaults, f)
            
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
    
    # Load the models
    def loadModels(self, loadDir, loadDefFile, genFile, discFile):
        # Load in the defaults
        with open(loadDir + os.sep + loadDefFile, "r") as f:
            self.defaults = json.load(f)
        self.vocab = self.defaults["vocab"]

        # Create new models
        if self.dev != "cpu":
            self.generator = Generator(self.vocab, self.defaults["M_gen"], self.defaults["B_gen"], self.defaults["O_gen"], self.defaults["noiseDist"], self.defaults["hiddenSize"], self.batchSize, self.defaults["embedding_size_gen"], self.defaults["sequence_length"], self.defaults["num_heads"], gpu)
            self.discriminator = Discriminator(self.defaults["T_disc"], self.defaults["B_disc"], self.defaults["O_disc"], "none", self.defaults["hiddenSize"], len(self.vocab), self.defaults["embedding_size_disc"], self.defaults["num_heads"], self.defaults["pooling"], gpu)
        else:
            self.generator = Generator(self.vocab, self.defaults["M_gen"], self.defaults["B_gen"], self.defaults["O_gen"], self.defaults["noiseDist"], self.defaults["hiddenSize"], self.batchSize, self.defaults["embedding_size_gen"], self.defaults["sequence_length"], self.defaults["num_heads"], self.device)
            self.discriminator = Discriminator(self.defaults["T_disc"], self.defaults["B_disc"], self.defaults["O_disc"], "none", self.defaults["hiddenSize"], len(self.vocab), self.defaults["embedding_size_disc"], self.defaults["num_heads"], self.defaults["pooling"], self.device)
        self.n_D = self.defaults["n_D"]
        self.HideAfterEnd = self.defaults["HideAfterEnd"]

        # Load in the models
        self.generator.loadModel(loadDir, genFile)
        self.discriminator.loadModel(loadDir, discFile)
