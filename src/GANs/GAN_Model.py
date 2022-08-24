from distutils.ccompiler import gen_lib_options
from ..helpers.helpers import encode_sentences
from ..helpers.helpers import encode_sentences_one_hot
from ..helpers.helpers import addPadding
from ..helpers.helpers import addPadding_one_hot


from .models.losses import wasserstein_disc
from .models.losses import wasserstein_disc_split
from .models.losses import wasserstein_gen
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
    #   B_gen - Number of generator blocks in the generator
    #   O_gen - Number of MHA blocks in the generator
    #   gausNoise - True to add pure gaussian noise in the generator output
    #               encoding, False to not add this noise
    #   T_disc - Number of transformer blocks in each discriminator block
    #   B_disc - Number of discriminator blocks in the discriminator
    #   O_disc - Number of output MHA blocks in the discrimiantor
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
    def __init__(self, vocab, M_gen, B_gen, O_gen, gausNoise, T_disc, B_disc, O_disc, batchSize, embedding_size_gen, embedding_size_disc, sequence_length, num_heads, dynamic_n, Lambda_n, HideAfterEnd, n_D, pooling, gen_outEnc_mode, embed_mode_gen, embed_mode_disc, alpha, Lambda, Beta1, Beta2, device, saveSteps, saveDir, genSaveFile, discSaveFile, trainGraphFile, loadInEpoch, delWhenLoaded):
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
            self.generator = Generator(vocab, M_gen, B_gen, O_gen, gausNoise, batchSize, embedding_size_gen, sequence_length, num_heads, embed_mode_gen, gen_outEnc_mode, gpu)
            self.discriminator = Discriminator(T_disc, B_disc, O_disc, "none", batchSize, len(vocab), embedding_size_disc, sequence_length, num_heads, pooling, embed_mode_disc, gpu)
        else:
            self.generator = Generator(vocab, M_gen, B_gen, O_gen, gausNoise, batchSize, embedding_size_gen, sequence_length, num_heads, embed_mode_gen, gen_outEnc_mode, device)
            self.discriminator = Discriminator(T_disc, B_disc, O_disc, "none", batchSize, len(vocab), embedding_size_disc, sequence_length, num_heads, pooling, embed_mode_disc, device)
        
        # The optimizer for the models
        self.optim_gen = torch.optim.Adam(self.generator.parameters(), alpha, betas=[Beta1, Beta2])
        self.optim_disc = torch.optim.Adam(self.discriminator.parameters(), alpha, betas=[Beta1, Beta2])
        
        
    def one_hot(a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
    
    
    # For WGAN-GP loss, we need a gradient penalty to add
    # to the discriminator loss. This function gets that
    # penalty
    # Inputs:
    #   x - A sample of real data
    #   x_tilde - A sample of data from the generator
    def get_gradient_penalty(self, x, lens_real, x_tilde, lens_fake):
        # Get the device
        device = self.device
        if self.dev == "partgpu":
            device = gpu

        # Sample a uniform distribution to get a random number, epsilon
        epsilon = torch.rand((self.batchSize, 1, 1), requires_grad=True, device=device)
        
        # Create a new tensor fo the same shape as the real and fake data
        lens_hat = epsilon.squeeze()*lens_real + (1-epsilon.squeeze())*lens_fake.squeeze()
        epsilon = epsilon.expand(x.shape)
        x_hat = epsilon*x + (1-epsilon)*x_tilde

        # We only want the gradients of the discriminator
        x_hat = x_hat.clone().detach()
        x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
        lens_hat = lens_hat.clone().detach()
        lens_hat = torch.autograd.Variable(lens_hat, requires_grad=True)
        
        # Send the transformed and combined data through the
        # discriminator
        disc_x_hat = self.discriminator(x_hat, lens_hat)
        
        # Get the gradients of the discriminator output
        gradients = torch.autograd.grad(outputs=disc_x_hat, inputs=(x_hat, lens_hat),
                              grad_outputs=torch.ones(disc_x_hat.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        
        # Get the final penalty and return it
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.Lambda
    
    
    
    
    # Given some encoded sentence data, return masks masking tokens
    # after the first <END> token in the sentence
    # Input:
    #   data - Data of the shape (N, S, V)
    # Output:
    #   Masks of the shape (N, S)
    def getMasks(self, data):
        # Position of any <END> tokens in the outut
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
        
        # Generate a tensor used to mask the MHA
        masks = torch.zeros((data.shape[0], data.shape[1]+1), dtype=torch.bool, requires_grad=False, device=self.device)
        
        # Make all parts of the tensor after the <PAD> tokens
        # True to signify that the position should not be attended to
        for i in range(0, len(end_pos)):
            if end_pos[i] == -1:
                continue
            masks[i, end_pos[i]+2:] = True
            
        # Return the output and the masks
        return masks.to(self.device if self.dev != "partgpu" else gpu), end_pos.to(self.device if self.dev != "partgpu" else gpu)
    
    
    
    # Train the models
    # Input:
    #   X - A list of sentences to train the models on
    #   epochs - Number of epochs to train the models for
    def train_model(self, X, epochs):
        # Save the sentence lengths
        lens_labels = torch.tensor([len(i.split(" ")) for i in X], dtype=torch.int8, device=self.device, requires_grad=False)

        # Encode the all the data if self.loadInEpoch is false
        if self.loadInEpoch == False:
            X_orig_one_hot = np.array(encode_sentences_one_hot(X, self.vocab_inv, self.sequence_length, self.delWhenLoaded, self.device), dtype=object)
            s = X_orig_one_hot.shape[0]
        else:
            X = np.array(X, dtype=object)
            s = X.shape[0]
        
        # Save loss values over training for the loss plot
        self.genLoss = []
        self.discLoss = []
        self.discLoss_real = []
        self.discLoss_fake = []
        
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
            for i in range(0, max(self.n_D, 1) if self.dynamic_n == False else (1 if r_d > self.Lambda_n*r_g or L_p_g == None else 0)):
                # Sample data for the discriminator
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
                with torch.no_grad():
                    Y, lens_fake = self.generator.forward_train()

                # Generate masks for the lengths
                # Masks are of shape (N, S) where each element is a 1 or 0.
                # 0 indicates no masking. 1 indicates a masking position.
                # lens_fake_argmax = torch.argmax(lens_fake, dim=-1)
                # masks = torch.zeros(Y.shape[0], Y.shape[1]+1, device=self.device, requires_grad=False)
                # for i in range(0, Y.shape[0]):
                #     # +1 since the lengths are going to be appended to the beginning of the
                #     # sequence and should not be masked
                #     masks[i, lens_fake_argmax[i].long().item()+1:] = 1
                # masks = masks.to(torch.bool)
                # if self.dev == "partgpu":
                #     masks = masks.to(gpu)
                
                # # Generate masks for the fake data
                # if self.HideAfterEnd:
                #     masks = self.getMasks(Y)
                
                # Send the generated output through the discriminator
                # to get a batch of predictions on the fake sentences
                disc_fake = torch.squeeze(self.discriminator(Y, lens_fake, masks))
                
                # Get a real data subset using one_hot encoding
                if self.loadInEpoch == True:
                    # Load in more data until no more data is availble
                    # or the requested batch size is obtained
                    real_X = np.array([])
                    while real_X.shape[0] < self.batchSize or disc_nums.shape[0] == 0:
                        # Get more data if needed
                        disc_sub = disc_nums[:self.batchSize]
                        disc_nums = disc_nums[self.batchSize:]
                        
                        # Save the data
                        if len(real_X) == 0:
                            real_X = np.array(encode_sentences_one_hot(X[disc_sub.cpu().numpy()].tolist(), self.vocab_inv, self.sequence_length, False, cpu), dtype=object)
                        else:
                            real_X = np.concatenate((real_X, np.array(encode_sentences_one_hot(X[disc_sub.cpu().numpy()].tolist(), self.vocab_inv, self.sequence_length, False, cpu), dtype=object)[:self.batchSize-real_X.shape[0]]))
                    
                    # If disc_nums is empty, a problem occured
                    assert disc_nums.shape[0] > 0, "Not enough data under requested sequence length"
                else:
                    real_X = X_orig_one_hot[disc_sub.cpu().numpy()]
                
                # Add padding to the subset
                real_X = addPadding_one_hot(real_X, self.vocab_inv, self.sequence_length)
                if self.dev == "partgpu":
                    real_X = real_X.to(gpu)
                else:
                    real_X = real_X.to(self.device)

                # Get the real data masks and lengths
                # masks, lens_real = self.getMasks(real_X)
                _, lens_real = self.getMasks(real_X)

                # One hot encode the lengths
                lens_real = torch.nn.functional.one_hot(lens_real, self.sequence_length)

                # Get the gradient penalty
                gradient_penalty = self.get_gradient_penalty(real_X, lens_real, Y, lens_fake)

                # We don't need the generated output anymore
                del Y
                
                # Send the generated output through the discriminator
                # to get a batch of predictions on the real sentences
                disc_real = torch.squeeze(self.discriminator(real_X, lens_real.float(), masks))
                
                # Get the discriminator loss
                #discLoss = minimax_disc(disc_real, disc_fake)
                discLoss = wasserstein_disc(disc_real, disc_fake)
                
                discLoss_fake, discLoss_real = wasserstein_disc_split(disc_real, disc_fake)

                # The cost of the discriminator is the loss + the penalty
                discCost = discLoss + gradient_penalty
                
                # Backpropogate the cost
                discCost.backward()
                
                # Step the optimizer
                self.optim_disc.step()
                self.optim_disc.zero_grad()

                # Delete all discriminator stuff as its no longer needed
                gradient_penalty = gradient_penalty.cpu().detach().item()
                del disc_sub, disc_fake, real_X, disc_real, lens_real, lens_fake
            
            # Train the generator next
            self.optim_gen.zero_grad()
            for i in range(0, 1 if self.dynamic_n == False else (1 if r_d <= self.Lambda_n*r_g or L_p_g == None else 0)):
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
                with torch.no_grad():
                    Y, lens_fake = self.generator.forward_train()

                # Generate masks for the lengths
                # Masks are of shape (N, S) where each element is a 1 or 0.
                # 0 indicates no masking. 1 indicates a masking position.
                # lens_fake_argmax = torch.argmax(lens_fake, dim=-1)
                # masks = torch.zeros(Y.shape[0], Y.shape[1]+1, device=self.device, requires_grad=False)
                # for i in range(0, Y.shape[0]):
                #     # +1 since the lengths are going to be appended to the beginning of the
                #     # sequence and should not be masked
                #     masks[i, lens_fake_argmax[i].long().item()+1:] = 1
                # masks = masks.to(torch.bool)
                # if self.dev == "partgpu":
                #     masks = masks.to(gpu)
                
                # Generate masks for the fake data
                if self.HideAfterEnd:
                    masks = self.getMasks(Y)
                
                # Send the generated output through the discriminator
                # to get a batch of predictions on the fake sentences
                disc_fake = torch.squeeze(self.discriminator(Y, lens_fake, masks))
                
                # Get the generator loss
                #genLoss = minimax_gen(disc_fake)
                genLoss = wasserstein_gen(disc_fake)

                
                # Backpropogate the loss
                genLoss.backward()
                
                # Step the optimizer
                self.optim_gen.step()
                self.optim_gen.zero_grad()
            
                # Delete all generator stuff as its no longer needed
                del disc_sub, disc_fake, Y, lens_fake

            
            # Dynamic n updates:
            if self.dynamic_n == True:
                if L_p_g == None:
                    L_p_g = -genLoss.item() # flip loss as the paper uses L_g = fake, we use L_d = -fake
                    L_p_d = -discLoss.item() # flip loss as the paper uses L_d = real - fake, we use L_d = fake - real
                L_c_g, L_c_d = -genLoss.item(), -discLoss.item()
                r_g, r_d = np.abs((L_c_g - L_p_g)/L_p_g), np.abs((L_c_d - L_p_d)/L_p_d)
                L_p_g, L_p_d = -genLoss.item(), -discLoss.item()
            
            
            # Flip the maximizing values to represent the actual value
            genLoss *= -1
            discLoss_real *= -1
            discLoss *= -1
            
            
            # Save the loss values
            self.genLoss.append(genLoss.item())
            self.discLoss.append(discLoss.item())
            self.discLoss_real.append(discLoss_real.item())
            self.discLoss_fake.append(discLoss_fake.item())
            
            print(f"Epoch: {epoch}   Generator Loss: {round(genLoss.item(), 4)}     Discriminator Real: {round(discLoss_real.item(), 4)}     Discriminator Fake: {round(discLoss_fake.item(), 4)}    Discriminator Loss: {round(discLoss.item(), 4)}    GP: {round(gradient_penalty, 4)}", end="")
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
                fix, ax = plt.subplots()
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
    
    # Load the models
    def loadModels(self, loadDir, genFile, discFile):
        self.generator.loadModel(loadDir, genFile)
        self.discriminator.loadModel(loadDir, discFile)
