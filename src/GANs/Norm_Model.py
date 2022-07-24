from helpers.helpers import encode_sentences
from helpers.helpers import encode_sentences_one_hot
from helpers.helpers import addPadding
from helpers.helpers import addPadding_one_hot


from models.losses import binary_cross_entropy_loss


from models.Generator import Generator
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



class Norm_Model(nn.Module):
    # Inputs:
    #   vocab - A dictionary of vocab where the keys are integers and the
    #           values are words
    #   M_gen - Number of noise encoding blocks in the generator
    #   B_gen - Number of generator blocks in the generator
    #   O_gen - Number of MHA blocks in the generator
    #   gausNoise - True to add pure gaussian noise in the generator output
    #               encoding, False to not add this noise
    #   batchSize - Size to bach data into
    #   embedding_size_gen - Embedding size of the generator
    #   sequence_length - The max length of the sentence to train the model on
    #   num_heads - Number of heads for the MHA modules
    #   gen_outEnc_mode - How should the generator encode its output? ("norm" or "gumb")
    #   embed_mode_gen - What embedding mode should be used for the
    #                    generator? ("norm" or "custom")
    #   alpha - Learning rate of the model
    #   Lambda - Lambda value used for gradient penalty in disc loss
    #   Beta1 - Adam beta 1 term
    #   Beta2 - Adam beta 2 term
    #   device - Device to put the model on
    #   saveSteps - Number of steps until the model is saved
    #   saveDir - Directory to save the model to
    #   genSaveFile - Name of the file to save the generator model to
    #   trainGraphFile - File to save training graph during training
    #   loadInEpoch - Should the data be loaded in as needed instead of
    #                 before training (True if so, False to load before training)
    #   delWhenLoaded - Delete the data as it's loaded in to save space?
    #                   Note: This is automatically False if loadInEpoch is True
    def __init__(self, vocab, M_gen, B_gen, O_gen, gausNoise, batchSize, embedding_size_gen, sequence_length, num_heads, gen_outEnc_mode, embed_mode_gen, alpha, Lambda, Beta1, Beta2, device, saveSteps, saveDir, genSaveFile, trainGraphFile, loadInEpoch, delWhenLoaded):
        super(Norm_Model, self).__init__()
        
        # Save the needed variables
        self.vocab = vocab
        self.vocab_inv = {vocab[i]:i for i in vocab.keys()}
        self.sequence_length = sequence_length
        self.batchSize = batchSize
        self.Lambda = Lambda
        self.loadInEpoch = loadInEpoch
        self.delWhenLoaded = delWhenLoaded if self.loadInEpoch == False else False
        
        # Saving paramters
        self.saveSteps = saveSteps
        self.saveDir = saveDir
        self.genSaveFile = genSaveFile
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
        
        # The generator model
        if self.dev != "cpu":
            self.generator = Generator(vocab, M_gen, B_gen, O_gen, gausNoise, batchSize, embedding_size_gen, sequence_length, num_heads, embed_mode_gen, gen_outEnc_mode, gpu)
        else:
            self.generator = Generator(vocab, M_gen, B_gen, O_gen, gausNoise, batchSize, embedding_size_gen, sequence_length, num_heads, embed_mode_gen, gen_outEnc_mode, device)
        
        # The optimizer for the model
        self.optim = torch.optim.Adam(self.generator.parameters(), alpha, betas=[Beta1, Beta2])
        
        
    def one_hot(a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
    
    
    
    # Train the models
    # Input:
    #   Y - A list of sentences to train the models on
    #   epochs - Number of epochs to train the models for
    def train_model(self, Y, epochs):
        # Encode the all the data if self.loadInEpoch is false
        if self.loadInEpoch == False:
            Y_orig_one_hot = np.array(encode_sentences_one_hot(Y, self.vocab_inv, self.sequence_length, self.delWhenLoaded, self.device), dtype=object)
            s = Y_orig_one_hot.shape[0]
        else:
            Y = np.array(Y, dtype=object)
            s = Y.shape[0]
        
        # Save loss values over training for the loss plot
        self.losses = []
        
        # Train the model for epochs number of epochs
        for epoch in range(1, epochs+1):
            # Model saving
            if epoch % self.saveSteps == 0:
                self.saveModels(self.saveDir, self.genSaveFile, epoch)
            
            # Create a list of indices which the Discriminator
            # has left to see and the Generator has left to see
            real_nums = torch.randperm(s, device=self.device)
            
            self.optim.zero_grad()
            
            # Sample data for the discriminator
            real_sub = real_nums[:self.batchSize]
            real_nums = real_nums[self.batchSize:]

            # Put the data on the correct device
            if self.dev == "partgpu":
                real_sub = real_sub.to(gpu)
                real_nums = real_nums.to(gpu)
            else:
                real_sub = real_sub.to(self.device)
                real_nums = real_nums.to(self.device)
            
            # Generate some data from the generator
            Y_fake = self.generator.forward_train()
            
            # Get a real data subset using one_hot encoding
            if self.loadInEpoch == True:
                # Load in more data until no more data is availble
                # or the requested batch size is obtained
                Y = np.array([])
                while Y.shape[0] < self.batchSize or real_nums.shape[0] == 0:
                    # Get more data if needed
                    real_sub = real_nums[:self.batchSize]
                    real_nums = real_nums[self.batchSize:]
                    
                    # Save the data
                    if len(Y) == 0:
                        Y = np.array(encode_sentences_one_hot(Y[real_sub.cpu().numpy()].tolist(), self.vocab_inv, self.sequence_length, False, cpu), dtype=object)
                    else:
                        Y = np.concatenate((Y, np.array(encode_sentences_one_hot(Y[real_sub.cpu().numpy()].tolist(), self.vocab_inv, self.sequence_length, False, cpu), dtype=object)[:self.batchSize-Y.shape[0]]))
                
                # If real_nums is empty, a problem occured
                assert real_nums.shape[0] > 0, "Not enough data under requested sequence length"
            else:
                Y = Y_orig_one_hot[real_sub.cpu().numpy()]
            
            # Add padding to the subset
            Y = addPadding_one_hot(Y, self.vocab_inv, self.sequence_length)
            if self.dev == "partgpu":
                Y = Y.to(gpu)
            else:
                Y = Y.to(self.device)
            
            # Get the loss
            loss = binary_cross_entropy_loss(Y_fake, Y)
            
            # Backpropogate the loss
            loss.backward()
            
            # Step the optimizer
            self.optim.step()
            self.optim.zero_grad()
            
            
            # Save the loss values
            self.losses.append(loss.item())
            
            print(f"Epoch: {epoch}   Loss: {loss.item()}\n")
    
    
    
    
    # Save the models and a training graph
    def saveModels(self, saveDir, genFile, epoch=None):
        if epoch == None:
            self.generator.saveModel(saveDir, genFile)
        else:
            l = len(genFile.split(".")[-1])+1
            genFile = genFile[:-l] + f" - {epoch}.pkl"
            
            self.generator.saveModel(saveDir, genFile)
            
            if self.trainGraphFile:
                fix, ax = plt.subplots()
                y = [i for i in range(1, len(self.losses)+1)]
                ax.plot(y, self.losses, label="Gen loss")
                ax.set_title("Gen over epochs")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Loss")
                ax.legend()
                plt.savefig(self.saveDir + os.sep + self.trainGraphFile)
                plt.close()
    
    # Load the models
    def loadModels(self, loadDir, genFile, discFile=None):
        self.generator.loadModel(loadDir, genFile)
