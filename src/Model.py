from helpers import encode_sentences
from helpers import encode_sentences_one_hot
from helpers import addPadding
from helpers import addPadding_one_hot


from losses import wasserstein_disc
from losses import wasserstein_gen
from losses import minimax_disc
from losses import minimax_gen
from losses import minimax_loss


from Generator import Generator
from Discriminator import Discriminator
import torch
from torch import nn
import numpy as np





class Model(nn.Module):
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
    #   alpha - Learning rate of the model
    #   device - Device to put the model on
    #   saveSteps - Number of steps until the model is saved
    #   saveDir - Directory to save the model to
    #   genSaveFile - Name of the file to save the generator model to
    #   discSaveFile - Name of the file to save the discriminator model to
    def __init__(self, vocab, M_gen, N_gen, N_disc, batchSize, embedding_size, sequence_length, num_heads, trainingRatio, decRatRate, alpha, device, saveSteps, saveDir, genSaveFile, discSaveFile):
        super(Model, self).__init__()
        
        # The ratio must not have a lower value for the discriminator (1)
        assert trainingRatio[0]<=trainingRatio[1], "The training ratio must have a grater number in the zeroth index"
        
        # Save the needed variables
        self.vocab = vocab
        self.vocab_inv = {vocab[i]:i for i in vocab.keys()}
        self.sequence_length = sequence_length
        self.batchSize = batchSize
        self.trainingRatio = trainingRatio
        self.decRatRate = decRatRate
        
        # Saving paramters
        self.saveSteps = saveSteps
        self.saveDir = saveDir
        self.genSaveFile = genSaveFile
        self.discSaveFile = discSaveFile
        
        # The generator and discriminator models
        self.generator = Generator(vocab, M_gen, N_gen, batchSize, embedding_size, sequence_length, num_heads, device)
        self.discriminator = Discriminator(N_disc, batchSize, len(vocab), embedding_size, sequence_length, num_heads)
        
        # The optimizer for the model
        self.optim_gen = torch.optim.RMSprop(self.generator.parameters(), alpha)
        self.optim_disc = torch.optim.RMSprop(self.discriminator.parameters(), alpha)
        
        
    def one_hot(a, num_classes):
      return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
    
    
    
    # Train the models
    # Input:
    #   X - A list of sentences to train the models on
    #   epochs - Number of epochs to train the models for
    def train_model(self, X, epochs):
        # Encode the sentences
        X_orig = np.array(encode_sentences(X, self.vocab_inv, self.sequence_length, self.generator.Word2Vec), dtype=object)
        X_orig_one_hot = np.array(encode_sentences_one_hot(X, self.vocab_inv, self.sequence_length), dtype=object)
        
        # Train the model for epochs number of epochs
        for epoch in range(1, epochs+1):
            # Model saving
            if epoch % self.saveSteps == 0:
                self.saveModels(self.saveDir, self.genSaveFile, self.discSaveFile)
            
            # Create a list of indices which the Discriminator
            # has left to see and the Generator has left to see
            disc_nums = torch.randperm(len(X_orig))
            gen_nums = torch.randperm(len(X_orig))
            
            # Train the discriminator first
            for i in range(0, max(self.trainingRatio[1], 1)):
                # Sample the data to get data the generator and
                # discriminator will see
                disc_sub = disc_nums[:self.batchSize]
                disc_nums = disc_nums[self.batchSize:]
                gen_sub = gen_nums[:self.batchSize]
                gen_nums = gen_nums[self.batchSize:]
                
                # Generate some data from the generator
                Y = self.generator.forward_train()
                
                # Send the generated output through the discriminator
                # to get a batch of predictions on the fake sentences
                disc_fake = torch.squeeze(self.discriminator(Y)) # Predictions
                
                # Get a real data subset using one_hot encoding
                real_X = X_orig_one_hot[disc_sub.detach().numpy()]
                
                # Add padding to the subset
                real_X = addPadding_one_hot(real_X, self.vocab_inv, self.sequence_length)
                
                # Send the real output throuh the discriminator to
                # get a batch of predictions on the real sentences
                disc_real = torch.squeeze(self.discriminator(real_X)) # Predictions
                
                # Get the discriminator loss. Negative to
                # maximize the loss
                #discLoss = -minimax_loss(disc_real, disc_fake)
                
                # Get the discriminator loss
                #discLoss = minimax_disc(disc_real, disc_fake)
                discLoss = wasserstein_disc(disc_real, disc_fake)
                
                # Backpropogate the loss
                discLoss.backward()
                
                # Clip the loss
                torch.nn.utils.clip_grad_value_(self.parameters(), 0.01)
                
                # Step the optimizer
                self.optim_disc.step()
                self.optim_disc.zero_grad()
            
            # Train the generator next
            for i in range(0, max(self.trainingRatio[0], 1)):
                # Get subset indices of the data for the generator
                # and discriminator
                disc_sub = disc_nums[:self.batchSize]
                disc_nums = disc_nums[self.batchSize:]
                gen_sub = gen_nums[:self.batchSize]
                gen_nums = gen_nums[self.batchSize:]
                
                # Generate some data from the generator
                Y = self.generator.forward_train()
                
                # Send the generated output through the discriminator
                # to get a batch of predictions on the fake sentences
                disc_fake = torch.squeeze(self.discriminator(Y)) # Predictions
                
                # Get a real data subset using one_hot encoding
                #real_X = X_orig_one_hot[disc_sub.detach().numpy()]
                
                # Add padding to the subset
                #real_X = addPadding_one_hot(real_X, self.vocab_inv, self.sequence_length)
                
                # Send the real output throuh the discriminator to
                # get a batch of predictions on the real sentences
                #disc_real = torch.squeeze(self.discriminator(real_X)) # Predictions
                
                # Get the generator loss. Positive to
                # minimize the loss
                #genLoss = minimax_loss(disc_real, disc_fake)
                
                # Get the generator loss
                genLoss = minimax_gen(disc_fake)
                
                # Backpropogate the loss
                genLoss.backward()
                
                # Step the optimizer
                self.optim_gen.step()
                self.optim_gen.zero_grad()
            
            
            # Decrease the rate
            if epochs%self.decRatRate == 0 and self.decRatRate > 0:
                self.trainingRatio[0] -= 1
                self.trainingRatio[1] -= 1
            
            
            print(f"Epoch: {epoch}   Generator Loss: {round(genLoss.item(), 2)}     Discriminator Loss: {round(discLoss.item(), 2)}\n")
            
            # Iterate until the number of items in the list
            # is lower than the batch num
            # while disc_nums.shape[0] >= self.batchSize and gen_nums.shape[0] >= self.batchSize:
            #     # Get subset indices of the data for the generator
            #     # and discriminator
            #     disc_sub = disc_nums[:self.batchSize]
            #     disc_nums = disc_nums[self.batchSize:]
            #     gen_sub = gen_nums[:self.batchSize]
            #     gen_nums = gen_nums[self.batchSize:]
                
            #     # Generate some data from the generator
            #     Y = self.generator.forward_train()
                
            #     # Send the generated output through the discriminator
            #     # to get a batch of predictions on the fake sentences
            #     y_pred_gen = torch.squeeze(self.discriminator(Y)) # Predictions
            #     y_true_gen = torch.ones((self.batchSize)) # Labels
                
            #     # Get a real data subset using one_hot encoding
            #     real_X = X_orig_one_hot[disc_sub.detach().numpy()]
                
            #     # Add padding to the subset
            #     real_X = addPadding_one_hot(real_X, self.vocab_inv, self.sequence_length)
                
            #     # Send the real output throuh the discriminator to
            #     # get a batch of predictions on the real sentences
            #     y_pred_real = torch.squeeze(self.discriminator(real_X)) # Predictions
            #     y_true_real = torch.negative(torch.ones((self.batchSize))) # Labels
                
            #     # Combine the predictions and true labels
            #     y_pred = torch.cat((y_pred_gen, y_pred_real))
            #     y_true = torch.cat((y_true_gen, y_true_real))
                
            #     # Get the final loss for this batch
            #     genLoss = minimax_gen(y_pred_gen)
            #     discLoss = minimax_disc(y_true, y_pred)
            #     loss = genLoss + discLoss
            #     #loss = wasserstein_loss(y_true, y_pred)
                
            #     # Backpropogate the loss
            #     loss.backward()
                
            #     # Step the optimizer
            #     self.optim.step()
            #     self.optim.zero_grad()
                
            #     print(loss.item())
    
    
    
    
    # Save the models
    def saveModels(self, saveDir, genFile, discFile):
        self.generator.saveModel(saveDir, genFile)
        self.discriminator.saveModel(saveDir, discFile)
    
    # Load the models
    def loadModels(self, loadDir, genFile, discFile):
        self.generator.loadModel(loadDir, genFile)
        self.discriminator.loadModel(loadDir, discFile)