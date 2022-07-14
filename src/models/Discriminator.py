from blocks.discBlock import discBlock
from blocks.inTrans import inTrans
from torch import nn
import torch
import os




class Discriminator(nn.Module):
    # Inputs:
    #   N - The number of discriminator blocks to use
    #   outMode - How should the output be transformed?
    #             ("none", "sigmoid", or "tanh")
    #   batchSize - Batch size of the input sequence
    #   vocab_size - The size of the vocab used by the generator.
    #                Note: This value is the embedding size
    #                of the input tensor.
    #   embedding_size - Embedding size for each word in the input
    #                    sequence sentence.
    #   sequence_length - Number of words in the input sequence
    #   num_heads - Number of heads to use in the MHA block
    #   pooling - What pooling mode should be used? ("avg", "max", or "none")
    #   device - Device to put the model on
    def __init__(self, N, outMode, batchSize, vocab_size, embedding_size, sequence_length, num_heads, pooling, device):
        super(Discriminator, self).__init__()

        self.device = device
        self.outMode = outMode.lower()
        
        # Input linear layers to turn the input embedding size
        # (the vocab size) into the desired embedding size
        self.LL1 = nn.Linear(vocab_size, vocab_size//10, device=device)
        self.LL2 = nn.Linear(vocab_size//10, vocab_size//100, device=device)
        self.LL3 = nn.Linear(vocab_size//100, vocab_size//500, device=device)
        self.LL4 = nn.Linear(vocab_size//500, embedding_size, device=device)
        
        # Create the discriminator blocks. Note, each
        # block halves the sequence length
        if pooling == "avg" or pooling == "max":
            blocks = [discBlock(embedding_size, sequence_length//(2**i), num_heads, pooling) for i in range(N)]
        else:
            blocks = [discBlock(embedding_size, sequence_length, num_heads, pooling) for i in range(N)]
        self.discBlocks = nn.Sequential(*blocks).to(device)
        
        # Create the class token which will be a vector of 1s
        self.clsTok = torch.ones(batchSize, 1, embedding_size, device=device)
        
        # Transformer blocks
        self.trans1 = inTrans(embedding_size, num_heads, embedding_size).to(device)
        self.trans2 = inTrans(embedding_size, num_heads, embedding_size).to(device)
        
        # Final feed-forward layer
        self.out_FF = nn.Linear(embedding_size, 1, device=device)
        self.Tanh = nn.Tanh().to(device)
        self.Sigmoid = nn.Sigmoid().to(device)
    
    
    
    # Input:
    #   3-D tensor of shape (N, sequence_length, vocab_size)
    # Output
    #   2-D tensor of shape (N, 1) where each value is the
    #   prediction on how real the input is between -1 and 1
    def forward(self, X):
        # Apply the linear transformation to get the embeddings
        # to the desired embedding size
        X = self.LL1(X)
        X = self.LL2(X)
        X = self.LL3(X)
        X = self.LL4(X)
        
        # Send the input through the discriminator blocks
        X = self.discBlocks(X)
        
        # Add the class token to the output of the blocks
        X = torch.cat((self.clsTok, X), dim=1)
        
        # Send the output through some transformer blocks
        X = self.trans1(X) + X
        X = self.trans2(X)
        
        # Get the class token from the sequence for each
        # batch sequence
        X = X[:, 0]
        
        # Send the token through a feed-forward network layer
        X = self.out_FF(X)
        if self.outMode == "sigmoid":
            X = self.Sigmoid(X)
        elif self.outMode == "tanh":
            X = self.Tanh(X)
        
        return X
    
    
    
    # Save the model
    def saveModel(self, saveDir, saveFile):
        # Check if the directory exists. If it doesn't
        # create it
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)
        
        # Save the model
        torch.save(self.state_dict(), saveDir + os.sep + saveFile)
    
    
    # Load the model
    def loadModel(self, loadDir, loadFile):
        self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device))