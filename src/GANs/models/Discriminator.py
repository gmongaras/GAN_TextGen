from blocks.discBlock import discBlock
from blocks.inTrans import inTrans
from torch import nn
import torch
import os
from blocks.MHAwithNorm import MHAwithNorm




class Discriminator(nn.Module):
    # Inputs:
    #   T - Number of transformer blocks in each discriminator block
    #   B - Number of discriminator blocks in the discriminator
    #   O - Number of output MHA blocks in the discrimiantor
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
    #   embed_mode - The embedding mode to be used ("fc" or "pca")
    #   device - Device to put the model on
    def __init__(self, T, B, O, outMode, batchSize, vocab_size, embedding_size, num_heads, pooling, embed_mode, device):
        super(Discriminator, self).__init__()

        self.device = device
        self.outMode = outMode.lower()
        self.embed_mode = embed_mode.lower()
        self.embedding_size = embedding_size
        
        # If the embed mode is PCA, use the PCA algorithm to transform
        # the input of shape [vocab size] to the shape [embedding_size]
        
        # If the embed mode is FC, use a FC layer to make the transformation
        if self.embed_mode != "pca":
            self.encodingTransform = nn.Linear(vocab_size, embedding_size, device=device)
        
        # Create the discriminator blocks. Note, each
        # block halves the sequence length if pooling is used
        blocks = [discBlock(T, embedding_size, num_heads, pooling) for i in range(B)]
        self.discBlocks = nn.Sequential(*blocks).to(device)
        
        # Create the class token which will be a vector of 0.5s
        self.clsTok = torch.ones(batchSize, 1, embedding_size, device=device, requires_grad=False)/2
        
        # Output MHA blocks
        self.outEmb = nn.ModuleList([MHAwithNorm(embedding_size, embedding_size, embedding_size, num_heads) for i in range(O)]).to(device)
        
        # Final feed-forward layer
        self.out_FC = nn.Linear(embedding_size, 1, device=device)
        self.Tanh = nn.Tanh().to(device)
        self.Sigmoid = nn.Sigmoid().to(device)
    
    
    
    # Input:
    #   3-D tensor of shape (N, sequence_length, vocab_size)
    # Output
    #   2-D tensor of shape (N, 1) where each value is the
    #   prediction on how real the input is between -1 and 1
    def forward(self, X):
        # Apply the encoding transformation to get the embeddings
        # to the desired embedding size
        if self.embed_mode == "pca":
            X = torch.pca_lowrank(X, self.embedding_size)[0]
        else:
            X = self.encodingTransform(X)
        
        # Send the input through the discriminator blocks
        X = self.discBlocks(X)
        
        # Add the class token to the output of the blocks
        X = torch.cat((self.clsTok, X), dim=1)
        
        # Send the output through some MHA blocks
        for O in self.outEmb:
            X = O(X, X)
        
        # Get the class token from the sequence for each
        # batch sequence
        X = X[:, 0]
        
        # Send the token through the FC layer and
        # an optional activation
        X = self.out_FC(X)
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