from ..blocks.discBlock import discBlock
from torch import nn
import torch
import os
from ..blocks.MHAwithNorm import MHAwithNorm




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
    def __init__(self, T, B, O, outMode, batchSize, vocab_size, embedding_size, sequence_length, num_heads, pooling, embed_mode, device):
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
        
        # Create the discriminator backbone. Note, each
        # block halves the sequence length if pooling is used
        # (NxS+1xE) -> (NxLxE)
        blocks = [discBlock(T, embedding_size, num_heads, pooling) for i in range(B)]
        self.disc_backbone = nn.Sequential(*blocks).to(device)
        
        # The discriminator classifier head
        # (NxL+1xE) -> (N)
        self.disc_head_B = nn.Sequential(*[
            discBlock(T, embedding_size, num_heads, "none") for i in range(0, O)
        ]).to(device)
        self.disc_head_L = nn.Linear(embedding_size, 1, device=device)
        
        # Create the class token which will be a tensor of 1s
        self.clsTok = torch.ones(batchSize, 1, embedding_size, device=device, requires_grad=False)
        
        # Final feed-forward layer
        self.out_FC = nn.Linear(embedding_size, 1, device=device)
        self.Tanh = nn.Tanh().to(device)
        self.Sigmoid = nn.Sigmoid().to(device)

        # Linear layer used to encode the lengths
        self.lensEnc = nn.Linear(sequence_length, embedding_size, device=device)
    
    
    
    # Input:
    #   X - 3-D tensor of shape (N, sequence_length, vocab_size)
    #   masks - An optional tensor of shape (N, S) used
    #           to mask the tokens after the first <END> token
    #           in each sequence
    # Output
    #   2-D tensor of shape (N, 1) where each value is the
    #   prediction on how real the input is between -1 and 1
    def forward(self, X, lens, masks=None):
        # Apply the encoding transformation to get the embeddings
        # to the desired embedding size
        if self.embed_mode == "pca":
            X = torch.pca_lowrank(X, self.embedding_size)[0]
        else:
            X = self.encodingTransform(X)

        # Encode the lengths from shape S to E
        lens = self.lensEnc(lens)

        # Append the lengths to the beginning of the inputs
        X = torch.cat((lens.unsqueeze(1), X), dim=1)
        
        # Send the input through the backbone
        if masks != None:
            X = self.disc_backbone[0](X, masks)
            for b in self.disc_backbone[1:]:
                X = b(X)
        else:
            X = self.disc_backbone(X)
        
        # Add the class token to the output of the backbone
        X = torch.cat((self.clsTok[:X.shape[0]], X), dim=1)
        
        # Get the predictions
        X = self.disc_head_B(X)[:, 0]
        X = self.disc_head_L(X)
        
        # Apply the optional activation
        if self.outMode == "sigmoid":
            X = self.Sigmoid(X)
        elif self.outMode == "tanh":
            X = self.Tanh(X)
        
        return X.squeeze()
    
    
    
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