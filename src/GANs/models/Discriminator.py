from ..blocks.discBlock import discBlock
from torch import nn
import torch
import os
from ..blocks.MHAwithNorm import MHAwithNorm
import math




class Discriminator(nn.Module):
    # Inputs:
    #   T - Number of transformer blocks in each discriminator block
    #   B - Number of discriminator blocks in the discriminator
    #   O - Number of output MHA blocks in the discrimiantor
    #   outMode - How should the output be transformed?
    #             ("none", "sigmoid", or "tanh")
    #   hiddenSize - Hidden linear size in the transformer blocks
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
    def __init__(self, T, B, O, outMode, hiddenSize, batchSize, vocab_size, embedding_size, sequence_length, num_heads, pooling, embed_mode, device):
        super(Discriminator, self).__init__()

        self.device = device
        self.outMode = outMode.lower()
        self.embed_mode = embed_mode.lower()
        self.embedding_size = embedding_size


        # Discriminator lookup table (linear without a bias)
        self.Hot2Enc = nn.Linear(vocab_size, embedding_size, bias=False, device=device)
        # self.LL1 = nn.Linear(vocab_size, vocab_size//100, device=device)
        # self.LL2 = nn.Linear(vocab_size//100, vocab_size//1000, device=device)
        # self.LL3 = nn.Linear(vocab_size//1000, embedding_size, device=device)
        
        # If the embed mode is PCA, use the PCA algorithm to transform
        # the input of shape [vocab size] to the shape [embedding_size]
        # If the embed mode is FC, use a FC layer to make the transformation
        if self.embed_mode != "pca":
            self.encodingTransform = nn.Linear(vocab_size, embedding_size, device=device)
        
        # Create the discriminator backbone. Note, each
        # block halves the sequence length if pooling is used
        # (NxS+1xE) -> (NxLxE)
        blocks = [discBlock(T, embedding_size, embedding_size, num_heads, hiddenSize, pooling) for i in range(B)]
        self.disc_backbone = nn.Sequential(*blocks).to(device)

        # Create the discriminator backbone for the lengths. Note, each
        # block halves the sequence length if pooling is used
        # (NxSx1) -> (NxLxE)
        Es = torch.linspace(1, embedding_size, B+1).int().numpy()
        blocks = [discBlock(T, Es[i], Es[i+1], num_heads, hiddenSize, pooling) for i in range(B)]
        self.disc_backbone2 = nn.Sequential(*blocks).to(device)
        
        # The discriminator classifier head
        # (NxL+1xE) -> (N)
        self.disc_head_B = nn.Sequential(*[
            discBlock(T, embedding_size, embedding_size, num_heads, hiddenSize, "none") for i in range(0, O)
        ]).to(device)
        self.disc_head_L = nn.Linear(embedding_size, 1, device=device)
        
        # Create the class token which will be a parameter
        # initialized to random values
        self.clsTok = nn.Parameter(torch.rand(batchSize, 1, embedding_size, device=device, requires_grad=True))
        
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
            pass
            #X = self.encodingTransform(X)

        # Encode the words
        X = self.Hot2Enc(X)

        # X = self.LL1(X)
        # X = self.LL2(X)
        # X = self.LL3(X)

        # # Get the lengths as numerical values
        # lens_num = torch.argmax(lens, dim=-1).cpu().detach()

        # # Replace the <PAD> tokens with 0 so that the
        # # discriminator doesn't worry about encoding it
        # for i in range(0, X.shape[0]):
        #     x = X[i]
        #     x[lens_num[i].item()+1:] = 0
        #     X[i] = x

        # # Encode the lengths from shape S to E
        # lens = self.lensEnc(lens)

        # # Append the lengths to the beginning of the inputs
        # X = torch.cat((lens.unsqueeze(1), X), dim=1)

        # Append the langths to the end of the inputs
        # X = torch.cat((X, lens.unsqueeze(-1)), dim=-1)
        
        # Send the inputs through the backbone
        if masks != None:
            X = self.disc_backbone[0](X, masks) # Only mask the first values
            for b in self.disc_backbone[1:]:
                X = b(X)
        else:
            X = self.disc_backbone(X)

        # Send the lengths through the backbone
        # Shapes: (NxSx1) -> (NxSxE)
        lens = self.disc_backbone2(lens.unsqueeze(-1))

        # Combine the lengths and the input
        X = X+lens
        
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