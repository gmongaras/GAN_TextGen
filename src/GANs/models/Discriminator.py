from ..blocks.discBlock import discBlock
from torch import nn
import torch
import os
from ..blocks.PositionalEncoding import PositionalEncoding




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
    #   num_heads - Number of heads to use in the MHA block
    #   pooling - What pooling mode should be used? ("avg", "max", or "none")
    #   device - Device to put the model on
    def __init__(self, T, B, O, outMode, hiddenSize, vocab_size, embedding_size, num_heads, pooling, device):
        super(Discriminator, self).__init__()

        # Saved parameters
        self.device = device
        self.outMode = outMode.lower()
        self.embedding_size = embedding_size

        # Discriminator lookup table (linear without a bias)
        # (N, S, V) -> (N, S, E)
        self.sent_lookup = nn.Linear(vocab_size, embedding_size, bias=False, device=device)
        
        # Create the discriminator backbone. Note, each
        # block halves the sequence length if pooling is used
        # (N, S, E) -> (N, S, E)
        blocks = [discBlock(T, embedding_size*2, embedding_size*2, num_heads, hiddenSize, pooling) for i in range(B)]
        self.disc_backbone = nn.Sequential(*blocks).to(device)
        
        # The discriminator classifier head which uses a class
        # token to classify a sentence as real or fake
        # (NxS+1xE) -> (N)
        self.disc_head = nn.Sequential(*[
            discBlock(T, embedding_size*2, embedding_size*2, num_heads, hiddenSize, "none") for i in range(0, O)
        ]).to(device)
        self.disc_head_L = nn.Linear(embedding_size*2, 1, device=device)
        
        # Create the class token which will be a parameter
        # initialized to random values
        self.clsTok = nn.Parameter(torch.rand(1, 1, embedding_size*2, device=device, requires_grad=True))
        
        # Optional output activations
        self.Tanh = nn.Tanh().to(device)
        self.Sigmoid = nn.Sigmoid().to(device)

        # Positional encodings for a sequence of shape (?, S, E)
        self.PositionalEncoding = PositionalEncoding(embedding_size, 0).to(device)
    
    
    
    # Input:
    #   X - 3-D tensor of shape (N, S, V)
    #   masks - An optional tensor of shape (N, S) used
    #           to mask the tokens after the first <END> token
    #           in each sequence
    # Output
    #   2-D tensor of shape (N, 1) where each value is the
    #     prediction on how real the input is
    def forward(self, X, masks=None):
        # Convert the sentences to the proper encoding size
        # (N, S, V) -> (N, S, E)
        X = self.sent_lookup(X)

        # Add positional encodings to the sentence
        # (N, S, E) -> (N, S, 2E)
        X = torch.cat((X, self.PositionalEncoding(torch.zeros(X.shape).to(self.device))), dim=-1)

        # Send the sentences through the backbone
        # (N, S, E) -> (N, S, E)
        if masks != None:
            # Mask all blocks
            for b in self.disc_backbone:
                X = b(X, masks)
        else:
            X = self.disc_backbone(X)

        # Add the class token to the output of the backbone output
        # (N, S, E) -> (N, S+1, E)
        X = torch.cat((self.clsTok.repeat(X.shape[0], 1, 1), X), dim=1)

        # Real/Fake predictions
        # (N, S+1, E) -> (N, 1, E)
        if masks != None:
            # Add a tensor of 0s so the cls token isn't masked
            masks = torch.cat((torch.zeros((masks.shape[0], masks.shape[1], 1), device=X.device), masks), dim=-1)
            
            # Extend the masks by 1
            masks = torch.cat((masks, masks[:, 0].unsqueeze(1)), dim=1)

            # Send the data through the model with masks
            for b in self.disc_head:
                X = b(X, masks)
            X = X[:, 0]
        else:
            X = self.disc_head(X)[:, 0]

        # Sent the class token through a linear layer to get
        # the final prediction from the model
        # (N, 1, E) -> (N, 1, 1) -> (N)
        X = self.disc_head_L(X).squeeze()

        
        # Apply the optional activation
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