from ..blocks.discBlock import discBlock
from torch import nn
import torch
import os
from ..blocks.outTrans import outTrans




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
        self.sent_lookup = nn.Linear(vocab_size, embedding_size, bias=False, device=device)
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
        # (NxSxE) -> (NxLxE)
        blocks = [discBlock(T, embedding_size, embedding_size, num_heads, hiddenSize, pooling) for i in range(B)]
        self.disc_backbone_S = nn.Sequential(*blocks).to(device)

        # Create the discriminator backbone for the lengths. Note, each
        # block halves the sequence length if pooling is used
        # (NxSx1) -> (NxLxE)
        Es = torch.linspace(1, embedding_size, B+1).int().numpy()
        blocks = [discBlock(T, embedding_size, embedding_size, num_heads, hiddenSize, pooling) for i in range(B)]
        self.disc_backbone_L = nn.Sequential(*blocks).to(device)
        # blocks = [discBlock(T, embedding_size, embedding_size, num_heads, hiddenSize, pooling) for i in range(max(B//2, 1))]
        # self.disc_backbone_L_S = nn.Sequential(*blocks).to(device)
        
        # The discriminator classifier head for sentences
        # (NxL+1xE) -> (N)
        self.disc_head_B_S = nn.Sequential(*[
            discBlock(T, embedding_size, embedding_size, num_heads, hiddenSize, "none") for i in range(0, O)
        ]).to(device)
        self.disc_head_L_S = nn.Linear(embedding_size, 1, device=device)

        # The discriminator classifier head for lengths
        # (NxL+1xE) -> (N)
        self.disc_head_B_L = nn.Sequential(*[
            outTrans(embedding_size, embedding_size, None, num_heads, device, hiddenSize) for i in range(0, O)
        ]).to(device)
        self.disc_head_L_L = nn.Linear(embedding_size, 1, device=device)
        
        # Create the class token which will be a parameter
        # initialized to random values
        self.clsTok_S = nn.Parameter(torch.rand(1, 1, embedding_size, device=device, requires_grad=True))
        self.clsTok_L = nn.Parameter(torch.rand(1, 1, embedding_size, device=device, requires_grad=True))
        
        # Final feed-forward layer
        self.Tanh = nn.Tanh().to(device)
        self.Sigmoid = nn.Sigmoid().to(device)

        # Lookup table for the length, just like for the vocab
        self.sent_lookup_L = nn.Linear(vocab_size, embedding_size, bias=False, device=device)
        self.lens_lookup = nn.Linear(embedding_size, embedding_size, bias=False, device=device)
    
    
    
    # Input:
    #   X - 3-D tensor of shape (N, sequence_length, vocab_size)
    #   masks - An optional tensor of shape (N, S) used
    #           to mask the tokens after the first <END> token
    #           in each sequence
    # Output
    #   2-D tensor of shape (N, 1) where each value is the
    #   prediction on how real the input is between -1 and 1
    def forward(self, X, lens, masks=None):
        masks = None

        # Convert the sentences and lengths to the proper encoding size
        # (N, S, ~) -> (N, S, E)
        # X_len = self.sent_lookup_L(X.clone().detach())
        X = self.sent_lookup(X)
        lens = lens.unsqueeze(-1).repeat(1, 1, self.embedding_size)
        lens = self.lens_lookup(lens)

        # Append the sentences to the lengths so the
        # lengths have sentence context
        # lens = torch.cat((X_len, lens.unsqueeze(-1)), dim=-1)

        # Encode the sentences for the lengths
        # X_len = self.disc_backbone_L_S(X_len)

        # Send the sentences through the backbone
        if masks != None:
            # Mask all blocks
            for b in self.disc_backbone_S:
                X = b(X, masks)
        else:
            X = self.disc_backbone_S(X)

        # Send the lengths through the backbone
        # Shapes: (NxSx1) -> (Nx2SxE)
        lens = self.disc_backbone_L(lens)

        # Add the lengths and sentence encodings together
        intermediate_X = X.clone().detach()
        intermediate_X = torch.cat((self.clsTok_L.repeat(intermediate_X.shape[0], 1, 1), intermediate_X), dim=1)
        # lens = (lens / torch.norm(lens, dim=-1)) + (X_len / torch.norm(X_len, dim=-1))

        # Cut off the sentence part of the lengths
        #lens = lens[:, X.shape[1]:]

        # Add the class token to the output of the backbone output
        X = torch.cat((self.clsTok_S.repeat(X.shape[0], 1, 1), X), dim=1)
        lens = torch.cat((self.clsTok_L.repeat(lens.shape[0], 1, 1), lens), dim=1)


        # Sentence predictions
        if masks != None:
            # Add a tensor of 0s so the cls token isn't masked
            masks = torch.cat((torch.zeros((masks.shape[0], masks.shape[1], 1), device=X.device), masks), dim=-1)
            
            # Extend the masks by 1
            masks = torch.cat((masks, masks[:, 0].unsqueeze(1)), dim=1)

            # Send the data through the model with masks
            for b in self.disc_head_B_S:
                X = b(X, masks)
            X = X[:, 0]
        else:
            X = self.disc_head_B_S(X)[:, 0]
        X = self.disc_head_L_S(X)



        # Lenth predictions
        if masks != None:
            # Add a tensor of 0s so the cls token isn't masked
            masks = torch.cat((torch.zeros((masks.shape[0], masks.shape[1], 1), device=X.device), masks), dim=-1)
            
            # Extend the masks by 1
            masks = torch.cat((masks, masks[:, 0].unsqueeze(1)), dim=1)

            # Send the data through the model with masks
            for b in self.disc_head_B_L:
                lens = b(lens, masks)
            lens = lens[:, 0]
        else:
            for b in self.disc_head_B_L:
                lens = b(lens, intermediate_X)
            lens = lens[:, 0]
            # lens = self.disc_head_B_L(lens)[:, 0]
        lens = self.disc_head_L_L(lens)


        
        # Apply the optional activation
        if self.outMode == "sigmoid":
            X = self.Sigmoid(X)
        elif self.outMode == "tanh":
            X = self.Tanh(X)
        
        return X.squeeze(), lens.squeeze()
    
    
    
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