from torch import nn
import torch
import os
from blocks.inTrans import inTrans
from blocks.outTrans import outTrans
from blocks.PositionalEncoding import PositionalEncoding




class Generator(nn.Module):
    # Inputs:
    #   vocab - The vocabulary of the model
    #   M - Number of input embedding blocks
    #   N - Number of output embedding blocks
    #   batchSize - Size of the batch of sentences to generate
    #   embedding_size - Size of each word embedding
    #   sequence_length - Max sequence length of the output sentence
    #   num_heads - Number of heads in the MHA modules
    def __init__(self, vocab, M, N, batchSize, embedding_size, sequence_length, num_heads, device):
        super(Generator, self).__init__()
        
        # Saved states
        self.vocab = vocab
        self.vocab_inv = {vocab[i]:i for i in vocab.keys()}
        self.batchSize = batchSize
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.device = device
        
        # Input embedding (noise to some sort of embedding)
        modules = [inTrans(embedding_size, num_heads, embedding_size) for i in range(M)]
        self.inEmb = nn.Sequential(*modules).to(device)
        
        # Output Embedding (<Start> to some output sequence)
        self.outEmb = nn.ModuleList([outTrans(embedding_size, embedding_size, num_heads, embedding_size) for i in range(N)])
        
        # Used to encode each word from a number to a vector
        self.Word2Vec = nn.Embedding(len(vocab.keys()), embedding_size)
        
        # Positional encoding block
        self.PositionalEncoding = PositionalEncoding(embedding_size, 0.1, 100000)
        
        # Softmax block for the output
        self.soft = nn.Sequential(
            nn.Linear(embedding_size, len(self.vocab.keys())),
            nn.Softmax(-1),
        )
    
    
    
    
    # Input:
    #   A random noise tensor of shape (self.sequence_length, self.embedding_size)
    # Output:
    #   A string of max length 256 words
    def forward(self, noise):
        # Put the model in test/eval mode
        self.eval()
        
        # Generate some noise
        #noise = torch.rand((self.sequence_length, self.embedding_size), requires_grad=False)
        
        # Send the noise through the input transformers
        Z = self.inEmb(noise)
        
        # Initialize the output of the model to a bunch of <PAD> tokens
        Y = torch.tensor(self.vocab_inv["<PAD>"], dtype=torch.int, device=self.device, requires_grad=False)
        Y = self.Word2Vec(Y) # Embed the token
        Y = torch.broadcast_to(Y, (self.batchSize, self.sequence_length, self.embedding_size)).clone() # Broadcast
        
        # Change the first token of the output to <START> tokens
        Y[:, 0] = self.Word2Vec(torch.tensor(self.vocab_inv["<START>"], dtype=torch.int, device=self.device, requires_grad=False))
        Y_noEnc = Y.clone()
        del Y
        
        # The output sentences
        out_sent = [[] for i in range(self.batchSize)]
        
        # Iterate to generate a sentence of new words
        for tok in range(1, self.sequence_length):
            # Save the output tensor before psoitional encoding
            Y = Y_noEnc.clone()
            
            # Positionally encode the output
            Y = self.PositionalEncoding(Y_noEnc)
            
            # Send the output through the output decoder
            for block in self.outEmb:
                Y = block(Z, Y)
                
            # Get the token from the output
            out_tok = Y[:, tok]
            
            # Send the output through a softmax block
            out_tok_soft = self.soft(out_tok)
            
            # Get the argmax of the output tokens
            out_tok = torch.argmax(out_tok_soft, dim=-1)
            
            # Save the sentences
            for i in range(self.batchSize):
                out_sent[i].append(out_tok[i])
                #out_sent[i].append(self.vocab[out_tok[i].detach().item()])
            
            # Encode the output token
            out_tok = self.Word2Vec(out_tok)
            
            # Add the new token to the output
            Y_noEnc[:, tok] = out_tok
        
        # Return the output
        return out_sent[0]
    
    
    # Train the model
    def trainModel(self, X, Y):
        # Iterate epochs number of times to train the model
        for epoch in self.epochs:
            # Generate some noise
            noise = torch.rand((self.sequence_length, self.embedding_size))
            
            # Send the noise through the input transformers
            Z = self.inEmb(noise)
            
            # Initialize the output of the model to a bunch of <PAD> tokens
            Y = torch.tensor(self.vocab_inv["<PAD>"], dtype=torch.int, device=self.device, requires_grad=False)
            Y = self.Word2Vec(Y) # Embed the token
            Y = torch.broadcast_to(Y, (self.batchSize, self.sequence_length, self.embedding_size)).clone() # Broadcast
            
            # Change the first token of the output to <START> tokens
            Y[:, 0] = self.Word2Vec(torch.tensor(self.vocab_inv["<START>"], dtype=torch.int, device=self.device, requires_grad=False))
            Y_noEnc = Y.clone()
            del Y
            
            # The output sentences
            out_sent = [[] for i in range(self.batchSize)]
            
            # Iterate to generate a sentence of new words
            for tok in range(1, self.sequence_length):
                # Save the output tensor before psoitional encoding
                Y = Y_noEnc.clone()
                
                # Positionally encode the output
                Y = self.PositionalEncoding(Y_noEnc)
                
                # Send the output through the output decoder
                for block in self.outEmb:
                    Y = block(Z, Y)
                    
                # Get the token from the output
                out_tok = Y[:, tok]
                
                # Send the output through a softmax block
                out_tok_soft = self.soft(out_tok)
                
                # Get the argmax of the output tokens
                out_tok = torch.argmax(out_tok_soft, dim=-1)
                
                # Save the sentences
                for i in range(self.batchSize):
                    out_sent[i].append(self.vocab[out_tok[i].detach().item()])
                
                # Loss stuff
                
                # Encode the output token
                out_tok = self.Word2Vec(out_tok)
                
                # Add the new token to the output
                Y_noEnc[:, tok] = out_tok
            
        # Return the output
        return out_sent
    
    
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
        self.load_state_dict(torch.load(loadDir + os.sep + loadFile))