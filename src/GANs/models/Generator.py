from torch import nn
import torch
import os
from ..blocks.outTrans import outTrans
from ..blocks.inTrans import inTrans
from ..blocks.PositionalEncoding import PositionalEncoding
from ..blocks.MHA import MHA
from ..blocks.MHAwithNorm import MHAwithNorm




class Generator(nn.Module):
    # Inputs:
    #   vocab - The vocabulary of the model
    #   M - Number of input noise embedding blocks
    #   B - Number of transformer blocks to encode the input sequence
    #   O - Number of transformer blocks to get the output sequence
    #   noiseDist - Distribution to sample noise from. Can be one of 
    #               (\"norm\", \"unif\", \"trunc\" (for truncated normal))
    #   hiddenSize - Hidden linear size in the transformer blocks
    #   batchSize - Size of the batch of sentences to generate
    #   embedding_size - Size of each word embedding
    #   sequence_length - Max sequence length of the output sentence
    #   num_heads - Number of heads in the MHA modules
    #   device - The device to put the model on
    def __init__(self, vocab, M, B, O, noiseDist, hiddenSize, batchSize, embedding_size, sequence_length, num_heads, device):
        super(Generator, self).__init__()
        
        # Saved parameters
        self.vocab = vocab
        self.vocab_inv = {vocab[i]:i for i in vocab.keys()}
        self.batchSize = batchSize
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.device = device
        
        # Input embedding (noise to some sort of embedding)
        modules = [nn.Linear(self.embedding_size, self.embedding_size) for i in range(M)]
        self.inEmb = nn.Sequential(*modules).to(device)
        
        # Transformer blocks with cross attention noise
        self.generator = nn.Sequential(
            *[outTrans(embedding_size, noiseDist, num_heads, device, hiddenSize) for i in range(B)]
        ).to(device)
        
        # Output embedding transformer blocks without noise
        self.outEmb = nn.Sequential(
            *[inTrans(embedding_size, embedding_size, num_heads, hiddenSize) for i in range(O)]
        ).to(device)
        
        # Lookup table to convert a distribution to an encoded tensor
        self.Word2Vec = nn.Linear(len(vocab.keys()), embedding_size, bias=False, device=device)
        
        # Positional encoding block
        self.PositionalEncoding = PositionalEncoding(embedding_size, 0.0, len(self.vocab)).to(device)
        
        # Softmax block for the output
        self.soft = nn.Sequential(
            nn.Linear(embedding_size, len(self.vocab.keys())),
            nn.Softmax(dim=-1),
        ).to(device)

        # Noise distribution for the model
        self.noiseDist = noiseDist
        if noiseDist == "unif":
            self.dist = torch.distributions.uniform.Uniform(-1, 1)
        else:
            self.dist = torch.distributions.normal.Normal(0, 1)
    
    
    
    
    # Forward pass
    # Input:
    #   None
    # Output:
    #   A 3-D tensor of shape (N, sequence_length, vocab_size)
    #      where the vocab_size is a softmaxed output
    def forward(self, training=False):
        # Put the model in test/eval mode
        if training:
            self.train()
        else:
            self.eval()
        
        # Generate some noise of shape (N, S, E)
        if self.noiseDist == "trunc":
            a = -1.5
            b = 1.5
            noise = torch.nn.init.trunc_normal_(torch.empty((self.batchSize, self.sequence_length, self.embedding_size)), a=a, b=b).to(self.device)
        else:
            noise = self.dist.sample((self.batchSize, self.sequence_length, self.embedding_size)).float().to(self.device)
        noise.requires_grad = True
        
        # Send the noise through the input blocks
        # to disentagle it
        w = self.inEmb(noise)
        
        # Get a batch of sentences from the noise
        return self.forward_(w)
        
        
    # Forward pass given noise, w
    # Input:
    #   w - Noise of shape (N, S, E)
    # Output:
    #   A 3-D tensor of shape (N, sequence_length, vocab_size)
    #      where the vocab_size is a softmaxed output
    def forward_(self, w):
        # Get positional encodings for all tokens including future
        # tokens that will be generated
        posEnc = self.PositionalEncoding(torch.zeros(w.shape, requires_grad=True, device=self.device))

        Y = None
        
        # The tokenzied output sentences
        out_sent = [[] for i in range(self.batchSize)]
        lens_sent = [[] for i in range(self.batchSize)]
        
        # Iterate to generate a sentence of new words
        for tok in range(0, self.sequence_length):
            # Seed the next word using noise from a distribution. The
            # noise will be of shape (N, 1, E) to append to the
            # current sequence
            if self.noiseDist == "trunc":
                a = -1.5
                b = 1.5
                seeds = torch.nn.init.trunc_normal_(torch.empty((self.batchSize, 1, self.embedding_size)), a=a, b=b).to(self.device)
            else:
                seeds = self.dist.sample((self.batchSize, 1, self.embedding_size)).float().to(self.device)
            seeds.requires_grad = True

            # Add the seends to the sequence
            if Y == None:
                Y = seeds
            else:
                Y = torch.cat((Y, seeds), dim=1) # Add the seeds to the sequence

            # Add the positional encodings to the current
            # sequence and prepare for the next word prediction
            output = Y + posEnc[:, :tok+1]
            
            # Send the input through the generator blocks
            # while adding noise, w
            # (N, S, E) -> (N, S, E)
            for block in self.generator:
                output = block(output, w[:, 0:Y.shape[1]])
                
            # Send the input through the output embedding blocks
            # without any noise
            # (N, S, E) -> (N, S, E)
            output = self.outEmb(output)
                
            # Get the token from the output of the
            # embedding blocks
            # (N, S, E) -> (N, 1, E)
            out_tok = output[:, tok]

            # Send the output through a softmax block
            # (N, 1, E) -> (N, 1, V)
            out_tok_soft = self.soft(out_tok)
            
            # Save the softmax output for the discriminator
            for i in range(self.batchSize):
                out_sent[i].append(out_tok_soft[i])
            
            # Encode the output token from
            # softmax to a word embedding
            out_tok = self.Word2Vec(out_tok_soft)

            # Add the new token to the current
            # generated sequence
            Y = torch.cat((Y.clone()[:, :tok], (out_tok).unsqueeze(1)), dim=1)
                
        
        # Turn the output into a tensor
        out_sent = [torch.stack(sent) for sent in out_sent]
        out_sent = torch.stack(out_sent)


        # Return the output:
        # (N, S, V)
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
        self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device))
