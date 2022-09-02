from torch import nn
import torch
import os
from ..blocks.outTrans import outTrans
from ..blocks.inTrans import inTrans
from ..blocks.PositionalEncoding import PositionalEncoding
from ..blocks.CustomEmb import CustomEmb
from ..blocks.MHA import MHA
from ..blocks.MHAwithNorm import MHAwithNorm




class Generator(nn.Module):
    # Inputs:
    #   vocab - The vocabulary of the model
    #   M - Number of input embedding blocks
    #   B - Number of transformer blocks when encoding the output
    #   O - Number of MHA blocks after the transformer blocks
    #       when encoding the output
    #   gausNoise - True to add pure gaussian noise in the B output blocks,
    #               False to not add gaussian noise
    #   batchSize - Size of the batch of sentences to generate
    #   embedding_size - Size of each word embedding
    #   sequence_length - Max sequence length of the output sentence
    #   num_heads - Number of heads in the MHA modules
    #   embed_mode - What embedding mode should be used for the
    #                generator? ("norm" or "custom")
    #   outEnc - What encodning mode should be used for the
    #            output sequences which will be fed into the
    #            discriminator? ("norm", "gumb")
    #   device - The device to put the model on
    def __init__(self, vocab, M, B, O, gausNoise, batchSize, embedding_size, sequence_length, num_heads, embed_mode, outEnc, device):
        super(Generator, self).__init__()
        
        # Saved states
        self.vocab = vocab
        self.vocab_inv = {vocab[i]:i for i in vocab.keys()}
        self.gausNoise = gausNoise
        self.batchSize = batchSize
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.embed_mode = embed_mode.lower()
        self.embed_mode = embed_mode if (embed_mode == "norm" or embed_mode == "custom") else "norm"
        self.outEnc = outEnc.lower()
        self.device = device
        
        # Input embedding (noise to some sort of embedding)
        modules = [nn.Linear(self.sequence_length, self.sequence_length) for i in range(M)]
        self.inEmb = nn.Sequential(*modules).to(device)
        
        # Transformer blocks
        self.generator = nn.ModuleList([outTrans(embedding_size, embedding_size, gausNoise, num_heads, device, 512) for i in range(B)]).to(device)
        
        # Output embedding MHA blocks
        self.outEmb = nn.ModuleList([inTrans(embedding_size, num_heads, 512) for i in range(O)]).to(device)
        
        # If the embed_mode is "custom", use the custom embedding mode
        if embed_mode == "custom":
            self.CustomEmb = CustomEmb(len(vocab.keys()), embedding_size, [len(vocab.keys())//2, len(vocab.keys())//4, 1000, 100], True, self.device)
        # If the embed_mode is "norm", use Word2Vec embeddings
        else:
            self.Word2Vec = nn.Embedding(len(vocab.keys()), embedding_size).to(device)
        
        # Positional encoding block
        self.PositionalEncoding = PositionalEncoding(embedding_size, 0.0, len(self.vocab)).to(device)
        
        # Softmax block for the output
        self.soft = nn.Sequential(
            nn.Linear(embedding_size, len(self.vocab.keys())),
            nn.Softmax(-1),
        ).to(device)
        
        # Potential Gumbel Linear block for the output
        if outEnc == "gumb":
            self.gumb_linear = nn.Linear(embedding_size, len(self.vocab.keys())).to(device)

        # Normal distribution for the model
        self.normDist = torch.distributions.normal.Normal(0, 1)

        # <LEN> token for the model
        self.lenTok = torch.ones((self.batchSize, 1, embedding_size), device=device, dtype=torch.float32, requires_grad=False)

        # Model used to predict the legths of the model
        self.lenGen = nn.ModuleList([outTrans(embedding_size, embedding_size, gausNoise, num_heads, device, 512) for i in range(2)]).to(device)

        # Linear layer used to decode the lengths
        self.lensDec_E = nn.Linear(embedding_size, 1, device=device)
        self.lensDec_S = nn.Sequential(
            nn.Linear(sequence_length, self.sequence_length, device=device),
            nn.Softmax(dim=-1),
        ).to(device)
    
    
    
    
    # Forward pass
    # Input:
    #   None
    # Output:
    #   A 3-D tensor of shape (N, sequence_length, vocab_size)
    #      where the vocab_size is a softmaxed output
    #   A 3-D tensor of shape (N, sequence_length)
    #      where each batch element is the softmax output representing
    #      the probability of the length of the Nth sentence
    def forward_(self, training=False):
        # Put the model in test/eval mode
        if training:
            self.train()
        else:
            self.eval()
        
        # Generate some noise
        noise = torch.rand((self.batchSize, self.sequence_length), requires_grad=False, device=self.device)
        
        # Send the noise through the input transformers
        w = self.inEmb(noise)
        w = torch.unsqueeze(w, dim=-1).repeat(1, 1, self.embedding_size)
        
        # Depending on the embedding mode, pick how to
        # Get a forward pass from the network
        if self.embed_mode == "custom":
            print("Custom is no longer supported. Defaulting to norm")
        return self.forward(w)
        
        
    # Forward using normal Word2Vec embeddings
    def forward(self, w):
        # Get positional encodings for all tokens including future
        # tokens that will be generated
        posEnc = self.PositionalEncoding(torch.zeros(w.shape, requires_grad=True, device=self.device))

        # Seed the next word prediction rom the model
        seeds = self.normDist.sample((self.batchSize, 1, self.embedding_size)).float().to(self.device)
        if seeds.requires_grad == False:
            seeds.requires_grad = True
        
        # Add the seed tokens to the input
        Y = seeds
        
        # The tokenzied output sentences
        out_sent = [[] for i in range(self.batchSize)]
        
        # Iterate to generate a sentence of new words
        for tok in range(0, self.sequence_length):
            # Initialize the input as the current
            # output with positional encodings
            output = Y + posEnc[:, :tok+1]
            
            # Send the input through the generator blocks
            # (N, S, E) -> (N, S, E)
            for block in self.generator:
                output = block(w[:, 0:Y.shape[1]], output)
                
            # Send the input through the output embedding blocks
            # (N, S, E) -> (N, S, E)
            for block in self.outEmb:
                output = block(output)
                
            # Get the token from the output
            # (N, S, E) -> (N, 1, E)
            out_tok = output[:, tok]
            
            # Send the output through a softmax block
            # (N, 1, E) -> (N, 1, V)
            out_tok_soft = self.soft(out_tok)
            
            # If the output encoding mode is "gumb", then
            # use the softmax gumbel function as opposed
            # to the softmax function
            if (self.outEnc == "gumb"):
                out_tok_soft = torch.nn.functional.gumbel_softmax(torch.log(torch.clamp(self.gumb_linear(out_tok), 0.00001, torch.inf)), dim=-1)
            
            # Save the softmax output for the
            for i in range(self.batchSize):
                out_sent[i].append(out_tok_soft[i])
            
            # Add the new token to the output and add a new
            # seed to the sequence
            if tok+1 < self.sequence_length:
                # Add the new token to the output
                Y = torch.cat((Y.clone()[:, :tok], (out_tok).unsqueeze(1)), dim=1)

                # Seed the next word using gaussian noise
                seeds = self.normDist.sample((self.batchSize, 1, self.embedding_size)).float().to(self.device)
                Y = torch.cat((Y, seeds), dim=1) # Add the seeds
                
        
        # Turn the output into a tensor
        out_sent = [torch.stack(sent) for sent in out_sent]
        out_sent = torch.stack(out_sent)



        ## Feed the whole output into the generator.
        ## Note: We are using the output with an attached graph
        ## so that each outputted word is effected by the
        ## gradient of the length token.

        # Get the length estimation from the second model
        # (N, S, E) -> (N, S, E)
        lens = Y
        for block in self.lenGen:
            lens = block(lens, lens)

        # Decode the lengths
        # (N, S, E) -> (N, S, 1)
        lens = self.lensDec_E(lens).squeeze()
        # (N, S) -> (N, S)
        lens = self.lensDec_S(lens)

        # For each length, replace the values with PAD tokens
        pad_tok = torch.nn.functional.one_hot(torch.tensor(self.vocab_inv["<PAD>"], dtype=torch.int64, device=self.device, requires_grad=False), len(self.vocab))
        pad_tok = pad_tok.float().to(self.device)
        for i in range(0, lens.shape[0]):
            out_sent[i, torch.argmax(lens, dim=-1)[i].item()+1:] = pad_tok.clone()

        
        # Return the output:
        # (N, S, V) and (N, S)
        return out_sent, lens
    
    
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
