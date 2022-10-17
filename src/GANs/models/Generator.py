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
    #   M - Number of input noise embedding blocks
    #   B - Number of transformer blocks to encode the input sequence
    #   O - Number of transformer blocks to get the output sequence
    #   L - Number of transformer blocks to encode the lengths
    #   noiseDist - Distribution to sample noise from. Can be one of 
    #               (\"norm\", \"unif\", \"trunc\" (for truncated normal))
    #   hiddenSize - Hidden linear size in the transformer blocks
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
    def __init__(self, vocab, M, B, O, L, noiseDist, hiddenSize, batchSize, embedding_size, sequence_length, num_heads, embed_mode, outEnc, device):
        super(Generator, self).__init__()
        
        # Saved states
        self.vocab = vocab
        self.vocab_inv = {vocab[i]:i for i in vocab.keys()}
        self.batchSize = batchSize
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.embed_mode = embed_mode.lower()
        self.embed_mode = embed_mode if (embed_mode == "norm" or embed_mode == "custom") else "norm"
        self.outEnc = outEnc.lower()
        self.device = device
        
        # Input embedding (noise to some sort of embedding)
        modules = [nn.Linear(self.embedding_size, self.embedding_size) for i in range(M)]
        self.inEmb = nn.Sequential(*modules).to(device)
        
        # Transformer blocks
        self.generator = nn.ModuleList([outTrans(embedding_size, embedding_size, noiseDist, num_heads, device, hiddenSize) for i in range(B)]).to(device)
        
        # Output embedding transformer blocks
        self.outEmb = nn.ModuleList([inTrans(embedding_size, embedding_size, num_heads, hiddenSize) for i in range(O)]).to(device)
        
        # If the embed_mode is "custom", use the custom embedding mode
        if embed_mode == "custom":
            self.CustomEmb = CustomEmb(len(vocab.keys()), embedding_size, [len(vocab.keys())//2, len(vocab.keys())//4, 1000, 100], True, self.device)
        # If the embed_mode is "norm", use Word2Vec embeddings
        else:
            # self.Word2Vec = nn.Embedding(len(vocab.keys()), embedding_size).to(device)
            self.Word2Vec = nn.Linear(len(vocab.keys()), embedding_size, bias=False, device=device)
        
        # Positional encoding block
        self.PositionalEncoding = PositionalEncoding(embedding_size, 0.0, len(self.vocab)).to(device)
        
        # Softmax block for the output
        self.soft = nn.Sequential(
            nn.Linear(embedding_size, len(self.vocab.keys())),
            nn.Softmax(dim=-1),
        ).to(device)
        
        # Potential Gumbel Linear block for the output
        if outEnc == "gumb":
            self.gumb_linear = nn.Linear(embedding_size, len(self.vocab.keys())).to(device)

        # Noise distribution for the model
        self.noiseDist = noiseDist
        if noiseDist == "unif":
            self.dist = torch.distributions.uniform.Uniform(-1, 1)
        else:
            self.dist = torch.distributions.normal.Normal(0, 1)

        # Used to embed the sentecen from V -> E for the lengths
        self.lensEmbedding = nn.Linear(len(vocab), embedding_size, device=device, bias=False)

        # Token used to predict the lengths
        self.lensTok = nn.Parameter(torch.rand(1, 1, embedding_size, device=device, requires_grad=True))

        # Model used to predict the legths of the model
        self.lenGen = nn.Sequential(*[inTrans(embedding_size, embedding_size, num_heads, hiddenSize) for i in range(L)]).to(device)

        # Linear layer used to decode the lengths
        self.lensDec_E = nn.Linear(embedding_size, 1, device=device)
        self.lensDec_S = nn.Sequential(
            #nn.Linear(sequence_length, sequence_length, device=device),
            nn.Softmax(dim=-1),
        ).to(device)
        self.lensDec_S_2 = nn.Sequential(
            nn.Linear(embedding_size, sequence_length, device=device),
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
    def forward(self, training=False):
        # Put the model in test/eval mode
        if training:
            self.train()
        else:
            self.eval()
        
        # Generate some noise
        if self.noiseDist == "trunc":
            a = -1.5
            b = 1.5
            noise = torch.nn.init.trunc_normal_(torch.empty((self.batchSize, self.sequence_length, self.embedding_size)), a=a, b=b).to(self.device)
        else:
            noise = self.dist.sample((self.batchSize, self.sequence_length, self.embedding_size)).float().to(self.device)
        noise.requires_grad = True
        
        # Send the noise through the input blocks
        w = self.inEmb(noise)
        #w = torch.unsqueeze(w, dim=-1).repeat(1, 1, self.embedding_size)
        
        # Depending on the embedding mode, pick how to
        # Get a forward pass from the network
        if self.embed_mode == "custom":
            print("Custom is no longer supported. Defaulting to norm")
        return self.forward_(w)
        
        
    # Forward using normal Word2Vec embeddings
    def forward_(self, w):
        # Get positional encodings for all tokens including future
        # tokens that will be generated
        posEnc = self.PositionalEncoding(torch.zeros(w.shape, requires_grad=True, device=self.device))

        # Seed the next word using noise from a distribution
        if self.noiseDist == "trunc":
            a = -1.5
            b = 1.5
            seeds = torch.nn.init.trunc_normal_(torch.empty((self.batchSize, 1, self.embedding_size)), a=a, b=b).to(self.device)
        else:
            seeds = self.dist.sample((self.batchSize, 1, self.embedding_size)).float().to(self.device)
        if seeds.requires_grad == False:
            seeds.requires_grad = True
        
        # Add the seed tokens to the input
        Y = seeds
        
        # The tokenzied output sentences
        out_sent = [[] for i in range(self.batchSize)]
        lens_sent = [[] for i in range(self.batchSize)]
        
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

            # Save the intermediate output for the lens estimator
            for i in range(self.batchSize):
                lens_sent[i].append(out_tok[i])
            
            # Send the output through a softmax block
            # (N, 1, E) -> (N, 1, V)
            out_tok_soft = self.soft(out_tok)
            
            # If the output encoding mode is "gumb", then
            # use the softmax gumbel function as opposed
            # to the softmax function
            if (self.outEnc == "gumb"):
                out_tok_soft = torch.nn.functional.gumbel_softmax(torch.log(torch.clamp(self.gumb_linear(out_tok), 0.00001, torch.inf)), dim=-1)
            
            # Save the softmax output for the discriminator
            for i in range(self.batchSize):
                out_sent[i].append(out_tok_soft[i])

            # Get the argmax of the output tokens
            # out_tok = torch.argmax(out_tok_soft, dim=-1)
            
            # Encode the output token
            # out_tok = self.Word2Vec(out_tok)
            out_tok = self.Word2Vec(out_tok_soft)

            # Add the new token to the output
            Y = torch.cat((Y.clone()[:, :tok], (out_tok).unsqueeze(1)), dim=1)
            
            # If the sequence has not ended, add
            # a new random token to the end of the sentence
            if tok+1 < self.sequence_length:
                # Seed the next word using noise from a distribution
                if self.noiseDist == "trunc":
                    a = -1.5
                    b = 1.5
                    seeds = torch.nn.init.trunc_normal_(torch.empty((self.batchSize, 1, self.embedding_size)), a=a, b=b).to(self.device)
                else:
                    seeds = self.dist.sample((self.batchSize, 1, self.embedding_size)).float().to(self.device)
                seeds.requires_grad = True

                Y = torch.cat((Y, seeds), dim=1) # Add the seeds to the sequence
                
        
        # Turn the output into a tensor
        out_sent = [torch.stack(sent) for sent in out_sent]
        out_sent = torch.stack(out_sent)
        lens_sent = [torch.stack(out) for out in lens_sent]
        lens_sent = torch.stack(lens_sent)










        # # Initiailze the model output to <START> tokens
        # Y = torch.broadcast_to(self.Word2Vec.to(self.device)(torch.tensor(self.vocab_inv["<START>"], dtype=torch.int, device=self.device, requires_grad=False)), (self.batchSize, 1, self.embedding_size)).clone()
        
        # # Get positional encodings for all tokens including future
        # # tokens that will be generated
        # posEnc = self.PositionalEncoding(torch.zeros(w.shape, requires_grad=True, device=self.device))
        
        # # Add the positional encodings to the input tokens
        # Y += posEnc[:, 0:1]
        
        # # The tokenzied output sentences
        # t = torch.nn.functional.one_hot(torch.tensor(self.vocab_inv["<START>"], dtype=torch.int64, device=self.device, requires_grad=False), len(self.vocab))
        # t = t.float().to(self.device)
        # t.requires_grad = True
        # out_sent = [[t] for i in range(self.batchSize)]
        
        # # Iterate to generate a sentence of new words
        # for tok in range(1, self.sequence_length):
        #     # Send the output through the output decoder
        #     output = Y
        #     for block in self.generator:
        #         output = block(w[:, 0:Y.shape[1]], output)
                
        #     # Get the token from the output
        #     out_tok = output[:, tok-1]
            
        #     # Send the output through a softmax block
        #     out_tok_soft = self.soft(out_tok)
            
        #     # Get the argmax of the output tokens
        #     out_tok = torch.argmax(out_tok_soft, dim=-1)
            
        #     # Save the softmax output
        #     for i in range(self.batchSize):
        #         out_sent[i].append(out_tok_soft[i])
        #     #    #out_sent[i].append(self.vocab[out_tok[i].detach().item()])
            
        #     # Encode the output token
        #     out_tok = self.Word2Vec(out_tok)
            
        #     # Save the tokenized new word
        #     #for i in range(self.batchSize):
        #     #    out_sent[i].append(out_tok[i])
            
        #     # Add the new token to the output
        #     #Y = Y.clone()
        #     #Y[:, tok] = out_tok
        #     Y = torch.cat((Y, torch.unsqueeze(out_tok, dim=1)), dim=1)
        #     Y[:, tok] += posEnc[:, tok]
        
        # # Turn the output into a tensor
        # out_sent = [torch.stack(sent) for sent in out_sent]
        # out_sent = torch.stack(out_sent)














        
        # Return the output:
        # (N, S, V) and (N, S)
        return out_sent

    


    def forward_lens(self, out_sent):
        # Get the length estimation from the second model
        # (N, S, V) -> (N, S, E)
        out_sent = self.lensEmbedding(out_sent.clone().detach())
        lens = self.PositionalEncoding(out_sent)
        for block in self.lenGen:
            lens = block(lens)

        # Decode the lengths
        # (N, S, E) -> (N, S, 1)
        lens = self.lensDec_E(lens).squeeze()
        # (N, S) -> (N, S)
        lens = self.lensDec_S(lens)



        # # Get the length estimation from the second model
        # # (N, S, E) -> (N, S, E)
        # out_sent = self.Word2Vec(out_sent.clone().detach())
        # lens = self.PositionalEncoding(out_sent)
        # lens = torch.cat((self.lensTok.repeat(lens.shape[0], 1, 1), lens), dim=1)
        # for block in self.lenGen:
        #     lens = block(lens)

        # # Get the token from the model
        # lens = lens[:, 0]

        # # Decode the token
        # # (N, E) -> (N, S)
        # lens = self.lensDec_S_2(lens)



        return lens
    
    
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
