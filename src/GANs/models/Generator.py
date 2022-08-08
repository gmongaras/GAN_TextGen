from torch import nn
import torch
import os
from ..blocks.outTrans import outTrans
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
        #modules = [inTrans(embedding_size, num_heads, embedding_size) for i in range(M)]
        #self.inEmb = nn.Sequential(*modules).to(device)
        modules = [nn.Linear(self.sequence_length, self.sequence_length) for i in range(M)]
        self.inEmb2 = nn.Sequential(*modules).to(device)
        
        # Transformer blocks
        self.transBlocks = nn.ModuleList([outTrans(embedding_size, embedding_size, gausNoise, num_heads, embedding_size, device) for i in range(B)]).to(device)
        
        # Output embedding MHA blocks
        self.outEmb = nn.ModuleList([MHAwithNorm(embedding_size, embedding_size, embedding_size, num_heads) for i in range(O)]).to(device)
        
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
    
    
    
    
    # Forward pass used after training
    # Input:
    #   A random noise tensor of shape (self.sequence_length)
    # Output:
    #   A string of max length 256 words
    def forward(self, noise):
        # Put the model in test/eval mode
        self.eval()
        
        # Make sure the noise is in the correct format
        if len(noise.shape) != 2:
            noise = torch.unsqueeze(noise, dim=0)
        noise = noise.to(self.device)
        
        # Send the noise through the input transformers
        w = self.inEmb2(noise)
        w = torch.unsqueeze(w, dim=-1).repeat(1, 1, self.embedding_size)
        
        # Depending on the embedding mode, pick how to
        # Get a forward pass from the network
        if self.embed_mode == "custom":
            return self.forward_custom(w)
        return self.forward_norm(w)
        
        
    # Forward using normal Word2Vec embeddings
    def forward_norm(self, w):
        # Initiailze the model output to <START> tokens
        Y = torch.broadcast_to(self.Word2Vec.to(self.device)(torch.tensor(self.vocab_inv["<START>"], dtype=torch.int, device=self.device, requires_grad=False)), (self.batchSize, 1, self.embedding_size)).clone()
        
        # Get positional encodings for all tokens including future
        # tokens that will be generated
        posEnc = self.PositionalEncoding(torch.zeros(w.shape, requires_grad=True, device=self.device))
        
        # Add the positional encodings to the input tokens
        Y += posEnc[:, 0:1]
        
        # Used to get the next prediction from the model
        nextTok = torch.broadcast_to(self.Word2Vec.to(self.device)(torch.tensor(self.vocab_inv["<NEXT>"], dtype=torch.int, device=self.device, requires_grad=False)), (self.batchSize, 1, self.embedding_size)).clone()
        nextTok = nextTok.float().to(self.device)
        
        # Add the next tokens to the input
        Y = torch.cat((Y, nextTok), dim=1)
        Y[:, 1] += posEnc[:, 1]
        
        # The tokenzied output sentences
        t = torch.tensor(self.vocab_inv["<START>"], dtype=torch.int, device=self.device, requires_grad=False)
        out_sent = [[t] for i in range(self.batchSize)]
        
        # Iterate to generate a sentence of new words
        for tok in range(1, self.sequence_length):
            # Send the input through the tranformer blocks
            output = Y.clone()
            n = w[:, 0:Y.shape[1]]
            for block in self.outEmb:
                output = block(n, output)
                
            # Send the input through the output MHA blocks
            for block in self.outEmb:
                output = block(output, output)
                
            # Get the token from the output
            out_tok = output[:, tok]
            
            # Send the output through a softmax block
            out_tok_soft = self.soft(out_tok)
            
            # Get the argmax of the output tokens
            out_tok = torch.argmax(out_tok_soft, dim=-1)
            
            # Save the softmax output
            #for i in range(self.batchSize):
            #    out_sent[i].append(out_tok_soft[i])
            #    #out_sent[i].append(self.vocab[out_tok[i].detach().item()])
            
            # Save the tokenized new word
            for i in range(self.batchSize):
                out_sent[i].append(out_tok[i])
                
            # Encode the output token
            out_tok = self.Word2Vec(out_tok)
            
            # Add the new token to the output and add a new
            # <NEXT> token to the sequence
            if tok+1 < self.sequence_length:
                Y[:, tok] = out_tok + posEnc[:, tok] # Replace the <NEXT> token with the new token prediction with position encodings
                Y = torch.cat((Y, nextTok.clone()), dim=1) # Add the <NEXT> token
                Y[:, tok+1] += posEnc[:, tok+1] # Add positional encodings to the <NEXT> token
        
        # Turn the output into a tensor
        #out_sent = [torch.stack(sent) for sent in out_sent]
        #out_sent = torch.stack(out_sent)
        
        # Return the output
        return torch.stack(out_sent[0])

    
    
    # Forward using custom word embeddings
    def forward_custom(self, w):
        # Initiailze the model output to <START> tokens
        # These tokens are one-hot encoded and will be
        # transformed during training
        Y = torch.broadcast_to(torch.nn.functional.one_hot(torch.tensor(self.vocab_inv["<START>"], dtype=torch.int64, device=self.device, requires_grad=False), len(self.vocab)), (self.batchSize, 1, len(self.vocab))).clone()
        Y = Y.float()
        
        # Get positional encodings for all tokens including future
        # tokens that will be generated
        posEnc = self.PositionalEncoding(torch.zeros(w.shape, requires_grad=False, device=self.device))
        
        # Used to get the next prediction from the model
        nextTok = torch.broadcast_to(torch.nn.functional.one_hot(torch.tensor(self.vocab_inv["<NEXT>"], dtype=torch.int64, device=self.device, requires_grad=False), len(self.vocab)), (self.batchSize, 1, len(self.vocab))).clone()
        nextTok = nextTok.float().to(self.device)
        
        # Add the next tokens to the input
        Y = torch.cat((Y, nextTok), dim=1)
        
        # The tokenzied output sentences
        t = torch.tensor(self.vocab_inv["<START>"], dtype=torch.int, device=self.device, requires_grad=False)
        out_sent = torch.zeros((w.shape[0], w.shape[1], len(self.vocab)), requires_grad=False, dtype=torch.float32, device=self.device)
        out_sent[:, 0] = t.broadcast_to(out_sent.shape[0], out_sent.shape[2])
        
        # Iterate to generate a sentence of new words
        for tok in range(1, self.sequence_length):
            # Create a new tensor, Y_hat which is transformed
            # into the wanted embedding size
            Y_hat = self.CustomEmb(Y)
            
            # Add positional encodings to the transformed input
            Y_hat += posEnc[:, 0:tok]
            
            # Send the output through the transformer blocks
            output = Y_hat
            for block in self.transBlocks:
                output = block(w[:, 0:Y_hat.shape[1]], output)
                
            # Send the input through the output MHA blocks
            for block in self.outEmb:
                output = block(output, output)
                
            # Get the token from the output
            out_tok = output[:, tok]
            
            # Send the output through a softmax block
            out_tok_soft = self.soft(out_tok)
            
            # Get the argmax of the output tokens
            out_tok = torch.argmax(out_tok_soft, dim=-1)
            
            # Save the softmax output
            #for i in range(self.batchSize):
            #    out_sent[i].append(out_tok_soft[i])
            #    #out_sent[i].append(self.vocab[out_tok[i].detach().item()])
            
            # Save the tokenized new word
            out_sent[:, tok] = out_tok[tok]
                
            # Encode the output token
            out_tok = torch.nn.functional.one_hot(out_tok.long(), len(self.vocab)).float()
            
            
            # Add the new token to the output and add a new
            # <NEXT> token to the sequence
            if tok+1 < self.sequence_length:
                Y[:, tok] = out_tok # Replace the <NEXT> token with the new token
                Y = torch.cat((Y, nextTok.clone()), dim=1) # Add the <NEXT> token
        
        # Turn the output into a tensor
        #out_sent = [torch.stack(sent) for sent in out_sent]
        #out_sent = torch.stack(out_sent)
        
        # Return the output
        return torch.stack(out_sent[0])
    
    

    # Forward pass used during training
    # Input:
    #   HideAfterEnd - True to hide any tokens after the <END> token in the 
    #                  discriminator MHA with a mask, False to keep these tokens visibile
    # Output:
    #   A 3-D tensor of shape (N, sequence_length, vocab_size)
    #      where the vocab_size is a softmaxed output
    #   If HideAfterEnd is True, then a masks tensor of shape
    #      (N, sequence_length) is also outputted
    def forward_train(self, HideAfterEnd=False):
        # Put the model in train mode
        self.train()
        
        # Generate some noise
        noise = torch.rand((self.batchSize, self.sequence_length), requires_grad=False, device=self.device)
        
        # Send the noise through the input transformers
        w = self.inEmb2(noise)
        w = torch.unsqueeze(w, dim=-1).repeat(1, 1, self.embedding_size)
        
        # Depending on the embedding mode, pick how to
        # Get a forward pass from the network
        if self.embed_mode == "custom":
            return self.forward_train_custom(w)
        return self.forward_train_norm(w, HideAfterEnd)
    
    
    # Forward train using normal Word2Vec embeddings
    def forward_train_norm(self, w, HideAfterEnd):
        # Initiailze the model output to <START> tokens
        Y = torch.broadcast_to(self.Word2Vec.to(self.device)(torch.tensor(self.vocab_inv["<START>"], dtype=torch.int, device=self.device, requires_grad=False)), (self.batchSize, 1, self.embedding_size)).clone()
        
        # Get positional encodings for all tokens including future
        # tokens that will be generated
        posEnc = self.PositionalEncoding(torch.zeros(w.shape, requires_grad=True, device=self.device))
        
        # Add the positional encodings to the input tokens
        Y += posEnc[:, 0:1]
        if Y.requires_grad == False:
            Y.requires_grad = True
        
        # Used to get the next prediction from the model
        nextTok = torch.broadcast_to(self.Word2Vec.to(self.device)(torch.tensor(self.vocab_inv["<NEXT>"], dtype=torch.int, device=self.device, requires_grad=False)), (self.batchSize, 1, self.embedding_size)).clone()
        nextTok = nextTok.float().to(self.device)
        if nextTok.requires_grad == False:
            nextTok.requires_grad = True
        
        # Add the next tokens to the input
        Y = torch.cat((Y, nextTok), dim=1)
        Y[:, 1] += posEnc[:, 1]
        
        # The tokenzied output sentences
        t = torch.nn.functional.one_hot(torch.tensor(self.vocab_inv["<START>"], dtype=torch.int64, device=self.device, requires_grad=False), len(self.vocab))
        t = t.float().to(self.device)
        if t.requires_grad == False:
            t.requires_grad = True
        out_sent = [[t] for i in range(self.batchSize)]
        
        # Iterate to generate a sentence of new words
        for tok in range(1, self.sequence_length):
            # Send the input through the transformer blocks
            output = Y
            for block in self.transBlocks:
                output = block(w[:, 0:Y.shape[1]], output)
                
            # Send the input through the output MHA blocks
            for block in self.outEmb:
                output = block(output, output)
                
            # Get the token from the output
            out_tok_b = output[:, tok]
            
            # Send the output through a softmax block
            out_tok_soft = self.soft(out_tok_b)
            
            # Get the argmax of the output tokens
            out_tok = torch.argmax(out_tok_soft, dim=-1)
            
            # If the output encoding mode is "gumb", then
            # use the softmax gumbel function as opposed
            # to the softmax function
            if (self.outEnc == "gumb"):
                out_tok_soft = torch.nn.functional.gumbel_softmax(torch.log(torch.clamp(self.gumb_linear(out_tok_b), 0.00001, torch.inf)), dim=-1)
            
            # Save the softmax output
            for i in range(self.batchSize):
                out_sent[i].append(out_tok_soft[i])
            
            # Encode the output token
            out_tok = self.Word2Vec(out_tok)
            
            # Save the tokenized new word
            #for i in range(self.batchSize):
            #    out_sent[i].append(out_tok[i])
            
            # Add the new token to the output and add a new
            # <NEXT> token to the sequence
            if tok+1 < self.sequence_length:
                Y = torch.cat((Y.clone()[:, :tok], (out_tok + posEnc[:, tok]).unsqueeze(1)), dim=1)
                Y = torch.cat((Y, nextTok.clone()), dim=1) # Add the <NEXT> token
                Y[:, tok+1] += posEnc[:, tok+1] # Add positional encodings to the <NEXT> token
                
        
        # Turn the output into a tensor
        out_sent = [torch.stack(sent) for sent in out_sent]
        out_sent = torch.stack(out_sent)
        
        
        if HideAfterEnd:
            # Position of any <END> tokens in the outut
            end_pos = Y.shape[1]*torch.ones((Y.shape[0]), requires_grad=False, dtype=torch.int16)
            
            # Get the position of the first <END> token for each generated sequence
            whereEnd = torch.where(torch.argmax(out_sent, dim=-1).cpu() == self.vocab_inv["<END>"])
            uniq = torch.unique(whereEnd[0], dim=0, sorted=False, return_inverse=True, return_counts=True)
            vals = torch.zeros(uniq[0].shape, dtype=torch.int16, requires_grad=False)
            i = 0
            for j in range(0, uniq[0].shape[0]):
                vals[j] = whereEnd[1][i]
                i += uniq[2][j]
            end_pos[uniq[0]] = vals.to(torch.int16)
            end_pos = end_pos.long()
            
            # Generate a tensor used to mask the MHA
            masks = torch.zeros((out_sent.shape[0], out_sent.shape[1]), dtype=torch.bool, requires_grad=False, device=self.device)
            
            # Make all parts of the tensor after the <PAD> tokens
            # True to signify that the position should not be attended to
            for i in range(0, len(end_pos)):
                if end_pos[i] == -1:
                    continue
                masks[i, end_pos[i]+1:] = True
                
            # Return the output and the masks
            return out_sent, masks.to(self.device) 
        
        # Return the output
        return out_sent
    
    
    # Forward train using normal custom embeddings
    def forward_train_custom(self, w):
        # Initiailze the model output to <START> tokens
        # These tokens are one-hot encoded and will be
        # transformed during training
        Y = torch.broadcast_to(torch.nn.functional.one_hot(torch.tensor(self.vocab_inv["<START>"], dtype=torch.int64, device=self.device, requires_grad=False), len(self.vocab)), (self.batchSize, 1, len(self.vocab))).clone()
        Y = Y.float()
        Y.requires_grad = True
        
        # Get positional encodings for all tokens including future
        # tokens that will be generated
        posEnc = self.PositionalEncoding(torch.zeros(w.shape, requires_grad=True, device=self.device))
        
        # Used to get the next prediction from the model
        nextTok = torch.broadcast_to(torch.nn.functional.one_hot(torch.tensor(self.vocab_inv["<NEXT>"], dtype=torch.int64, device=self.device, requires_grad=False), len(self.vocab)), (self.batchSize, 1, len(self.vocab))).clone()
        nextTok = nextTok.float().to(self.device)
        nextTok.requires_grad = True
        
        # Add the next tokens to the input
        Y = torch.cat((Y, nextTok), dim=1)
        
        # The tokenzied output sentences
        t = torch.nn.functional.one_hot(torch.tensor(self.vocab_inv["<START>"], dtype=torch.int64, device=self.device, requires_grad=False), len(self.vocab))
        t = t.float().to(self.device)
        t.requires_grad = True
        out_sent = [[t] for i in range(self.batchSize)]
        
        # Iterate to generate a sentence of new words
        for tok in range(1, self.sequence_length):
            # Create a new tensor, Y_hat which is transformed
            # into the wanted embedding size
            Y_hat = self.CustomEmb(Y)
            
            # Add positional encodings to the transformed input
            Y_hat += posEnc[:, 0:tok]
            
            # Send the output through the transformer blocks
            output = Y_hat
            for block in self.transBlocks:
                output = block(w[:, 0:Y_hat.shape[1]], output)
                
            # Send the input through the output MHA blocks
            for block in self.outEmb:
                output = block(output, output)
                
            # Get the token from the output
            out_tok_b = output[:, tok]
            
            # Send the output through a softmax block
            out_tok_soft = self.soft(out_tok_b)
            
            # Get the argmax of the output tokens
            out_tok = torch.argmax(out_tok_soft, dim=-1)
            
            # If the output encoding mode is "gumb", then
            # use the softmax gumbel function as opposed
            # to the softmax function
            if (self.outEnc == "gumb"):
                out_tok_soft = torch.nn.functional.gumbel_softmax(torch.log(torch.clamp(self.gumb_linear(out_tok_b), 0.00001, torch.inf)), dim=-1)
            
            # Save the softmax output
            for i in range(self.batchSize):
                out_sent[i].append(out_tok_soft[i])
            #    #out_sent[i].append(self.vocab[out_tok[i].detach().item()])
            
            # Encode the output token
            out_tok = torch.nn.functional.one_hot(out_tok.long(), len(self.vocab)).float()
            out_tok.requires_grad = True
            
            # Save the tokenized new word
            #for i in range(self.batchSize):
            #    out_sent[i].append(out_tok[i])
            
            # Add the new token to the output and add a new
            # <NEXT> token to the sequence
            if tok+1 < self.sequence_length:
                Y[:, tok] = out_tok # Replace the <NEXT> token with the new token
                Y = torch.cat((Y, nextTok.clone()), dim=1) # Add the <NEXT> token
        
        # Turn the output into a tensor
        out_sent = [torch.stack(sent) for sent in out_sent]
        out_sent = torch.stack(out_sent)
        
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
        self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device))
