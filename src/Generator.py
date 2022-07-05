from torch import nn
import torch
import os
from blocks.inTrans import inTrans
from blocks.outTrans import outTrans
from blocks.PositionalEncoding import PositionalEncoding
from blocks.CustomEmb import CustomEmb




class Generator(nn.Module):
    # Inputs:
    #   vocab - The vocabulary of the model
    #   M - Number of input embedding blocks
    #   N - Number of output embedding blocks
    #   batchSize - Size of the batch of sentences to generate
    #   embedding_size - Size of each word embedding
    #   sequence_length - Max sequence length of the output sentence
    #   num_heads - Number of heads in the MHA modules
    #   embed_mode - What embedding mode should be used for the
    #                generator? ("norm" or "custom")
    #   device - The device to put the model on
    def __init__(self, vocab, M, N, batchSize, embedding_size, sequence_length, num_heads, embed_mode, device):
        super(Generator, self).__init__()
        
        # Saved states
        self.vocab = vocab
        self.vocab_inv = {vocab[i]:i for i in vocab.keys()}
        self.batchSize = batchSize
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        embed_mode = embed_mode.lower()
        self.embed_mode = embed_mode if (embed_mode == "norm" or embed_mode == "custom") else "norm"
        self.device = device
        
        # Input embedding (noise to some sort of embedding)
        #modules = [inTrans(embedding_size, num_heads, embedding_size) for i in range(M)]
        #self.inEmb = nn.Sequential(*modules).to(device)
        modules = [nn.Linear(self.sequence_length, self.sequence_length) for i in range(M)]
        self.inEmb2 = nn.Sequential(*modules).to(device)
        
        # Output Embedding (<Start> to some output sequence)
        self.outEmb = nn.ModuleList([outTrans(embedding_size, embedding_size, num_heads, embedding_size, device) for i in range(N)]).to(device)
        
        # If the embed_mode is "custom", use the custom embedding mode
        if embed_mode == "custom":
            self.CustomEmb = CustomEmb(len(vocab.keys()), embedding_size, [len(vocab.keys())//2, len(vocab.keys())//4, len(vocab.keys()), 1000, 100], True, self.device)
        # If the embed_mode is "norm", use Word2Vec embeddings
        else:
            self.Word2Vec = nn.Embedding(len(vocab.keys()), embedding_size).to(device)
        
        # Positional encoding block
        self.PositionalEncoding = PositionalEncoding(embedding_size, 0.0, 10000000).to(device)
        
        # Softmax block for the output
        self.soft = nn.Sequential(
            nn.Linear(embedding_size, len(self.vocab.keys())),
            nn.Softmax(-1),
        ).to(device)
    
    
    
    
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
        
        
        # Generate some noise
        #noise = torch.rand((self.batchSize, self.sequence_length, self.embedding_size), requires_grad=False)
        
        # Send the noise through the input transformers
        w = self.inEmb2(noise)
        w = torch.unsqueeze(w, dim=-1).repeat(1, 1, self.embedding_size)
        
        # Initiailze the model output to <START> tokens
        Y = torch.broadcast_to(self.Word2Vec.to(self.device)(torch.tensor(self.vocab_inv["<START>"], dtype=torch.int, device=self.device, requires_grad=False)), (self.batchSize, 1, self.embedding_size)).clone()
        
        # Get positional encodings for all tokens including future
        # tokens that will be generated
        posEnc = self.PositionalEncoding(torch.zeros(w.shape, requires_grad=True, device=self.device))
        
        # Add the positional encodings to the input tokens
        Y += posEnc[:, 0:1]
        
        # The tokenzied output sentences
        t = torch.tensor(self.vocab_inv["<START>"], dtype=torch.int, device=self.device, requires_grad=False)
        out_sent = [[t] for i in range(self.batchSize)]
        
        # Iterate to generate a sentence of new words
        for tok in range(1, self.sequence_length):
            # Send the output through the output decoder
            output = Y
            for block in self.outEmb:
                output = block(w[:, 0:Y.shape[1]], output)
                
            # Get the token from the output
            out_tok = output[:, tok-1]
            
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
            
            # Add the new token to the output
            #Y = Y.clone()
            #Y[:, tok] = out_tok
            Y = torch.cat((Y, torch.unsqueeze(out_tok, dim=1)), dim=1)
            Y[:, tok] += posEnc[:, tok]
        
        # Turn the output into a tensor
        #out_sent = [torch.stack(sent) for sent in out_sent]
        #out_sent = torch.stack(out_sent)
        
        # Return the output
        return torch.stack(out_sent[0])
    
    

    # Forward pass used during training
    # Input:
    #   Nothing
    # Output:
    #   A 3-D tensor of shape (N, sequence_length, vocab_size)
    #   where the vocab_size is a softmaxed output
    def forward_train(self):
        # Put the model in train mode
        self.train()
        
        # Generate some noise
        noise = torch.rand((self.batchSize, self.sequence_length), requires_grad=False, device=self.device)
        
        # Send the noise through the input transformers
        #Z = self.inEmb(noise)
        w = self.inEmb2(noise)
        w = torch.unsqueeze(w, dim=-1).repeat(1, 1, self.embedding_size)
        
        # Depending on the embedding mode, pick how to
        # Get a forward pass from the network
        if self.embed_mode == "custom":
            return self.forward_train_custom(w)
        return self.forward_train_norm(w)
    
    
    # Forward train using normal Word2Vec embeddings
    def forward_train_norm(self, w):
        # Initiailze the model output to <START> tokens
        Y = torch.broadcast_to(self.Word2Vec.to(self.device)(torch.tensor(self.vocab_inv["<START>"], dtype=torch.int, device=self.device, requires_grad=False)), (self.batchSize, 1, self.embedding_size)).clone()
        
        # Get positional encodings for all tokens including future
        # tokens that will be generated
        posEnc = self.PositionalEncoding(torch.zeros(w.shape, requires_grad=True, device=self.device))
        
        # Add the positional encodings to the input tokens
        Y += posEnc[:, 0:1]
        
        # The tokenzied output sentences
        t = torch.nn.functional.one_hot(torch.tensor(self.vocab_inv["<START>"], dtype=torch.int64, device=self.device, requires_grad=False), len(self.vocab))
        t = t.float().to(self.device)
        t.requires_grad = True
        out_sent = [[t] for i in range(self.batchSize)]
        
        # Iterate to generate a sentence of new words
        for tok in range(1, self.sequence_length):
            # Send the output through the output decoder
            output = Y
            for block in self.outEmb:
                output = block(w[:, 0:Y.shape[1]], output)
                
            # Get the token from the output
            out_tok = output[:, tok-1]
            
            # Send the output through a softmax block
            out_tok_soft = self.soft(out_tok)
            
            # Get the argmax of the output tokens
            out_tok = torch.argmax(out_tok_soft, dim=-1)
            
            # Save the softmax output
            for i in range(self.batchSize):
                out_sent[i].append(out_tok_soft[i])
            #    #out_sent[i].append(self.vocab[out_tok[i].detach().item()])
            
            # Encode the output token
            out_tok = self.Word2Vec(out_tok)
            
            # Save the tokenized new word
            #for i in range(self.batchSize):
            #    out_sent[i].append(out_tok[i])
            
            # Add the new token to the output
            #Y = Y.clone()
            #Y[:, tok] = out_tok
            Y = torch.cat((Y, torch.unsqueeze(out_tok, dim=1)), dim=1)
            Y[:, tok] += posEnc[:, tok]
        
        # Turn the output into a tensor
        out_sent = [torch.stack(sent) for sent in out_sent]
        out_sent = torch.stack(out_sent)
        
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
            
            # Send the output through the output decoder
            output = Y_hat
            for block in self.outEmb:
                output = block(w[:, 0:Y_hat.shape[1]], output)
                
            # Get the token from the output
            out_tok = output[:, tok-1]
            
            # Send the output through a softmax block
            out_tok_soft = self.soft(out_tok)
            
            # Get the argmax of the output tokens
            out_tok = torch.argmax(out_tok_soft, dim=-1)
            
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
            
            # Add the new token to the output
            #Y = Y.clone()
            #Y[:, tok] = out_tok
            Y = torch.cat((Y, torch.unsqueeze(out_tok, dim=1)), dim=1)
        
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