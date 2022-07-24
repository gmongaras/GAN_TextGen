import torch
import numpy as np
from .models.LSTM import LSTM




def main():
    ### Parameters ###
    
    # Vocabualary parameters
    vocabType = "char"          # The vocabulary type ("word" or "char")
    input_file = "data/Fortunes/data.txt"
    vocab_file = "vocab_fortunes.csv"
    
    # Saving/Loading paramters
    saveDir = "models"
    genSaveFile = "gen_model.pkl"
    discSaveFile = "disc_model.pkl"
    trainGraphFile = "trainGraph.png"
    
    loadDir = "models"
    genLoadFile = "gen_model.pkl"
    discLoadFile = "disc_model.pkl"
    
    
    # Model parameters
    modelType = "rnn"          # Model type to use ("rnn", "transformer", or "both")
    outputType = "char"         # What should the model output? ("word" or "char")
    batchSize = 128             # Batch size used when training the model
    seqLength = 300             # Length of the sequence to train the model on
    
    # RNN parameters (if used)
    #
    
    # Transformer parameters (if used)
    #
    
    # Words parameters (if used)
    encodingDim = 1             # Size to encode each word in the sequence to
    
    # Character parameters (if used)
    #
    
    
    
    ### Load in the vocabulary and data ###
    
    
    
    
    ### Create the model ###
    if modelType.lower() == "transformer":
        model = RNN()
    else:
        model = Transformer()
    
    
    
    
    
    ### Train the model ###
    N = 10
    E = 1
    S = 2
    H = 20
    V = 100
    layers = 2
    x_t = torch.rand((N, S, E))
    torch.nn.LSTM(input_size=E, hidden_size=H, num_layers=4)(x_t)
    x_t2 = LSTM(E, H, V, layers, 0.1, torch.device("cpu"))(x_t)
    
    

if __name__=='__main__':
    main()