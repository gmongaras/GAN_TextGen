import torch
import numpy as np
from .Model import Model
from ..helpers.helpers import loadVocab
from .helpers.load_chars import load_chars
from .helpers.load_words import load_words


cpu = torch.device('cpu')
try:
    if torch.has_mps:
        gpu = torch.device("mps")
    else:
        gpu = torch.device('cuda:0')
except:
    gpu = torch.device('cuda:0')




def main():
    ### Parameters ###
    
    # Vocabualary parameters
    input_file = "data/Fortunes/data.txt"
    vocab_file = "vocab_chars.csv"
    
    # Saving/Loading paramters
    saveDir = "models"
    saveFile = "model.pkl"
    trainGraphFile = "trainGraph.png"
    saveSteps = 100
    
    loadDir = "models"
    loadFile = "model.pkl"
    
    
    # Model parameters
    modelType = "transformer"           # Model type to use ("rnn", "transformer", or "both")
    outputType = "char"         # What should the model output? ("word" or "char")
    epochs = 20000              # Number of epochs to train the model for
    batchSize = 128             # Batch size used when training the model
    seqLength = 300             # Length of the sequence to train the model on
    input_size = 1              # (E) The embedding size of the input for each char or word
    layers = 1                  # The number of LSTM blocks stacked ontop of each other
    dropout = 0.2               # Dropout rate in the model
    
    device = "cpu"              # Device to put the model on ("cpu", "partgpu", or "fullgpu")
    
    # RNN parameters (if used)
    hidden_size = 256           # (H) The size of the hidden state
    
    # Transformer parameters (if used)
    num_heads = 1               # Number of heads for each MHA block
    
    # Word or Character parameters
    maxLen = 300                # Max sequence length to load in
    
    # Words parameters (if used)
    encodingDim = 1             # Size to encode each word in the sequence to
    
    # Character parameters (if used)
    #
    
    
    
    ### Load in the vocabulary and data ###
    
    # Load in the vocab
    vocab = loadVocab(vocab_file)
    vocab_inv = {vocab[i]:i for i in vocab}
    
    # Load in the data
    if outputType == "word":
        X, y = load_words()
    else:
        X, y = load_chars(input_file, maxLen, vocab_inv)
    
    
    
    ### Create the model ###
    if modelType.lower() == "transformer":
        model = Model(modelType, outputType, input_size,
                      len(vocab), layers, dropout, device, saveDir, saveFile,
                      num_heads=num_heads)
    else:
        model = Model(modelType, outputType, input_size, 
                      len(vocab), layers, dropout, device, saveDir, saveFile,
                      hidden_size=hidden_size)
    
    
    
    
    
    ### Train the model ###
    
    model.trainModel(X, y, epochs, batchSize, saveSteps)
    
    

if __name__=='__main__':
    main()