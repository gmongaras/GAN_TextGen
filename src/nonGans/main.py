import torch
import numpy as np
from .Model import Model
from .BothModel import Both
from ..helpers.helpers import loadVocab
from .helpers.load_chars import load_chars
from .helpers.load_words import load_words
from .helpers.load_both import load_both


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
    modelType = "both"           # Model type to use ("rnn", "transformer", or "both")
    outputType = "char"         # What should the model output? ("word" or "char") (Note: This is not used if modelType is "both")
    epochs = 20000              # Number of epochs to train the model for
    batchSize = 128             # Batch size used when training the model
    layers = 1                  # The number of LSTM blocks stacked ontop of each other
    dropout = 0.2               # Dropout rate in the model
    
    device = "cpu"              # Device to put the model on ("cpu", "partgpu", or "fullgpu")
    
    # RNN parameters (if used)
    hidden_size = 256           # (H) The size of the hidden state
    
    # Transformer parameters (if used)
    num_heads = 1               # Number of heads for each MHA block
    
    # Both model parameters (if used)
    T = 4                       # Number of transformer layers in this model
    L = 4                       # Number of LSTM layers in this model
    word_length = 20            # Max length of each word
    
    # Word or Character parameters
    seqLength = 64              # Length of the sequence to train the model on (number of words or characters)
    input_size = 30             # (E) The embedding size of the input for each char or word
    lower = False               # True to lowercase chars/words when embedding. False otherwise
    
    # Words parameters (if used)
    limit = 10                  # Limit on the number of sentences to load in
    
    # Character parameters (if used)
    #
    
    
    
    ### Load in the vocabulary and data ###
    
    # Load in the vocab
    vocab = loadVocab(vocab_file)
    vocab_inv = {vocab[i]:i for i in vocab}
    
    # Load in the data
    if modelType == "both":
        y = load_both(input_file, seqLength, vocab_inv, word_length, lower, limit)
    elif outputType == "word":
        X, y = load_words(input_file, seqLength, vocab_inv, input_size, lower, limit)
    else:
        X, y = load_chars(input_file, seqLength, vocab_inv, lower)
    
    
    
    ### Create the model ###
    if modelType.lower() == "both":
        model = Both(input_size, vocab_inv, dropout, device, 
                     saveDir, saveFile, num_heads, hidden_size,
                     T, L, word_length)
    elif modelType.lower() == "transformer":
        model = Model(modelType, outputType, input_size,
                      len(vocab), layers, dropout, device, saveDir, saveFile,
                      num_heads=num_heads)
    else:
        model = Model(modelType, outputType, input_size, 
                      len(vocab), layers, dropout, device, saveDir, saveFile,
                      hidden_size=hidden_size)
    
    
    
    
    
    ### Train the model ###
    
    if modelType.lower() == "both":
        model.trainModel(y, epochs, batchSize, saveSteps)
    else:
        model.trainModel(X, y, epochs, batchSize, saveSteps)
    
    

if __name__=='__main__':
    main()