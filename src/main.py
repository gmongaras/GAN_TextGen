import torch
from Generator import Generator
from helpers import loadVocab






def main():
    # Paramters
    input_file = "data/data.txt"
    vocab_file = "vocab.csv"
    
    # Saving/Loading paramters
    saveDir = "models/"
    saveFile = "model.pkl"
    loadDir = "models/"
    loadFile = "model.pkl"
    
    
    
    ### Load in the data ###
    ...
    from Generator import Generator
    X_1 = torch.rand((20, 64, 5))
    X_2 = torch.rand((20, 64, 10))
    
    
    ### Load in the vocab ###    
    vocab = loadVocab(vocab_file)
    
    
    ### Create the model ###
    
    # Model paramters
    M = 2
    N = 2
    batchSize = 1
    embedding_size = 10
    sequence_length = 64
    num_heads = 2
    
    noise = torch.rand((sequence_length, embedding_size), requires_grad=False)
    
    model = Generator(vocab, M, N, batchSize, embedding_size, sequence_length, num_heads, torch.device("cpu"))
    out = model(noise)
    model.saveModel(saveDir, saveFile)
    for i in out:
        print(vocab[i.item()], end=" ")
    print()
    
    
main()