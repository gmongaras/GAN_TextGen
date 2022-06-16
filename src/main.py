import torch
from Generator import Generator






def main():
    # Paramters
    input_file = "data/data.txt"
    vocab = {0: "hello", 1: ":)", 2:":("}
    
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
    
    
    
    
    ### Create the model ###
    
    # Model paramters
    vocab = {0:"<START>", 1:"<END>", 2:"<PAD>", 3:"hello"}
    M = 2
    N = 2
    batchSize = 1
    embedding_size = 10
    sequence_length = 64
    num_heads = 2
    
    model = Generator(vocab, M, N, batchSize, embedding_size, sequence_length, num_heads, torch.device("cpu"))
    out = model()
    model.saveModel(saveDir, saveFile)
    #for i in out:
    #    print(" ".join(vocab[i.item()]))
    #print()
    
    
main()