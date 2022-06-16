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
    vocab = {0:"<START>", 1:"<END>", 2:"<PAD>"}
    X_1 = torch.rand((20, 64, 5))
    X_2 = torch.rand((20, 64, 10))
    
    
    
    
    ### Create the model ###
    model = Generator(vocab, 2, 2, 10, 10, 64, 2, torch.device("cpu"))
    out = model()
    model.saveModel(saveDir, saveFile)
    for i in out:
        print(" ".join(i))
    print()
    
    
main()