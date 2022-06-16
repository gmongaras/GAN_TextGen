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
    
    
    
    
    ### Create the model ###
    gen = Generator()
    out = gen()
    gen.saveModel(saveDir, saveFile)
    for i in out:
        print(vocab[out[i].item()], end=" ")
    print()
    
    
main()