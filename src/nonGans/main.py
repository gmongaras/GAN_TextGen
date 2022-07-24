import torch
import numpy as np
from models.LSTM import LSTM




def main():
    ### Parameters ###
    
    # Paramters
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
    
    
    # ---
    
    
    
    
    
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