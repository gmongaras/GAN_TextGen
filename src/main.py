import torch
from Diff_GAN_Model import Diff_GAN_Model
from GAN_Model import GAN_Model
from helpers.helpers import loadVocab






def main():
    # Paramters
    input_file = "data/Fortunes/data_small.txt"
    vocab_file = "vocab_fortunes.csv"
    
    # Saving/Loading paramters
    saveDir = "models"
    genSaveFile = "gen_model.pkl"
    discSaveFile = "disc_model.pkl"
    trainGraphFile = "trainGraph.png"
    
    loadDir = "models"
    genLoadFile = "gen_model - 50.pkl"
    discLoadFile = "disc_model - 50.pkl"
    
    
    
    ### Load in the data ###
    sentences = []
    with open(input_file, "r") as file:
        for line in file:
            sentences.append(line.strip())
    
    
    ### Load in the vocab ###    
    vocab = loadVocab(vocab_file)
    
    
    ### Create the model ###
    
    # Model paramters
    M_gen = 2
    N_gen = 2
    N_disc = 2
    batchSize = 10
    embedding_size = 20
    sequence_length = 64
    num_heads = 2
    
    # Training parameters
    trainingMode = "gan" # How should the models be trained ("gan" or "diff")
    pooling = "avg" # Pooling mode for the discriminator blocks ("avg", "max", or "none")
    embed_mode = "norm" # Embedding mode for the generator ("norm" or "custom")
    alpha = 0.0001
    Beta1 = 0 # Adam beta 1 term
    Beta2 = 0.9 # Adam beta 2 term
    Lambda = 10 # Lambda value used for gradient penalty in disc loss
    device = "cpu"  # cpu, partgpu, or fullgpu
    epochs = 50000
    trainingRatio = [1, 5] #Number of epochs to train the generator (0) vs the discriminator (1)
    decRatRate = -1 # Decrease the ratio after every decRatRate steps (-1 for not decrease)
    saveSteps = 10 # Number of steps until the model is saved
    
    # Diffusion GAN parameters (if used)
    Beta_0 = 0.0001 # Lowest possible Beta value, when t is 0
    Beta_T = 0.02 # Highest possible Beta value, when t is T
    T_min = 5 # Min diffusion steps when corrupting the data
    T_max = 1000 # Max diffusion steps when corrupting the data
    sigma = 0.05 # Standard deviation of the noise to add to the data
    d_target = 0.6 # Term used for the T scheduler denoting if the T change should
                   # be positive of negative depending on the disc output
    C = 5 # Constant for the T scheduler multiplying the change of T
    
    # Create the model
    if trainingMode.lower() == "diff":
        model = Diff_GAN_Model(vocab, M_gen, N_gen, N_disc, batchSize, 
                embedding_size, sequence_length, num_heads,
                trainingRatio, decRatRate, pooling, 
                embed_mode, alpha, Lambda,
                Beta1, Beta2, device, saveSteps, saveDir, 
                genSaveFile, discSaveFile, trainGraphFile,
                Beta_0, Beta_T, T_min, T_max, sigma, d_target, C)
    else:
        model = GAN_Model(vocab, M_gen, N_gen, N_disc, batchSize, 
                    embedding_size, sequence_length, num_heads,
                    trainingRatio, decRatRate, pooling, 
                    embed_mode, alpha, Lambda,
                    Beta1, Beta2, device, saveSteps, saveDir, 
                    genSaveFile, discSaveFile, trainGraphFile)
    
    
    ### Training The Model ###
    #model.loadModels(loadDir, genLoadFile, discLoadFile)
    model.train_model(sentences, epochs)
    #model.loadModels(loadDir, genLoadFile, discLoadFile)
    print()
    
    
    ### Model Saving and Predictions ###
    noise = torch.rand((sequence_length), requires_grad=False)
    out = model.generator(noise)
    #model.saveModels(saveDir, genSaveFile, discSaveFile)
    for i in out:
        print(vocab[i.item()], end=" ")
    
    
main()
