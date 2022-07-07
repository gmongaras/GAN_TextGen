import torch
from Generator import Generator
from Model import Model
from helpers import loadVocab






def main():
    # Paramters
    input_file = "data/data_small.txt"
    vocab_file = "vocab.csv"
    
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
    
    # Create the model
    model = Model(vocab, M_gen, N_gen, N_disc, batchSize, 
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
