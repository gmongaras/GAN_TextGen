import torch
from Generator import Generator
from Model import Model
from helpers import loadVocab






def main():
    # Paramters
    input_file = "data/data.txt"
    vocab_file = "vocab.csv"
    
    # Saving/Loading paramters
    saveDir = "models"
    genSaveFile = "gen_model.pkl"
    discSaveFile = "disc_model.pkl"
    trainGraphFile = "trainGraph.png"
    
    loadDir = "models"
    genLoadFile = "gen_model.pkl"
    discLoadFile = "disc_model.pkl"
    
    
    
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
    embedding_size = 10
    sequence_length = 128
    num_heads = 2
    alpha = 0.00005
    device = torch.device("cpu")
    epochs = 50
    trainingRatio = [1, 5] #Number of epochs to train the generator (0) vs the discriminator (1)
    decRatRate = 10 # Decrease the ratio after every decRatRate steps
    saveSteps = 10 # Number of steps until the model is saved
    
    # Create the model
    model = Model(vocab, M_gen, N_gen, N_disc, batchSize, 
                  embedding_size, sequence_length, num_heads,
                  trainingRatio, decRatRate, alpha, device,
                  saveSteps, saveDir, genSaveFile, discSaveFile,
                  trainGraphFile)
    
    
    ### Training The Model ###
    model.train_model(sentences, epochs)
    #model.loadModels(loadDir, genLoadFile, discLoadFile)
    print()
    
    
    ### Model Saving and Predictions ###
    noise = torch.rand((sequence_length, embedding_size), requires_grad=False)
    out = model.generator(noise)
    #model.saveModels(saveDir, genSaveFile, discSaveFile)
    for i in out:
        print(vocab[i.item()], end=" ")
    
    
main()