import torch
from .Diff_GAN_Model import Diff_GAN_Model
from .GAN_Model import GAN_Model
from .Norm_Model import Norm_Model
from ..helpers.helpers import loadVocab
import click
from typing import Optional






@click.command()

# File loading parameters
@click.option("--input_file", "input_file", type=str, default="data/Text2/data2.txt", help="Location of the txt file with sentences to train the model", required=False)
@click.option("--vocab_file", "vocab_file", type=str, default="vocab_text2.csv", help="Location of the csv file storing the vocabulary dictionary", required=False)
@click.option("--num_to_load", "num_to_load", type=int, default=-1, help="Number of sentences to load from the input file (-1 for all)", required=False)


# Model/graph saving parameters
@click.option("--saveDir", "saveDir", type=str, default="models", help="Path to save models to", required=False)
@click.option("--genSaveFile", "genSaveFile", type=str, default="gen_model.pkl", help="File to save generator models to", required=False)
@click.option("--discSaveFile", "discSaveFile", type=str, default="disc_model.pkl", help="File to save discriminator models to", required=False)
@click.option("--trainGraphFile", "trainGraphFile", type=str, default="trainGraph.png", help="File to save the loss graph to", required=False)


# Saved state loading parameters
@click.option("--saveDir", "saveDir", type=str, default="models", help="Path to save models to", required=False)


# General model parameters
@click.option("--M_gen", "M_gen", type=int, default=5, help="Number of input noise embedding blocks", required=False)
@click.option("--B_gen", "B_gen", type=int, default=4, help="Number of transformer blocks to encode the input sequence", required=False)
@click.option("--O_gen", "O_gen", type=int, default=4, help="Number of transformer blocks to get the output sequence", required=False)
@click.option("--embedding_size_gen", "embedding_size_gen", type=int, default=64, help="Word embedding size for the generator", required=False)
@click.option("--T_disc", "T_disc", type=int, default=2, help="Number of transformer blocks in each discriminator block", required=False)
@click.option("--B_disc", "B_disc", type=int, default=6, help="Number of discriminator blocks in the discriminator", required=False)
@click.option("--O_disc", "O_disc", type=int, default=6, help="Number of output transformer blocks in the discrimiantor", required=False)
@click.option("--embedding_size_disc", "embedding_size_disc", type=int, default=64, help="Word embedding size for the discriminator", required=False)
@click.option("--hiddenSize", "hiddenSize", type=int, default=512, help="Hidden linear size in the transformer blocks", required=False)
@click.option("--batchSize", "batchSize", type=int, default=64, help="Batch size used to train the model", required=False)
@click.option("--sequence_length", "sequence_length", type=int, default=64, help="Max number of words in sentence to train the model with", required=False)
@click.option("--num_heads", "num_heads", type=int, default=8, help="Number of heads in each MHA block", required=False)
@click.option("--noiseDist", "noiseDist", type=str, default="unif", help="Distribution to sample noise from. Can be one of (\"norm\", \"unif\", \"trunc\" (for truncated normal))", required=False)
@click.option("--costSlope", "costSlope", type=float, default=0.1, help="The slope of the GLS-GAN cost function", required=False)

@click.option("--pooling", "pooling", type=str, default="none", help="Pooling mode for the discriminator blocks (\"avg\" to use average pooling, \"max\" to use max pooling, or \"none\" to use no pooling)", required=False)
@click.option("--alpha", "alpha", type=float, default=0.0001, help="Model learning rate", required=False)
@click.option("--Beta1", "Beta1", type=float, default=0.0, help="Adam beta 1 term", required=False)
@click.option("--Beta2", "Beta2", type=float, default=0.9, help="Adam beta 2 term", required=False)
@click.option("--device", "device", type=str, default="partgpu", help="Device to put the model on (\"cpu\", \"fullgpu\", or \"partgpu\")", required=False)
@click.option("--epochs", "epochs", type=int, default=100000, help="Number of epochs to train the model", required=False)
@click.option("--n_D", "n_D", type=int, default=5, help="Number of times to train the discriminator more than the generator for each epoch", required=False)
@click.option("--saveSteps", "saveSteps", type=int, default=100, help="Number of steps until the model is saved", required=False)
@click.option("--loadInEpoch", "loadInEpoch", type=bool, default=False, help="Should the data be loaded in as needed instead of before training? (True if so, False to load before training)", required=False)
@click.option("--delWhenLoaded", "delWhenLoaded", type=bool, default=True, help="Delete the data as it's loaded in to free allocated memory? Note: This is automatically False if loadInEpoch is True", required=False)


# GAN parameters
@click.option("--Lambda", "Lambda", type=int, default=10, help="Lambda value used for gradient penalty in disc loss", required=False)
@click.option("--dynamic_n", "dynamic_n", type=bool, default=False, help="True to dynamically change the number of times to train the models. False otherwise", required=False)
@click.option("--Lambda_n", "Lambda_n", type=int, default=1, help="(Only used if dynamic_n is True) Amount to scale the generator over the discriminator to give the generator a higher weight (when >1) or the discriminator a higher weight (when <1)", required=False)
@click.option("--HideAfterEnd", "HideAfterEnd", type=bool, default=False, help="True to hide any tokens after the <END> token in the discriminator MHA with a mask, False to keep these tokens visibile", required=False)

def train(
    input_file: Optional[str],
    vocab_file: Optional[str],
    num_to_load: Optional[int],

    saveDir: Optional[str],
    genSaveFile: Optional[str],
    discSaveFile: Optional[str],
    trainGraphFile: Optional[str],
    M_gen: Optional[int],
    B_gen: Optional[int],
    O_gen: Optional[int],
    embedding_size_gen: Optional[int],
    T_disc: Optional[int],
    B_disc: Optional[int],
    O_disc: Optional[int],
    embedding_size_disc: Optional[int],
    hiddenSize: Optional[int],
    batchSize: Optional[int],
    sequence_length: Optional[int],
    num_heads: Optional[int],
    noiseDist: Optional[str],
    costSlope: Optional[float],

    pooling: Optional[str],
    alpha: Optional[float],
    Beta1: Optional[float],
    Beta2: Optional[float],
    device: Optional[str],
    epochs: Optional[int],
    n_D: Optional[int],
    saveSteps: Optional[int],
    loadInEpoch: Optional[bool],
    delWhenLoaded: Optional[bool],

    Lambda: Optional[int],
    dynamic_n: Optional[bool],
    Lambda_n: Optional[int],
    HideAfterEnd: Optional[bool],
    ):


    # Assertion for the number of sentences to load
    assert num_to_load > batchSize*(n_D+1) or num_to_load == -1, f"Number of sentences to load must be greater than batchSize*(n_D+1) = {batchSize*(n_D+1)}"
    
    
    ### Load in the data ###
    sentences = []
    i = 0
    with open(input_file, "r", encoding='utf-8') as file:
        for line in file:
            if (i == num_to_load and num_to_load != -1):
                break
            i += 1
            sentences.append(line.lower().strip())
    
    
    ### Load in the vocab ###    
    vocab = loadVocab(vocab_file)
    
    
    ### Create the model ###
    model = GAN_Model(vocab, M_gen, B_gen, O_gen,
            T_disc, B_disc, O_disc, hiddenSize, noiseDist, costSlope,
            batchSize, embedding_size_gen, embedding_size_disc,
            sequence_length, num_heads, dynamic_n, Lambda_n, HideAfterEnd,
            n_D, pooling, alpha, Lambda,
            Beta1, Beta2, device, saveSteps, saveDir, 
            genSaveFile, discSaveFile, trainGraphFile,
            loadInEpoch, delWhenLoaded)
    
    
    ### Training The Model ###
    #model.loadModels("models", "gen_model - 100.pkl", "disc_model - 100.pkl")
    model.train_model(sentences, epochs)
    print()
    
    
    ### Model Saving and Predictions ###
    with torch.no_grad():
        out = model.generator.forward(training=False)
        lens = model.generator.forward_lens(out)
    out = torch.argmax(out, dim=-1)[0]
    lens = torch.argmax(lens, dim=-1)[0]+1
    for i in out:
        print(vocab[i.item()], end=" ")
    print()
    print(f"lens: {lens}")
    
    
if __name__ == "__main__": 
    train()
