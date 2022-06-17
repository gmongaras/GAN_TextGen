# Convert a pickled file to a mobile version of that file



import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from Generator import Generator
import os
from helpers import loadVocab




def main():
    in_dir = "models"
    in_file = "gen_model.pkl"
    out_dir = "models_mobile"
    out_file = "model.ptl"
    
    vocab_file = "vocab.csv"
    

    # Model paramters
    vocab = loadVocab(vocab_file)
    M = 2
    N = 2
    batchSize = 1
    embedding_size = 10
    sequence_length = 64
    num_heads = 2


    with torch.no_grad():
        model = Generator(vocab, M, N, batchSize, embedding_size, sequence_length, num_heads, torch.device("cpu"))
        model.loadModel(in_dir, in_file)
        model.eval()
        noise = torch.rand((sequence_length, embedding_size), requires_grad=False)
        traced_script_module = torch.jit.trace(model, noise)
        traced_script_module_optimized = optimize_for_mobile(traced_script_module)
        traced_script_module_optimized._save_for_lite_interpreter(out_dir + os.sep + out_file)
    
    
main()