# Convert a pickled file to a mobile version of that file



import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from .models.Generator import Generator
import os
from ..helpers.helpers import loadVocab




def main():
    in_dir = "models"
    in_file = "gen_model.pkl"
    out_dir = "models_mobile"
    out_file = "model.ptl"
    
    vocab_file = "vocab_fortunes.csv"
    

    # Model paramters
    vocab = loadVocab(vocab_file)
    M = 2
    B = 2
    O = 2
    gausNoise = True
    batchSize = 1
    embedding_size = 20
    sequence_length = 64
    num_heads = 2
    embed_mode = "norm"


    with torch.no_grad():
        model = Generator(vocab, M, B, O, gausNoise, batchSize, embedding_size, sequence_length, num_heads, embed_mode, torch.device("cpu"))
        model.loadModel(in_dir, in_file)
        model.eval()
        noise = torch.rand((sequence_length), requires_grad=False)
        traced_script_module = torch.jit.trace(model, noise)
        traced_script_module_optimized = optimize_for_mobile(traced_script_module)
        traced_script_module_optimized._save_for_lite_interpreter(out_dir + os.sep + out_file)
    
    
main()
