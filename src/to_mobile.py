# Convert a pickled file to a mobile version of that file



import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from Generator import Generator
import os




def main():
    in_dir = "models"
    in_file = "model.pkl"
    out_dir = "models_mobile"
    out_file = "model.ptl"
    

    model = Generator()
    model.loadModel(in_dir, in_file)
    model.eval()
    traced_script_module = torch.jit.trace(model, torch.tensor(0))
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(out_dir + os.sep + out_file)
    
    
main()