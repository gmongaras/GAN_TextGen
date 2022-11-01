import torch
from .models.Generator import Generator
import click
from typing import Optional






@click.command()

# Saved state loading parameters
@click.option("--loadDir", "loadDir", type=str, help="Path to load the model from", required=True)
@click.option("--loadDefFile", "loadDefFile", type=str, help="File with generator defaults", required=True)
@click.option("--loadFile", "loadFile", type=str, help="File to load the generator model from", required=True)

# Other parameters
@click.option("--batchSize", "batchSize", type=int, default=1, help="Batch size used to train the model", required=False)
@click.option("--device", "device", type=str, default="gpu", help="Device to put the model on (\"cpu\", \"gpu\")", required=False)

def train(
    loadDir: str,
    loadDefFile: str,
    loadFile: str,
    batchSize: Optional[int],
    device: Optional[str],
    ):

    # Device for the model
    if not torch.cuda.is_available() and device.lower() == "gpu":
        print("Cuda GPU not availble, defaulting to CPU")
        device = torch.device("cpu")
    elif device.lower() == "gpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    ### Create the model ###
    model = Generator({"a": 1}, 1, 1, 1, "unif", 1, batchSize, 1, 1, 1, device)
    
    
    # Loading and inference
    model.loadModel(loadDir, loadFile, loadDefFile)

    # Prediction
    preds = model.generate(batchSize)
    for b in range(len(preds)):
        print(preds[b])
        print(len(preds[b].split(" ")))
    
    
if __name__ == "__main__": 
    train()
