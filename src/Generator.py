from torch import nn
import torch
from torch.nn.modules.activation import Softmax
import os




class Generator(nn.Module):
    def __init__(self,):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, 3),
            nn.Softmax(),
        )
    
    
    
    
    # Input:
    #   Nothing
    # Output:
    #   A string of max length 256 words
    def forward(self, X):
        noise = torch.rand((10, 256))
        
        return torch.argmax(self.model(noise), dim=-1)
    
    
    # Save the model
    def saveModel(self, saveDir, saveFile):
        # Check if the directory exists. If it doesn't
        # create it
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)
        
        # Save the model
        torch.save(self.state_dict(), saveDir + os.sep + saveFile)
    
    
    # Load the model
    def loadModel(self, loadDir, loadFile):
        self.load_state_dict(torch.load(loadDir + os.sep + loadFile))