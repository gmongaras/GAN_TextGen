from torch.fft import fftn
from torch import nn




# This block is used to embed one-hot encoded sentences
# into a better more encoded form
class CustomEmb(nn.Module):
    # Initialize the embeddings modules
    # Inputs:
    #   inEmb - The input embedding size (vocab size)
    #   outEmb - The output embedding size
    #   hiddenNodes - List of integers representing the number
    #                 of nodes for each hidden linear layer.
    #           Ex: [1, 2, 3] would have 1 node in the first
    #               hidden layer, 2 in the second, and 3 in the third.
    #   useFFT - True to use an FFT transformation, False otherwise
    #   device - Device to put this block on
    def __init__(self, inEmb, outEmb, hiddenNodes, useFFT, device):
        super(CustomEmb, self).__init__()
        
        self.useFFT = useFFT
        
        # Create the linear layers
        self.linear = [nn.Linear(inEmb, hiddenNodes[0])] +\
                      [nn.Linear(hiddenNodes[i-1], hiddenNodes[i]) for i in range(1, len(hiddenNodes))] +\
                      [nn.Linear(hiddenNodes[-1], outEmb)]
        self.linear = nn.Sequential(*self.linear).to(device)
    
    
    # Input:
    #   Tensor of shape (N, S, inEmb)
    # Output:
    #   Tensor of shape (N, S, outEmb)
    def forward(self, X):
        # Use a FFT transformation on the data if specified
        if self.useFFT:
            X = fftn(X, dim=-1).real
        
        # Send the data through the linear layers
        X = self.linear(X)
        
        # Return the output
        return X