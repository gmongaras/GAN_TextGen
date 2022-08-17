import torch
from torch import nn
from .models.BothModel import BothModel



cpu = torch.device('cpu')
try:
    if torch.has_mps:
        gpu = torch.device("mps")
    else:
        gpu = torch.device('cuda:0')
except:
    gpu = torch.device('cuda:0')






# This model is a combination of both a transformer and a LSTM.
# It is optimized to predict characters, not words.
class Both(nn.Module):
    # input_size (E) - The embedding size of the input.
    # vocab - The vocab mapping from a character to an integer
    # dropout - Rate to apply dropout to the model
    # device - Deivce to put the model on
    # saveDir - Directory to save model to
    # saveFile - File to save model to
    # num_heads - The number of heads for the Transformer model
    # hidden_size (H) - The size of the hidden state for the LSTM
    # T - Number of transformer blocks
    # L - The number of layers in the LSTM
    # W - max size of each word to encode
    def __init__(self, input_size, vocab, dropout, device, saveDir, saveFile, num_heads, hidden_size, T, L, W):
        super(Both, self).__init__()
        
        # Convert the device to a torch device
        if device.lower() == "fullgpu":
            if torch.cuda.is_available():
                dev = device.lower()
                device = torch.device('cuda:0')
            elif torch.has_mps == True:
                dev = "mps"
                device = torch.device('mps')
            else:
                dev = "cpu"
                print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
                device = torch.device('cpu')
        else:
            dev = device.lower()
            device = torch.device('cpu')
        self.device = device
        self.dev = dev
        
        self.vocab = vocab
        self.input_size = input_size
        self.W = W
        
        # Create the model
        if dev == "partgpu":
            self.model = BothModel(input_size, vocab, dropout, gpu, saveDir, saveFile, num_heads, hidden_size, T, L, W)
        else:
            self.model = BothModel(input_size, vocab, dropout, device, saveDir, saveFile, num_heads, hidden_size, T, L, W)
        
        # Optimizer
        self.optim = torch.optim.Adam(self.parameters())
        
        # Loss function to optimize this model.
        # Note that the model is being optimized to
        # generate integer representations of characters
        self.loss = torch.nn.MSELoss(reduction="mean")
    
    
    
    
    # Input:
    #   x - A tensor of shape (N, S, E) where each sequence is a
    #       sentence of encoded words.
    # Output:
    #   A tensor of shape (N, S, W) where each output along the
    #   S dimension is the predicted output for that timestep, S.
    #   Each vector of length W is the predicted word in numerical
    #   form.
    def forward(self, x):
        return self.model(x)
    
    
    
    # Train the model given a batch of words#
    # Input:
    #   X - A tensor of shape (N, S, W, V) where each sequence is a
    #       sentence of words and each character is a one-hot
    #       representation of that character
    #   epochs - Number of epochs to train model for
    #   batchSize - Size of each minibatch
    #   saveSteps - Number of steps until the model is saved
    def trainModel(self, y, epochs, batchSize, saveSteps):
        # Ensure the data are float tensors and on the correct device
        y = y.float().to(self.device)
        
        # Create batch data
        y_batches = torch.split(y, batchSize)
        N = y.shape[0]
        S = y.shape[1]
        del y


        # Train the model
        self.model.train()
        for epoch in range(1, epochs+1):
            # Iterate over all batches
            for b in range(len(y_batches)):
                # Get the data
                y_b = y_batches[b]
                
                # If the device is partgpu, then put the data on the
                # gpu
                if self.dev == "partgpu":
                    y_b = y_b.to(gpu)
                    
                # The predictions from the model
                preds = []
                
                # The inputs into the model. Initialize it
                # as all start words
                inputs = self.model.CharToWord_linear2(torch.tensor([self.vocab["¶"]] + [self.vocab["↔"] for i in range(0, self.W-1)], dtype=torch.float32, device=(self.device if self.dev != "partgpu" else gpu)).unsqueeze(0).unsqueeze(0).expand(N, -1, -1))
                    
                # Iterate over each part of the sequence
                for S in range(1, S+1):
                    # Get the predictions up to this point
                    pred = self.forward(inputs)
                    
                    # Encode the predictions. Save the
                    # intermediate output
                    pred = self.model.CharToWord_linear1(pred).squeeze()
                    preds.append(pred)
                    pred = self.model.CharToWord_linear2(pred).unsqueeze(1)
                    
                    # Add the output to the input
                    inputs = torch.cat((inputs, pred), dim=1)

                # Get the loss
                loss = self.loss(torch.stack(preds).permute(1, 0, 2), y_b).mean()

                # Update the model
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                
            # Save the model
            if epoch%saveSteps == 0:
                self.saveModel(self.saveDir, self.saveFile, epoch)

            print(f"Epoch #{epoch}     Loss: {loss}")