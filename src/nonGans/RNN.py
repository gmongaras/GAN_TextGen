import torch
from torch import nn
from .models.LSTM import LSTM, LSTM_torch
import os



cpu = torch.device('cpu')
try:
    if torch.has_mps:
        gpu = torch.device("mps")
    else:
        gpu = torch.device('cuda:0')
except:
    gpu = torch.device('cuda:0')




class RNN(nn.Module):
    # mode - What should the model output? ("word" or "char")
    # input_size (E) - The embedding size of the input.
    # hidden_size (H) - The size of the hidden state
    # vocab_size (V) - The size of the vocab to predict from
    # layers - The number of LSTM blocks stacked ontop of each other
    # dropout - Rate to apply dropout to the model
    # device - Deivce to put the model on
    # saveDir - Directory to save model to
    # saveFile - File to save model to
    def __init__(self, mode, input_size, hidden_size, vocab_size, layers, dropout, device, saveDir, saveFile):
        super(RNN, self).__init__()
        
        # Saved variables
        self.mode = mode.lower()
        self.saveDir = saveDir
        self.saveFile = saveFile
        
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
        
        # Use an LSTM
        if dev == "partgpu":
            self.model = LSTM_torch(input_size, hidden_size, vocab_size, layers, dropout, gpu)
        else:
            self.model = LSTM_torch(input_size, hidden_size, vocab_size, layers, dropout, device)
        
        # Optimizer
        self.optim = torch.optim.Adam(self.parameters())
        
        # Loss function to optimize this model.
        # Note that since the loss function calculates
        # the softmax of the inputs, the output of the model
        # should not be softmax probabilities
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
    
    
    
    # Given a batch a sequences of either words or
    # characters, get predictions from the model
    # Inputs:
    #   x - A batch of sequence of inputs of shape (N, S, E)
    # Outputs:
    #   A tensor of shape (N, S, V) where each output along the
    #   S dimension is the output vector of the next item
    #   in the sequence.
    def forward(self, x):
        return self.model(x)

    
    
    # Train the model given either a batch of sentences
    # Inputs:
    #   X - Input into the model
    #   y - Expected output from the model
    #   epochs - Number of epochs to train the model for
    #   batchSize - Size of each minibatch
    #   saveSteps - Number of steps until a new model is saved
    def trainModel(self, X, y, epochs, batchSize, saveSteps):
        # If the mode is "word", train the model
        # to generate words.
        if self.mode == "word":
            self.train_word(X, y, epochs, batchSize, saveSteps)
        else:
            self.train_char(X, y, epochs, batchSize, saveSteps)

    
    
    # Train a model to generate words
    # Inputs:
    #   X - Input into the model
    #   y - Expected output from the model
    #   epochs - Number of epochs to train the model for
    #   batchSize - Size of each minibatch
    #   saveSteps - Number of steps until a new model is saved
    def train_word(self, X, y, epochs, batchSize, saveSteps):
        pass

    
    
    # Train a model to generate characters
    # Inputs:
    #   X - Input into the model
    #   y - Expected output from the model
    #   epochs - Number of epochs to train the model for
    #   batchSize - Size of each minibatch
    #   saveSteps - Number of steps until a new model is saved
    def train_char(self, X, y, epochs, batchSize, saveSteps):
        # Ensure the data are float tensors and on the correct device
        X = X.float().to(self.device)
        y = y.float().to(self.device)
        
        # Create batch data
        X_batches = torch.split(X, batchSize)
        y_batches = torch.split(y, batchSize)


        # Train the model
        self.model.train()
        for epoch in range(1, epochs+1):
            # Iterate over all batches
            for b in range(len(X_batches)):
                # Get the data
                X_b = X_batches[b]
                y_b = y_batches[b]
                
                # If the device is partgpu, then put the data on the
                # gpu
                if self.dev == "partgpu":
                    X_b = X_b.to(gpu)
                    y_b = y_b.to(gpu)

                # Get the model predictions
                preds = self.model(X_b)

                # Get the loss
                loss = self.loss(preds[:, :-1].permute(0, 2, 1), y_b[:, 1:].permute(0, 2, 1)).mean()

                # Update the model
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                break
                
            # Save the model
            if epoch%saveSteps == 0:
                self.saveModel(self.saveDir, self.saveFile, epoch)

            print(f"Epoch #{epoch}     Loss: {loss}")
    
    
    
    # Save the model
    def saveModel(self, saveDir, saveFile, epoch=None):
        # Check if the directory exists. If it doesn't
        # create it
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)
        
        if epoch == None:
            torch.save(self.state_dict(), saveDir + os.sep + saveFile)
        else:
            l = len(saveFile.split(".")[-1])+1
            saveFile = saveFile[:-l] + f" - {epoch}.pkl"
            
            torch.save(self.state_dict(), saveDir + os.sep + saveFile)
            
            # if self.trainGraphFile:
            #     fix, ax = plt.subplots()
            #     y = [i for i in range(1, len(self.genLoss)+1)]
            #     ax.plot(y, self.genLoss, label="Gen loss")
            #     ax.plot(y, self.discLoss_real, label="Disc loss real")
            #     ax.plot(y, self.discLoss_fake, label="Disc loss fake")
            #     ax.plot(y, self.discLoss, label="Disc loss combined")
            #     ax.set_title("Gen and disc loss over epochs")
            #     ax.set_xlabel("Epochs")
            #     ax.set_ylabel("Loss")
            #     ax.legend()
            #     plt.savefig(self.saveDir + os.sep + self.trainGraphFile)
            #     plt.close()
    
    
    # Load the model
    def loadModel(self, loadDir, loadFile):
        self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device))