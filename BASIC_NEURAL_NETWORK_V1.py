import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import numpy as np
import matplotlib.pyplot as plt

from dataset_class_V2 import MergedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device is {device}.')


# This function sets all the required seeds to ensure the experiments are reproducible. Use it in your main code-file.
def set_seed(seed):
    # Set the seed for generating random numbers in Python
    random.seed(seed)
    # Set the seed for generating random numbers in NumPy
    np.random.seed(seed)
    # Set the seed for generating random numbers in PyTorch (CPU)
    torch.manual_seed(seed)
    # If you are using GPUs, set the seed for generating random numbers on all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure that all operations on GPU are deterministic (if possible)
    torch.backends.cudnn.deterministic = True
    # Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware
    torch.backends.cudnn.benchmark = False


seed_num = 41
set_seed(seed_num)


# Create a model class that inherits nn.Module
class Model(nn.Module):
    # input layer ( 2 features, Power Pressure) --> Hidden Layer 1 (H1) --> H2 --> Output (25 outputs)
    def __init__(self, in_features=2, h1=10, h2=10, out_features=25):
        super().__init__()  # instantiate our nn.Module, always have to do it
        self.fc1 = nn.Linear(in_features, h1) # we suppose fully connected layer (fc-> fully connected)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    # we need now the function to move everything forward
    def forward(self, x):
        # we choose relu
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


# Define a generic training loop function
def train_regression_model(model, train_loader, criterion, optimizer, num_epochs, device):
    # Move model to the specified device
    model.to(device)

    # To track the history of training losses
    train_losses = []

    for epoch in range(num_epochs):
        model.train()  # Make model into training mode
        train_loss = 0

        # YOU HAVE TO USE THE DATALOADER TO PERFORM BATCH TRAINING!!! NOT JUST THROW IN X_Train
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            train_loss += loss.item() * inputs.size(0) # x batch_size to account for the loss that is the avg per batch

            loss.backward()
            optimizer.step()

        # Calculate and store average loss
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        if epoch % 10 == 0 or epoch == num_epochs-1:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f}")

    return model, train_losses


# create an instance for the model
basic_model = Model()

# Use the classes you created to make your Dataset and Data
train_data = MergedDataset()
dataloader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)

# MEASURE LOSS
criterion = nn.MSELoss()
# CHOOSE ADAM OPTIMIZER
optimizer = torch.optim.Adam(basic_model.parameters(), lr=0.001)
epochs = 100

trained_model, losses = train_regression_model(model=basic_model,
                                               train_loader=dataloader,
                                               criterion=criterion,
                                               optimizer=optimizer,
                                               num_epochs=epochs,
                                               device=device)

# Plot Training loss at each Epoch
plt.plot(list(range(epochs)), losses, label="Training Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Training Loss Progression")
plt.legend()
plt.show()

