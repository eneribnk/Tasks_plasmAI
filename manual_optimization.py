import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
from dataset_class_V2 import MergedDataset
from utilities import setup_device, set_seed, train_regression_model, Model_dynamic
from torch.utils.data import random_split

device = setup_device()


# This function sets all the required seeds to ensure the experiments are reproducible. Use it in your main code-file.
seed_num = 41
set_seed(seed_num)

start_time = time.time()

dataset = MergedDataset('train_data_no_head_outer_corner_O2.csv')
dataset_test = MergedDataset('test_data_no_head_outer_corner_O2.csv')
# Set the sizes for training, validation, and  testing
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size    # 20% for validation


train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
learning_rates = np.logspace(-5, -3, num=10)
batch_sizes = [32, 64, 128]
weight_decays = np.logspace(-6, -1, num=10)
h1_values = [5, 10, 15, 20]
num_layers_values = [1, 2, 3]

# Initialize the best validation loss and corresponding hyperparameters
best_val_loss = float('inf')
best_params = None
test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)

for h1 in h1_values:
    for num_layers in num_layers_values:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for weight_decay in weight_decays:
                    print(f"Training with LR={lr}, Batch Size={batch_size}, weight decay={weight_decay}")

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                    # Create an instance for the model
                    basic_model = Model_dynamic(h1=h1, num_layers=num_layers)
                    criterion = nn.SmoothL1Loss()
                    # Use Adam optimizer with L2 regularization (weight decay)
                    optimizer = torch.optim.Adam(basic_model.parameters(), lr=lr, weight_decay=weight_decay)
                    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

                    epochs = 600

                    trained_model, losses, val_losses = train_regression_model(model=basic_model,
                                                                               train_loader=train_loader,
                                                                               val_loader=val_loader,
                                                                               criterion=criterion,
                                                                               optimizer=optimizer,
                                                                               num_epochs=epochs,
                                                                               device=device,
                                                                               patience=5,
                                                                               scheduler=None)
                    min_val_loss = min(val_losses) # choose the final val_loss, when the training stops
                    print(f"Validation Loss: {min_val_loss:.4f}")

                    if min_val_loss < best_val_loss:
                        best_val_loss = min_val_loss
                        best_h1 = h1
                        best_num_layers = num_layers
                        best_params = {'lr': lr, 'batch_size': batch_size, 'weight_decay': weight_decay,
                                       'h1': h1, 'layers': num_layers}
                        best_model = trained_model  # Save the best model


model_save_path = 'trained_model1_O2_optimized.pth'
torch.save(trained_model.state_dict(), model_save_path)
print(f"model saved to {model_save_path}")

# Plot Training loss and validation loss at each Epoch
plt.plot(list(range(len(losses))), losses, label="Training Loss")
plt.plot(list(range(len(losses))), val_losses, label="Validation Loss")
plt.ylabel("Total Loss")
plt.xlabel("Epoch")
plt.title("Training & Validation Loss Progression")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.rcParams.update({'font.size': 22})
plt.show()
