import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utilities import train_regression_model, Model


def objective(trial, train_dataset, val_dataset, device, num_of_epochs):
    # Hyperparameters to optimize
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)  # Log scale search for learning rate
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])  # Discrete search for batch size
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)  # Log scale search for weight decay

    # You can add here different layers and their sizes, so that we also tune the architecture.

    # Create DataLoader with the current batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # Create model instance
    model = Model()

    # Define the criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    # Train the model
    trained_model, train_losses, val_losses = train_regression_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_of_epochs,
        device=device,
        patience=10,
        scheduler=scheduler
    )
    # Get the validation loss for the best model
    '''
    This is not correct. The last one is not the smallest one.
    final_val_loss = val_losses[-1]'''

    min_val_loss = min(val_losses)

    return min_val_loss  # Minimize the validation loss
