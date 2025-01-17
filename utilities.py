import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Device setup (CUDA or CPU)
def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is {device}.')
    return device


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_regression_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience, scheduler):
    # Move model to the specified device
    model.to(device)

    # To track the history of training losses
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    early_counter = 0  # early-stopping counter
    best_model_state = None
    # mae_scores = []
    mse_values = []
    best_r2 = -float('inf')
    for epoch in range(num_epochs):
        model.train()  # Make model into training mode
        train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            assert outputs.shape == targets.shape, f"Output shape {outputs.shape} doesn't match target shape {targets.shape}"
            loss = criterion(outputs, targets)

            train_loss += loss.item() * inputs.size(0) # x batch_size to account for the loss that is the avg per batch

            loss.backward()
            optimizer.step()

        # Calculate and store average loss
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation step
        model.eval() #turn into evaluation mode
        val_loss = 0
        r2_values = []

        with torch.no_grad(): #disable gradient calc
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputsv = model(inputs)        #v from validation
                lossv = criterion(outputsv, targets)
                val_loss += lossv.item() * inputs.size(0)

                #r^2 calculation
                for col in range(outputsv.shape[1]):
                    pred_col = outputsv[:, col].cpu().numpy()  # Get predictions for the current column
                    target_col = targets[:, col].cpu().numpy()  # Get targets for the current column

                    r2 = r2_score(target_col, pred_col)
                    r2_values.append(r2)
                    # Calculate MSE for the current column
                    mse = mean_squared_error(target_col, pred_col)
                    mse_values.append(mse)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        mean_r2 = np.mean(r2_values)
        mean_mse = np.mean(mse_values)  # Calculate mean MSE for all columns

        # Adjust learning rate with ReduceLROnPlateau
        if scheduler:
            scheduler.step(val_loss)
            current_lr = scheduler.get_last_lr()[0]  # Get the learning rate of the first group
        else:
            current_lr = optimizer.param_groups[0]['lr']  # Fallback if no scheduler is provided

        if epoch % 10 == 0 or epoch == num_epochs-1: # print metrics
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Learning Rate: {current_lr:.4e}" )
            for col in range(outputs.shape[1]):
                print(f"  R^2 for column {col + 1}: {r2_values[col]:.4f}")
            print(f"  Mean R²: {mean_r2:.4f}")
            for col in range(outputs.shape[1]):
                print(f"  MSE for column {col + 1}: {mse_values[col]:.4f}")
            print(f"  Mean MSE for all columns: {mean_mse:.4f}")

        # save the model if mean R^2 improves
        # if mean_r2 > best_r2:
        #     best_r2 = mean_r2
        #     best_model_state = model.state_dict()

        # Early stopping implementation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0  # reset the counter if validation loss improves
            best_model_state = model.state_dict()
        else:
            early_counter += 1
            if early_counter >= patience:
                print(f"Early stopping triggered at {epoch + 1} epoch")
                model.load_state_dict(best_model_state)
                break

    return model, train_losses, val_losses


def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputst = model(inputs)
            losst = criterion(outputst, targets)
            test_loss += losst.item() * inputs.size(0)
            all_predictions.append(outputst.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    test_loss /= len(test_loader.dataset)
    return test_loss, all_predictions, all_targets


class Model(nn.Module):
    # input layer ( 2 features, Power Pressure) --> Hidden Layer 1 (H1) --> H2 --> Output (25 outputs)
    def __init__(self, in_features=2, h1=10, h2=10, out_features=10):
        super().__init__()  # instantiate our nn.Module, always have to do it
        self.fc1 = nn.Linear(in_features, h1) # we suppose fully connected layer (fc-> fully connected)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    # we need now the function to move everything forward
    def forward(self, x):
        # we choose relu
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.out(x)
        return x


def unscale(data, column_names, scaling_info):
    unscaled_data = data.copy()
    num_columns = data.shape[1]
    for i, column in enumerate(column_names):
        if i >= num_columns:  # Check if index exceeds the number of columns
            print(f"Warning: Index {i} is out of bounds for data with {num_columns} columns.")
            break
        mean = scaling_info[column]['mean']
        std = scaling_info[column]['std']
        unscaled_data[:, i] = (data[:, i] * std) + mean  # Reverse normalization
    return unscaled_data

#manual loss functions
class calculate_weighted_mse:

    def __init__(self, reduction): #i can choose for sum of errors or mean of errors or none for elemntwise
        self.reduction = reduction

    def forward(self, input, target):
        # Step 2: Check that input and target have the same shape
        if input.shape != target.shape:
            raise ValueError("Input and target must have the same shape")

        # Step 3: Compute the squared differences (squared error)
        squared_error = (input - target) ** 2
        #weighted_squared_error = squared_error * weights

        # Step 4: Apply the reduction method (mean, sum, or none)
        if self.reduction == 'mean':
            return squared_error.mean()  # Mean of squared errors
        elif self.reduction == 'sum':
            return squared_error.sum()  # Sum of squared errors
        else:
            return squared_error  # (element-wise squared error)




    # weights_tensor = torch.tensor([
    #     0.093312122, 0.102769082, 0.105229524, 0.118937088, 0.13627972,
    #     0.144720322, 0.153479654, 0.107621811, 0.035236323, 0.002414354
    # ], dtype=torch.float32)
    # Apply the weights to each column's squared error

class calculate_huber_loss:
    def __init__(self, delta: float = 1.0):

        #delta (float): The threshold at which to switch between squared loss and absolute loss.

        super(calculate_huber_loss, self).__init__()
        self.delta = delta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Compute the absolute error
        error = torch.abs(y_true - y_pred)

        # Calculate Huber loss based on the threshold delta
        loss = torch.where(error <= self.delta,
                           0.5 * error ** 2,  # Squared loss
                           self.delta * (error - 0.5 * self.delta))  # Absolute loss

        return loss.mean()  # Return the mean loss over the batch
#dynamic neural network to be used at the hyperparameter search
class Model_dynamic(nn.Module):
    def __init__(self, h1, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()  # List to hold layers

        # First layer
        self.layers.append(nn.Linear(2, h1))

        # Add hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(h1, h1))  # Each hidden layer has h1 neurons

        # Output layer
        self.out = nn.Linear(h1, 10)

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.elu(self.layers[i](x))  # Apply ELU after each layer
        x = self.out(x)  # Output layer
        return x