# Import necessary libraries
import os
import random
import pickle
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import generate_series, temporal_split, mse, mae, r2, ccc
from model import HOIST_without_claim

# Set random seed for reproducibility
def seed_torch(RANDOM_SEED=123):
    """
    Seed all sources of randomness for reproducibility.

    Args:
        RANDOM_SEED (int): Seed value.
    """
    random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load data from pickle files
mob_mat = pickle.load(open('./data/mob_mat.pkl', 'rb'))             # Mobility matrix
distance_mat = pickle.load(open('./data/distance_mat.pkl', 'rb'))   # Distance matrix
covid_tensor = pickle.load(open('./data/covid_tensor.pkl', 'rb'))   # COVID-19 data tensor
hospitalizations = pickle.load(open('./data/hospitalizations.pkl', 'rb'))  # Hospitalization data
hos_tensor = pickle.load(open('./data/hos_tensor.pkl', 'rb'))       # Hospitalization tensor
county_tensor = pickle.load(open('./data/county_tensor.pkl', 'rb')) # County data tensor
feat_name = pickle.load(open('./data/feat_name.pkl', 'rb'))         # Feature names
date_range = np.array(pickle.load(open('./data/date_range.pkl', 'rb')))  # Date range

# Print shapes of the loaded data
print("mob_mat shape:", mob_mat.shape)
print("distance_mat shape:", distance_mat.shape)
print("covid_tensor shape:", covid_tensor.shape)
print("hospitalizations shape:", hospitalizations.shape)
print("hos_tensor shape:", hos_tensor.shape)
print("county_tensor shape:", county_tensor.shape)
print("feat_name:", feat_name)
print("date_range shape:", date_range.shape)

# Data preprocessing and temporal split
# Expand dimensions and concatenate tensors
covid_tensor = np.expand_dims(covid_tensor, axis=2)
X = np.concatenate([covid_tensor, hos_tensor], axis=2)  # Input features
y = hospitalizations                                    # Target variable

print("X shape:", X.shape)
print("y shape:", y.shape)

# Generate series data with specified window and prediction sizes
X, y = generate_series(X, y, window_size=35, pred_size=28)

# Process date index
date_idx = np.expand_dims(date_range, axis=0)
date_idx = np.expand_dims(date_idx, axis=2)
date_idx, _ = generate_series(date_idx, y, window_size=35, pred_size=28, date=True)

# Filter out records with zero mean
range_idx = (y.mean(1) > 0)
county_tensor = county_tensor[range_idx]
y = y[range_idx]
X = X[range_idx]
print('Number of records after filtering:', len(y))

# Adjust mobility and distance matrices
mob_mat = mob_mat[range_idx, :][:, range_idx]
distance_mat = distance_mat[range_idx, :][:, range_idx]

# Apply logarithmic transformation to target variable
y = np.log(y + 1)

# Split data into training, validation, and test sets
train_x, val_x, test_x, train_y, val_y, test_y, train_idx, val_idx, test_idx, static, mats, normalize_dict, shuffle_idx = temporal_split(
    X, y, county_tensor, [mob_mat, distance_mat], 0.2, 0.2, norm='min-max', norm_mat=True
)

# Normalized matrices
norm_mob = mats[0]
norm_dist = mats[1]

# Print shapes of the training and validation data
print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)
print("val_x shape:", val_x.shape)
print("val_y shape:", val_y.shape)

# Initialize metric lists
mae_list = []
mae_exp_list = []
mse_list = []
mse_exp_list = []
r2_list = []
r2_exp_list = []
ccc_list = []
ccc_exp_list = []

# Number of runs for averaging results
runs = 5

for k in range(runs):
    # Set random seed for each run
    seed_torch(k)
    
    # Initialize model, optimizer, and loss function
    model = HOIST_without_claim(5, [4, 5, 5], 128, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss(reduction='none')

    # Training parameters
    epochs = 300
    batch_size = 128
    min_loss = float('inf')
    min_epoch = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        val_loss = []

        # Training batches
        for i in range(0, len(train_x), batch_size):
            # Prepare batch data
            batch_x = train_x[i:i + batch_size]
            batch_y = train_y[i:i + batch_size]
            batch_static = static[i:i + batch_size]
            batch_mob = norm_mob[i:i + batch_size, :][:, i:i + batch_size]
            batch_dist = norm_dist[i:i + batch_size, :][:, i:i + batch_size]

            # Convert to tensors and move to device
            batch_x = torch.tensor(batch_x).float().to(device)
            batch_y = torch.tensor(batch_y).float().unsqueeze(-1).to(device)
            batch_static = torch.tensor(batch_static).float().to(device)
            batch_mob = torch.tensor(batch_mob).float().to(device)
            batch_dist = torch.tensor(batch_dist).float().to(device)
            batch_mat = torch.cat([batch_mob.unsqueeze(-1), batch_dist.unsqueeze(-1)], dim=2)
            cur_static = [
                batch_static[:, :4],
                batch_static[:, 4:9],
                batch_static[:, 9:14],
                batch_mat,
            ]

            # Forward pass
            optimizer.zero_grad()
            output, aux_outputs = model(batch_x, cur_static)

            # Compute losses
            N, T, F = batch_y.shape
            dist = aux_outputs[0]
            weights = aux_outputs[1]
            y_p = (weights * batch_x).sum(-1).reshape(N, T, 1) * output.detach()
            y_pi = y_p.reshape(N, 1, T)
            y_pj = y_p.reshape(1, N, T)
            y_k = ((y_pi * y_pj) * dist.reshape(N, N, 1)).sum(1).reshape(N, T, 1)
            ising_loss = loss_fn(y_p + y_k, batch_y).mean(1).mean()

            # Total loss
            loss = loss_fn(output, batch_y).mean(1).mean() + ising_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Record loss
            epoch_loss.append(loss.item())

        # Validation loop
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for i in range(0, len(val_x), batch_size):
                # Prepare batch data
                batch_x = val_x[i:i + batch_size]
                batch_y = val_y[i:i + batch_size]
                batch_static = static[i:i + batch_size]
                batch_mob = norm_mob[i:i + batch_size, :][:, i:i + batch_size]
                batch_dist = norm_dist[i:i + batch_size, :][:, i:i + batch_size]

                # Convert to tensors and move to device
                batch_x = torch.tensor(batch_x).float().to(device)
                batch_y = torch.tensor(batch_y).float().unsqueeze(-1).to(device)
                batch_static = torch.tensor(batch_static).float().to(device)
                batch_mob = torch.tensor(batch_mob).float().to(device)
                batch_dist = torch.tensor(batch_dist).float().to(device)
                batch_mat = torch.cat([batch_mob.unsqueeze(-1), batch_dist.unsqueeze(-1)], dim=2)
                cur_static = [
                    batch_static[:, :4],
                    batch_static[:, 4:9],
                    batch_static[:, 9:14],
                    batch_mat,
                ]

                # Forward pass
                output, _ = model(batch_x, cur_static)
                loss = loss_fn(output, batch_y).mean(1).mean()

                # Record predictions and true values
                y_pred.extend(output.squeeze().cpu().numpy())
                y_true.extend(batch_y.squeeze().cpu().numpy())
                val_loss.append(loss.item())

        # Convert predictions and true values to numpy arrays
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        # Denormalize predictions and true values
        norm_pred = (y_pred * normalize_dict['y'][1]) + normalize_dict['y'][0]
        norm_true = (y_true * normalize_dict['y'][1]) + normalize_dict['y'][0]

        # Calculate metrics
        cur_mse = mse(norm_true, norm_pred)
        cur_mae = mae(norm_true, norm_pred)

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(
                f'Run: {k + 1}, Epoch: {epoch}, Train Loss: {np.mean(epoch_loss):.4f}, '
                f'Val Loss: {np.mean(val_loss):.4f}, MSE: {cur_mse:.2f}, MAE: {cur_mae:.2f}'
            )

        # Save the model if validation loss improves
        if cur_mae < min_loss:
            min_loss = cur_mae
            min_epoch = epoch
            torch.save(model.state_dict(), f'./model/hoist_{k}.pth')

    # Testing
    y_pred = []
    y_true = []
    weight_score = []
    batch_size = 128

    # Load the best model
    model.load_state_dict(torch.load(f'./model/hoist_{k}.pth'))
    model.eval()

    # Test loop
    for i in range(0, len(test_x), batch_size):
        # Prepare batch data
        batch_x = test_x[i:i + batch_size]
        batch_y = test_y[i:i + batch_size]
        batch_static = static[i:i + batch_size]
        batch_mob = norm_mob[i:i + batch_size, :][:, i:i + batch_size]
        batch_dist = norm_dist[i:i + batch_size, :][:, i:i + batch_size]

        # Convert to tensors and move to device
        batch_x = torch.tensor(batch_x).float().to(device)
        batch_y = torch.tensor(batch_y).float().unsqueeze(-1).to(device)
        batch_static = torch.tensor(batch_static).float().to(device)
        batch_mob = torch.tensor(batch_mob).float().to(device)
        batch_dist = torch.tensor(batch_dist).float().to(device)
        batch_mat = torch.cat([batch_mob.unsqueeze(-1), batch_dist.unsqueeze(-1)], dim=2)
        cur_static = [
            batch_static[:, :4],
            batch_static[:, 4:9],
            batch_static[:, 9:14],
            batch_mat,
        ]

        # Forward pass
        output, aux_outputs = model(batch_x, cur_static)

        # Record predictions, true values, and weights
        y_pred.extend(output.squeeze().cpu().numpy())
        y_true.extend(batch_y.squeeze().cpu().numpy())
        weight_score.extend(aux_outputs[1].squeeze().cpu().numpy())

    # Convert to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    weight_score = np.array(weight_score)

    # Denormalize predictions and true values
    norm_pred = (y_pred * normalize_dict['y'][1]) + normalize_dict['y'][0]
    norm_true = (y_true * normalize_dict['y'][1]) + normalize_dict['y'][0]

    # Calculate metrics
    test_mse = mse(norm_true, norm_pred)
    test_mae = mae(norm_true, norm_pred)
    test_r2 = r2(norm_true, norm_pred)
    test_ccc = ccc(norm_true, norm_pred)

    # Print test results
    print(
        f'Run: {k + 1}, Best Epoch: {min_epoch}, Test MSE: {test_mse:.2f}, '
        f'MAE: {test_mae:.2f}, R2: {test_r2:.2f}, CCC: {test_ccc:.2f}'
    )

    # Append metrics to lists
    mse_list.append(test_mse)
    mae_list.append(test_mae)
    r2_list.append(test_r2)
    ccc_list.append(test_ccc)
    mse_exp_list.append(mse(np.exp(norm_true), np.exp(norm_pred)))
    mae_exp_list.append(mae(np.exp(norm_true), np.exp(norm_pred)))
    r2_exp_list.append(r2(np.exp(norm_true), np.exp(norm_pred)))
    ccc_exp_list.append(ccc(np.exp(norm_true), np.exp(norm_pred)))

# Print average and standard deviation of metrics
print('MSE:', np.mean(mse_list), '±', np.std(mse_list))
print('MAE:', np.mean(mae_list), '±', np.std(mae_list))
print('R2:', np.mean(r2_list), '±', np.std(r2_list))
print('CCC:', np.mean(ccc_list), '±', np.std(ccc_list))
print('MSE (exp):', np.mean(mse_exp_list), '±', np.std(mse_exp_list))
print('MAE (exp):', np.mean(mae_exp_list), '±', np.std(mae_exp_list))
print('R2 (exp):', np.mean(r2_exp_list), '±', np.std(r2_exp_list))
print('CCC (exp):', np.mean(ccc_exp_list), '±', np.std(ccc_exp_list))
