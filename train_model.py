import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy import stats
from tqdm import tqdm

# Seed setting for reproducibility
def seed_torch(RANDOM_SEED=123):
    random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Initialize seed and device
seed_torch()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Random seed set to 123')
print(f'Using device: {device}')


# Utility Functions

def get_dist(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points.
    """
    R = 6371e3  # Earth radius in meters
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

def generate_series(data, y, window_size, pred_size, date=False):
    '''
    Generates time series data.
    data: N*T*F
    y: N*T
    Returns:
        series: N_samples x T_window x F
        targets: N_samples x T_pred
    '''
    series = []
    targets = []
    idx = window_size
    while idx + pred_size <= data.shape[1]:
        if date:
            series.append(data[:, idx-1, :])
        else:
            series.append(np.sum(data[:, idx-window_size:idx, :], axis=1))
        targets.append(np.sum(y[:, idx:idx+pred_size], axis=1))
        idx += pred_size
    return np.array(series), np.array(targets)

def temporal_split(x, y, static, mats, val_ratio, test_ratio, norm='min-max', norm_mat=True):
    '''
    Splits the data temporally into training, validation, and testing sets.
    '''
    # Shuffle regions
    num_regions = x.shape[0]
    shuffle_idx = np.arange(num_regions)
    np.random.shuffle(shuffle_idx)
    x = x[shuffle_idx]
    y = y[shuffle_idx]
    static = static[shuffle_idx]
    for i in range(len(mats)):
        mats[i] = mats[i][shuffle_idx, :][:, shuffle_idx]
    
    # Split indices
    seq_len = x.shape[1]
    test_len = int(seq_len * test_ratio)
    val_len = int(seq_len * test_ratio)  # Typically, val_ratio and test_ratio should sum to <=1
    train_len = seq_len - val_len - test_len
    
    train_x = x[:, :train_len, :]
    train_y = y[:, :train_len]
    train_idx = np.arange(train_len)
    
    val_x = x[:, train_len:train_len+val_len, :]
    val_y = y[:, train_len:train_len+val_len]
    val_idx = np.arange(train_len, train_len+val_len)
    
    test_x = x[:, train_len+val_len:, :]
    test_y = y[:, train_len+val_len:]
    test_idx = np.arange(train_len+val_len, seq_len)
    
    # Normalization
    normalize_dict = {}
    if norm == 'min-max':
        # Normalize features between 0 and 1
        for i in range(x.shape[2]):
            min_val = train_x[:, :, i].min()
            max_val = train_x[:, :, i].max()
            normalize_dict[i] = [min_val, max_val]
            train_x[:, :, i] = (train_x[:, :, i] - min_val) / (max_val - min_val) if (max_val - min_val) !=0 else 0
            val_x[:, :, i] = (val_x[:, :, i] - min_val) / (max_val - min_val) if (max_val - min_val) !=0 else 0
            test_x[:, :, i] = (test_x[:, :, i] - min_val) / (max_val - min_val) if (max_val - min_val) !=0 else 0
        
        # Normalize targets
        y_min = train_y.min()
        y_max = train_y.max()
        normalize_dict['y'] = [y_min, y_max]
        train_y = (train_y - y_min) / (y_max - y_min) if (y_max - y_min)!=0 else 0
        val_y = (val_y - y_min) / (y_max - y_min) if (y_max - y_min)!=0 else 0
        test_y = (test_y - y_min) / (y_max - y_min) if (y_max - y_min)!=0 else 0
        
        # Normalize static features
        static_min = static.min(axis=0)
        static_max = static.max(axis=0)
        normalize_dict['static'] = [static_min, static_max]
        static = (static - static_min) / (static_max - static_min) if (static_max - static_min).any() else 0
        
        # Normalize matrices if required
        if norm_mat:
            for i in range(len(mats)):
                mat_min = mats[i].min()
                mat_max = mats[i].max()
                normalize_dict[f'mat_{i}'] = [mat_min, mat_max]
                mats[i] = (mats[i] - mat_min) / (mat_max - mat_min) if (mat_max - mat_min)!=0 else 0
    
    return train_x, val_x, test_x, train_y, val_y, test_y, train_idx, val_idx, test_idx, static, mats, normalize_dict, shuffle_idx

def mse(y_true, y_pred, std=False):
    if std:
        return np.mean((y_true - y_pred)**2), np.std((y_true - y_pred)**2)
    else:
        return np.mean((y_true - y_pred)**2)

def mae(y_true, y_pred, std=False):
    if std:
        return np.mean(np.abs(y_true - y_pred)), np.std(np.abs(y_true - y_pred))
    else:
        return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred, std=False):
    return r2_score(y_true, y_pred)

def ccc(y_true, y_pred, std=False):
    return stats.pearsonr(y_true.flatten(), y_pred.flatten())[0]


# Model Definition

class STGNN_MTC_ICU_Forecaster(nn.Module):
    def __init__(self, input_dim, static_dim, hidden_dim, num_gat_layers=2, num_heads=4, dropout=0.3):
        super(STGNN_MTC_ICU_Forecaster, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Multi-Scale Temporal Convolutions
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=7, padding=3)
        
        # 2. Spatiotemporal Attention Mechanisms
        self.temporal_att = nn.MultiheadAttention(embed_dim=hidden_dim * 3, num_heads=4, dropout=dropout)
        
        # 3. Node Feature Projection
        self.node_proj = nn.Linear(static_dim, hidden_dim * 3)
        
        # 4. Graph Neural Network Layers (Using GAT)
        self.gat_layers = nn.ModuleList()
        for _ in range(num_gat_layers):
            gat_layer = nn.MultiheadAttention(embed_dim=hidden_dim * 3, num_heads=num_heads, dropout=dropout)
            self.gat_layers.append(gat_layer)
        
        # 5. Prediction Module with Uncertainty Estimation
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, static, adj):
        """
        x: [batch_size, T, F] - Dynamic features
        static: [batch_size, D_static] - Static features
        adj: [batch_size, N, N] - Adjacency matrix
        """
        batch_size, T, F = x.size()
        
        # 1. Multi-Scale Temporal Convolutions
        x_perm = x.permute(0, 2, 1)  # [batch_size, F, T]
        conv1 = F.relu(self.conv1(x_perm))  # [batch_size, hidden_dim, T]
        conv2 = F.relu(self.conv2(x_perm))  # [batch_size, hidden_dim, T]
        conv3 = F.relu(self.conv3(x_perm))  # [batch_size, hidden_dim, T]
        
        conv_concat = torch.cat([conv1, conv2, conv3], dim=1)  # [batch_size, hidden_dim*3, T]
        conv_out = conv_concat.permute(0, 2, 1)  # [batch_size, T, hidden_dim*3]
        
        # 2. Spatiotemporal Attention Mechanisms
        att_input = conv_out.permute(1, 0, 2)  # [T, batch_size, hidden_dim*3]
        att_output, _ = self.temporal_att(att_input, att_input, att_input)  # [T, batch_size, hidden_dim*3]
        att_output = att_output.permute(1, 0, 2)  # [batch_size, T, hidden_dim*3]
        
        temporal_rep = conv_out + att_output  # [batch_size, T, hidden_dim*3]
        
        # 3. Node Feature Projection
        node_features = self.node_proj(static)  # [batch_size, hidden_dim*3]
        node_features = node_features.unsqueeze(1)  # [batch_size, 1, hidden_dim*3]
        
        # 4. Graph Neural Network Layers (Using GAT)
        # Prepare for MultiheadAttention: [N, batch_size, embed_dim]
        node_features = node_features.repeat(1, adj.size(1), 1)  # [batch_size, N, hidden_dim*3]
        node_features = node_features.permute(1, 0, 2)  # [N, batch_size, hidden_dim*3]
        
        for gat in self.gat_layers:
            gat_output, _ = gat(node_features, node_features, node_features)  # [N, batch_size, hidden_dim*3]
            node_features = gat_output  # Update node features
        
        node_features = node_features.permute(1, 0, 2)  # [batch_size, N, hidden_dim*3]
        graph_rep = node_features.mean(dim=1)  # [batch_size, hidden_dim*3]
        
        # 5. Prediction Module with Uncertainty Estimation
        out = F.relu(self.fc1(graph_rep))  # [batch_size, hidden_dim]
        out = self.dropout(out)
        out = self.fc2(out)  # [batch_size, 1]
        
        return out  # [batch_size, 1]


# Data Loading and Preprocessing

# Load the .pkl files
mob_mat = pickle.load(open('./data/mob_mat.pkl', 'rb'))  # Mobility matrix: [N, N]
distance_mat = pickle.load(open('./data/distance_mat.pkl', 'rb'))  # Distance matrix: [N, N]
covid_tensor = pickle.load(open('./data/covid_tensor.pkl', 'rb'))  # COVID cases: [N, T]
hospitalizations = pickle.load(open('./data/hospitalizations.pkl', 'rb'))  # Hospitalizations: [N, T]
hos_tensor = pickle.load(open('./data/hos_tensor.pkl', 'rb'))  # Hospital features: [N, T, 4]
county_tensor = pickle.load(open('./data/county_tensor.pkl', 'rb'))  # Static features: [N, 14]
feat_name = pickle.load(open('./data/feat_name.pkl', 'rb'))  # Feature names
date_range = np.array(pickle.load(open('./data/date_range.pkl', 'rb')))  # Dates: [T]

# Print data shapes
print("mob_mat shape:", mob_mat.shape)
print("distance_mat shape:", distance_mat.shape)
print("covid_tensor shape:", covid_tensor.shape)
print("hospitalizations shape:", hospitalizations.shape)
print("hos_tensor shape:", hos_tensor.shape)
print("county_tensor shape:", county_tensor.shape)
print("feat_name:", feat_name)
print("date_range shape:", date_range.shape)

# Align regions across datasets
# Assuming all datasets have the same regions in the same order
regions_covid = set(range(covid_tensor.shape[0]))
regions_static = set(range(county_tensor.shape[0]))

# Data Preprocessing
# Expand covid_tensor to [N, T, 1] to concatenate with hos_tensor
covid_tensor = np.expand_dims(covid_tensor, axis=2)  # [N, T, 1]
X = np.concatenate([covid_tensor, hos_tensor], axis=2)  # [N, T, 5]
y = hospitalizations  # [N, T]

# Generate time series data
window_size = 35
pred_size = 28
X_series, y_series = generate_series(X, y, window_size=window_size, pred_size=pred_size)

# Filter regions with mean target > 0
range_idx = (y_series.mean(axis=1) > 0)
county_tensor = county_tensor[range_idx]
y_series = y_series[range_idx]
X_series = X_series[range_idx]
mob_mat = mob_mat[range_idx, :][:, range_idx]
distance_mat = distance_mat[range_idx, :][:, range_idx]
print(f'Number of regions after filtering: {len(y_series)}')

# Apply logarithmic transformation
y_series = np.log(y_series + 1)

# Temporal split
train_x, val_x, test_x, train_y, val_y, test_y, train_idx, val_idx, test_idx, static, mats, normalize_dict, shuffle_idx = temporal_split(
    X_series, y_series, county_tensor, [mob_mat, distance_mat], val_ratio=0.2, test_ratio=0.2, norm='min-max', norm_mat=True
)

norm_mob = mats[0]  # Normalized mobility matrix
norm_dist = mats[1]  # Normalized distance matrix

print('Data preprocessing completed.')
