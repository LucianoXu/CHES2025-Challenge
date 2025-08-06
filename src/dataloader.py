from typing import Callable, Literal
import numpy as np
from torch.utils.data import Dataset
from src.utils import calculate_HW, load_ctf_2025
import torch
from .config import Config

def standardize(X_train: np.ndarray, X_test: np.ndarray, device: str = 'cuda'):
    """
    GPU-accelerated standardization using PyTorch.
    
    Args:
        X_train: Training data to fit the scaler on
        X_test: Test data to transform
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Tuple of (standardized_X_train, standardized_X_test) as numpy arrays
    """
    # Convert to PyTorch tensors and move to GPU
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    
    # Calculate mean and std on training data
    mean = X_train_tensor.mean(dim=0, keepdim=True)
    std = X_train_tensor.std(dim=0, keepdim=True, unbiased=False)
    
    # Avoid division by zero
    std = torch.where(std == 0, torch.ones_like(std), std)
    
    # Standardize both datasets
    X_train_normalized = (X_train_tensor - mean) / std
    X_test_normalized = (X_test_tensor - mean) / std
    
    # Convert back to numpy arrays on CPU
    return X_train_normalized.cpu().numpy(), X_test_normalized.cpu().numpy()

def load_data(config: Config, device: str = 'cuda'):

    train_size = config["train_size"]
    val_size = config["val_size"]
    test_size = config["test_size"]

    (X_profiling, X_attack), \
    (Y_profiling, Y_attack), \
    (P_profiling, P_attack), \
    (K_profiling, K_attack) = load_ctf_2025(
        config["dataset"],
        byte=0, 
        train_begin=0, train_end=train_size + val_size, 
        test_begin=0, test_end=test_size)

    # normalization
    X_profiling, X_attack = standardize(X_profiling, X_attack, device)

    # split train into train and validation
    X_train = np.expand_dims(X_profiling[:train_size], 1)
    Y_train = Y_profiling[:train_size]
    P_train = P_profiling[:train_size]
    K_train = K_profiling[:train_size]
    
    X_val = np.expand_dims(X_profiling[train_size:train_size + val_size], 1)
    Y_val = Y_profiling[train_size:train_size + val_size]
    P_val = P_profiling[train_size:train_size + val_size]
    K_val = K_profiling[train_size:train_size + val_size]

    X_test = np.expand_dims(X_attack, 1)
    Y_test = Y_attack
    P_test = P_attack
    K_test = K_attack

    return (X_train, Y_train, P_train, K_train), (X_val, Y_val, P_val, K_val), (X_test, Y_test, P_test, K_test)


class SCA_Dataset(Dataset):
    def __init__(self, config: Config, X: np.ndarray, Y: np.ndarray, P: np.ndarray, K: np.ndarray):
        self.config = config
        self.X = X
        self.Y = Y
        self.P = P
        self.K = K

        self.leakage_fun : Callable[[np.ndarray], np.ndarray]

        if self.config['leakage'] == 'HW':
            self.leakage_fun = calculate_HW

        elif self.config['leakage'] == 'ID':
            self.leakage_fun = lambda x: x

        else:
            raise ValueError("Unsupported leakage model.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X[idx]
        sensitive = self.leakage_fun(self.Y[idx])

        trace = torch.from_numpy(trace).float()
        sensitive = torch.from_numpy(np.array(sensitive)).long()

        return trace, sensitive