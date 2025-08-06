from typing import Callable, Literal
import numpy as np
from torch.utils.data import Dataset
from src.utils import calculate_HW_single, load_ctf_2025
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
    '''
    Load the data from the dataset specified in the configuration.

    Args:
        config: Configuration object containing dataset parameters
        device: Device to use for data processing ('cuda' or 'cpu')

    Returns:
        Tuple of training, validation, and test datasets as numpy arrays
        X_train, X_val, X_test: shape (N, T, 1)
        other arrays: shape (N, )
    '''

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
    X_train = np.expand_dims(X_profiling[:train_size], -1)
    Y_train = Y_profiling[:train_size]
    P_train = P_profiling[:train_size]
    K_train = K_profiling[:train_size]
    
    X_val = np.expand_dims(X_profiling[train_size:train_size + val_size], -1)
    Y_val = Y_profiling[train_size:train_size + val_size]
    P_val = P_profiling[train_size:train_size + val_size]
    K_val = K_profiling[train_size:train_size + val_size]

    X_test = np.expand_dims(X_attack, -1)
    Y_test = Y_attack
    P_test = P_attack
    K_test = K_attack

    return (X_train, Y_train, P_train, K_train), (X_val, Y_val, P_val, K_val), (X_test, Y_test, P_test, K_test)

def data_augmentation(config: Config, X: np.ndarray, device: str) -> np.ndarray:
    """
    Apply data augmentation to the input traces, according to the configuration.

    If data augmentation is disabled, return the original traces.
    Otherwise, a new array with augmented traces is returned.
    
    Args:
        config: Configuration object containing augmentation parameters
        X: Input traces as a numpy array, shape (N, T, 1) or (N, T). Should be already standardized (zero mean, unit variance).
    
    Returns:
        Augmented traces as a numpy array
    """
    
    if not config["data_augmentation"]:
        return X
    
    # transform to torch
    if len(X.shape) == 3:
        squeeze_last = True
        X_tensor = torch.from_numpy(X).squeeze(-1).to(device)
    else:
        squeeze_last = False
        X_tensor = torch.from_numpy(X).to(device)

    N, T = X_tensor.shape

    # apply Gaussian noise
    sigma_ratio = float(config["aug_gaussian_noise"])
    if sigma_ratio > 0:
        noise = torch.randn_like(X_tensor) * sigma_ratio
        X_tensor += noise

    # apply random shift
    K = int(config["aug_random_shift"])
    if K > 0:
        # Sample per-trace shifts
        shifts = torch.randint(-K, K + 1, (N,), device=device)
        # Vectorized roll via gather
        idx = torch.arange(T, device=device).unsqueeze(0).expand(N, T)     # (N, T)
        idx = (idx - shifts.unsqueeze(1)) % T               # (N, T)
        X_tensor = X_tensor.gather(1, idx)     
    
    # ---- return with original shape ----
    if squeeze_last:
        X_tensor = X_tensor.unsqueeze(-1)  # (N, T, 1)
    return X_tensor.to('cpu').numpy()  # convert back to numpy and return

class SCA_Dataset(Dataset):
    def __init__(self, config: Config, X: np.ndarray, Y: np.ndarray, P: np.ndarray, K: np.ndarray):
        self.config = config
        self.X = X
        self.Y = Y
        self.P = P
        self.K = K

        self.leakage_fun : Callable

        if self.config['leakage'] == 'HW':
            self.leakage_fun = calculate_HW_single

        elif self.config['leakage'] == 'ID':
            self.leakage_fun = lambda x: x

        else:
            raise ValueError("Unsupported leakage model.")
        
    def augment(self, device: str):
        '''
        According to the configuration, return a new augmented dataset or the original dataset.
        '''
        if not self.config["data_augmentation"]:
            return self
        
        else:
            X_augmented = data_augmentation(self.config, self.X, device)
            return SCA_Dataset(
                config=self.config,
                X=X_augmented,
                Y=self.Y,
                P=self.P,
                K=self.K
            )


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):

        trace = self.X[idx]
        sensitive = self.leakage_fun(self.Y[idx])

        trace = torch.from_numpy(trace).float()
        sensitive = torch.from_numpy(np.array(sensitive)).long()

        return trace, sensitive