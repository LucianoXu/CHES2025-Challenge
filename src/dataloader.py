import os
from typing import Literal
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.utils import calculate_HW, load_ctf_2025
import torch
from .config import Config

class Custom_Dataset(Dataset):
    def __init__(self, config: Config):

        self.config = config

        train_size = config["train_size"]
        val_size = config["val_size"]
        test_size = config["test_size"]

        byte = 0
        (self.X_profiling, self.X_attack), \
        (self.Y_profiling, self.Y_attack), \
        (self.P_profiling, self.P_attack), \
        (self.K_profiling, self.K_attack) = load_ctf_2025(
            config["dataset"],
            byte=byte, 
            train_begin=0, train_end=train_size + val_size, 
            test_begin=0, test_end=test_size)

        # we know that we are using the same key
        self.correct_key = self.K_attack[0]

        self.scaler_std = StandardScaler()

        self.X_profiling = self.scaler_std.fit_transform(self.X_profiling)
        self.X_attack = self.scaler_std.transform(self.X_attack)

        # split train into train and validation
        self.X_train = self.X_profiling[:train_size]
        self.Y_train = self.Y_profiling[:train_size]
        
        self.X_val = self.X_profiling[train_size:train_size + val_size]
        self.Y_val = self.Y_profiling[train_size:train_size + val_size]

        self.X_test = self.X_attack
        self.Y_test = self.Y_attack


    def choose_phase(self, phase: Literal['train', 'validation', 'test']):
        if phase == 'train':
            self.X, self.Y = np.expand_dims(self.X_train, 1), self.Y_train
        elif phase == 'validation':
            self.X, self.Y = np.expand_dims(self.X_val, 1), self.Y_val
        elif phase == 'test':
            self.X, self.Y = np.expand_dims(self.X_test, 1), self.Y_test
        else:
            raise ValueError("Phase must be 'train', 'validation', or 'test'.")


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X[idx]
        sensitive = self.Y[idx]

        if self.config['leakage'] == 'HW':
            sensitive = calculate_HW(sensitive)

        elif self.config['leakage'] == 'ID':
            pass

        else:
            raise ValueError("Unsupported leakage model. Use 'HW' or 'ID'.")

        trace = torch.from_numpy(trace).float()
        sensitive = torch.from_numpy(np.array(sensitive)).long()

        return trace, sensitive