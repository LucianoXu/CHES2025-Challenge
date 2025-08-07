from typing import Literal
from dataclasses import dataclass
import json
import os


def create_config_template() -> dict:
    return {
        "output_dir": "./Results",
        "codename": "V",
        "expr_num": 0,
        "comment": "Comment for the experiment",
        
        "dataset": "./Dataset/CHES_2025/CHES_Challenge.h5",
        "train_size": 100_000,
        "val_size": 10_000,
        "test_size": 100_000,   # should be fixed to 100_000

        "leakage": "ID",

        # data augmentation parameters
        "data_augmentation": True,  # whether to use data augmentation
        "aug_gaussian_noise": 0.1, # standard deviation of the Gaussian noise
        "aug_random_shift": 30, # maximum random shift in samples

        "model": "mlp",
        "model_args": {
            "input_dim": 7000,
            "output_dim": 256,
            "layers": 3,
            "hidden_dim": 200,
            "activation": "relu",
        },

        "optimizer": "Adam",
        "optimizer_args": {},

        "lr": 1e-4,
        "batch_size": 256,
        "num_epochs": 5,

        "num_attaks": 10,
    }

class Config:

    def __init__(self, config: dict):
        self.config = config

    def __getitem__(self, key):
        return self.config[key]
    
    def __str__(self):
        return str(self.config)
    
    @property
    def name(self) -> str:
        return f"{self.config['codename']}{self.config['expr_num']}"

    @property
    def output_folder(self) -> str:
        return f"{self.config['output_dir']}/{self.name}/"
    
    @property
    def model_path(self) -> str:
        return f"{self.output_folder}/model.pth"
    
    @property
    def config_path(self) -> str:
        return f"{self.output_folder}/config.json"
    
    def get_json(self) -> str:
        return json.dumps(self.config, indent=4)
    
    def save_config(self):
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
