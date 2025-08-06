from typing import Literal
from dataclasses import dataclass



def create_config_template() -> dict:
    return {
        "output_dir": "./Results",
        "expr_num": 0,
        "dataset": "./Dataset/CHES_2025/CHES_Challenge.h5",
        "train_size": 100_000,
        "val_size": 10_000,
        "test_size": 100_000,
        "leakage": "ID",
        "model": "",
        "model_args": {},
        "optimizer": "",
        "optimizer_args": {}
    }

class Config:

    def __init__(self, config: dict):
        self.config = config

    def __getitem__(self, key):
        return self.config[key]

    @property
    def output_folder(self) -> str:
        return f"{self.config['output_dir']}/{self.config['expr_num']}/"
    
    @property
    def model_path(self) -> str:
        return f"{self.output_folder}/model.pth"
    
    def get_json(self) -> str:
        import json
        return json.dumps(self.config, indent=4)
