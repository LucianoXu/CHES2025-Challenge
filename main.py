
from src.experiment import experiment
from src.config import Config, create_config_template

if __name__ == "__main__":

    # Create a configuration
    # !! remember to increse the expr_num !!
    config_template = {
        "output_dir": "./Results",
        "expr_num": 5,
        "dataset": "./Dataset/CHES_2025/CHES_Challenge.h5",
        "train_size": 480_000,
        "val_size": 20_000,
        "test_size": 100000,
        "leakage": "ID",
        "model": "mlp",
        "model_args": {
            "input_dim": 7000,
            "output_dim": 256,
            "layers": 2,
            "hidden_dim": 100,
            "activation": "relu"
        },
        "optimizer": "Adam",
        "optimizer_args": {},
        "lr": 0.0001,
        "batch_size": 256,
        "num_steps": 50
    }
    
    # Create a Config object
    config = Config(config_template)
    
    # Train the model with the given configuration
    experiment(config)