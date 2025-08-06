# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import random
import numpy as np
import torch

from .dataloader import load_data, SCA_Dataset
from .trainer import trainer
from .config import Config, create_config_template

def experiment(expr_config: Config):
    '''
    Do the experiment of training and evaluating the model for the given configuration.
    The results will be saved in the output directory specified in the configuration, including configuration file, model weights, and evaluation results.
    '''

    print("Configuration:\n", expr_config.get_json())

    # Create output directory if it does not exist
    if not os.path.exists(expr_config.output_folder):
        os.makedirs(expr_config.output_folder, exist_ok=True)

    # initialize device
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda:0"

    # load data
    (X_train, Y_train, P_train, K_train), (X_val, Y_val, P_val, K_val), (X_test, Y_test, P_test, K_test)= load_data(expr_config, device)

    # save the configuration
    expr_config.save_config()

    datasets = {
        "train": SCA_Dataset(expr_config, X_train, Y_train, P_train, K_train),
        "val": SCA_Dataset(expr_config, X_val, Y_val, P_val, K_val),
        "test": SCA_Dataset(expr_config, X_test, Y_test, P_test, K_test),
    }

    model = trainer(
        config=expr_config, 
        datasets=datasets,
        device=device
    )

    torch.save(model.state_dict(), expr_config.model_path)