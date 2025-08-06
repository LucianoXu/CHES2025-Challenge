# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import random
from copy import deepcopy
import numpy as np
import torch

from .dataloader import Custom_Dataset
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data

    dataset_train = Custom_Dataset(config=expr_config)

    ##########################################################################

    dataset_train.choose_phase("train")
    dataset_val = deepcopy(dataset_train)
    dataset_val.choose_phase("validation")
    dataset_test = deepcopy(dataset_train)
    dataset_test.choose_phase("test")

    # save the configuration
    expr_config.save_config()

    batch_size = expr_config["batch_size"]
    num_workers = 0

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        ),
        "val": torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers
        ),                          
    }

    model = trainer(expr_config, dataloaders, dataset_test, device)

    torch.save(model.state_dict(), expr_config.model_path)