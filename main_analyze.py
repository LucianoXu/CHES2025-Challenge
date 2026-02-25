import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.dataloader import load_data, SCA_Dataset
from src.experiment import build_model, analyze


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DNN on the CHES 2025 Challenge dataset.")
    parser.add_argument("experiment_dir", type=str,
                        help="Path to the experiment output directory (contains config.json and model.pth)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Load saved configuration
    config_path = os.path.join(args.experiment_dir, "config.json")
    with open(config_path, "r") as f:
        config_template = json.load(f)
    config = Config(config_template)

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load data
    (X_train, Y_train, P_train, K_train), (X_val, Y_val, P_val, K_val), (X_test, Y_test, P_test, K_test) = load_data(config, device)

    datasets = {
        "train": SCA_Dataset(config, X_train, Y_train, P_train, K_train),
        "val": SCA_Dataset(config, X_val, Y_val, P_val, K_val),
        "test": SCA_Dataset(config, X_test, Y_test, P_test, K_test),
    }

    # Build model and load saved weights
    model = build_model(config, device)
    model_path = os.path.join(args.experiment_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    # Run evaluation
    writer = SummaryWriter(log_dir=args.experiment_dir)
    score = analyze(config, model, datasets, device, writer)
    writer.close()

    print(f"Final score: {score}")


if __name__ == "__main__":
    main()
