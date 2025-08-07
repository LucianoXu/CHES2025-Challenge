#!/usr/bin/env python3
"""
Configurable Random Training Loop Script

This script reads parameter ranges from a JSON file and continuously trains models 
with randomly generated configurations within those ranges.
"""

import json
import random
import time
import traceback
from pathlib import Path
import pandas as pd
from src.experiment import experiment
from src.config import Config

class RandomTrainingLoop:
    def __init__(self, config_file="random_config_ranges.json"):
        """
        Initialize the random training loop with configuration ranges.
        
        Args:
            config_file: Path to JSON file containing parameter ranges
        """
        self.config_file = config_file
        self.load_ranges()
    
    def load_ranges(self):
        """Load parameter ranges from configuration file."""
        try:
            with open(self.config_file, 'r') as f:
                self.ranges = json.load(f)
            print(f"Loaded parameter ranges from {self.config_file}")
        except FileNotFoundError:
            print(f"Warning: {self.config_file} not found. Using default ranges.")
            self.ranges = self.get_default_ranges()
        except json.JSONDecodeError as e:
            print(f"Error parsing {self.config_file}: {e}")
            print("Using default ranges.")
            self.ranges = self.get_default_ranges()
    
    def get_default_ranges(self):
        """Return default parameter ranges."""
        return {
            "codename": "RD",
            "leakage_options": ["HW", "ID"],
            "data_augmentation_options": [True, False],
            "aug_gaussian_noise_range": [0.01, 0.1],
            "aug_random_shift_range": [1, 30],
            "model_options": ["mlp"],
            "layers_range": [2, 6],
            "hidden_dim_options": [50, 100, 200, 300, 400, 500, 600],
            "activation_options": ["relu", "selu", "elu", "tanh"],
            "optimizer_options": ["Adam", "RMSprop"],
            "lr_options": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            "batch_size_options": [256, 500, 1000, 1500, 2000],
            "num_epochs_range": [3, 15],
            "train_size": 480000,
            "val_size": 20000,
            "test_size": 100000,
            "dataset_path": "./Dataset/CHES_2025/CHES_Challenge.h5",
            "output_dir": "./Results",
            "delay_between_experiments": 5
        }
    
    def generate_random_config(self, expr_num: int) -> dict:
        """
        Generate a random configuration based on the loaded ranges.
        
        Args:
            expr_num: The experiment number to use
            
        Returns:
            Dictionary containing the random configuration
        """
        # Base configuration
        config = {
            "output_dir": self.ranges["output_dir"],
            "codename": self.ranges["codename"],
            "expr_num": expr_num,
            "comment": f"Random configuration experiment #{expr_num}",
            "dataset": self.ranges["dataset_path"],
            "train_size": self.ranges["train_size"],
            "val_size": self.ranges["val_size"],
            "test_size": self.ranges["test_size"],
        }
        
        # Random parameters
        config["leakage"] = random.choice(self.ranges["leakage_options"])
        config["data_augmentation"] = random.choice(self.ranges["data_augmentation_options"])
        
        if config["data_augmentation"]:
            noise_range = self.ranges["aug_gaussian_noise_range"]
            config["aug_gaussian_noise"] = random.uniform(noise_range[0], noise_range[1])
            
            shift_range = self.ranges["aug_random_shift_range"]
            config["aug_random_shift"] = random.randint(shift_range[0], shift_range[1])
        else:
            config["aug_gaussian_noise"] = 0.01
            config["aug_random_shift"] = 3
        
        # Model configuration
        config["model"] = random.choice(self.ranges["model_options"])
        
        layers_range = self.ranges["layers_range"]
        layers = random.randint(layers_range[0], layers_range[1])
        hidden_dim = random.choice(self.ranges["hidden_dim_options"])
        activation = random.choice(self.ranges["activation_options"])
        
        # Adjust output_dim based on leakage model
        if config["leakage"] == "HW":
            output_dim = 9  # 0-8 for Hamming Weight
        else:  # ID
            output_dim = 256  # 0-255 for Identity
        
        config["model_args"] = {
            "input_dim": 7000,  # Fixed based on dataset
            "output_dim": output_dim,
            "layers": layers,
            "hidden_dim": hidden_dim,
            "activation": activation
        }
        
        # Training parameters
        config["optimizer"] = random.choice(self.ranges["optimizer_options"])
        config["optimizer_args"] = {}
        config["lr"] = random.choice(self.ranges["lr_options"])
        config["batch_size"] = random.choice(self.ranges["batch_size_options"])
        
        epochs_range = self.ranges["num_epochs_range"]
        config["num_epochs"] = random.randint(epochs_range[0], epochs_range[1])
        
        return config
    
    def get_next_experiment_number(self) -> int:
        """
        Find the next available experiment number by checking existing directories.
        
        Returns:
            The next experiment number to use
        """
        results_dir = Path(self.ranges["output_dir"])
        if not results_dir.exists():
            return 1
        
        rd_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith(self.ranges["codename"])]
        
        if not rd_dirs:
            return 1
        
        # Extract numbers from directory names
        numbers = []
        for d in rd_dirs:
            try:
                num = int(d.name[len(self.ranges["codename"]):])  # Remove codename prefix
                numbers.append(num)
            except ValueError:
                continue
        
        if not numbers:
            return 1
        
        return max(numbers) + 1
    
    def save_config(self, config: dict, filename: str):
        """Save configuration to file."""
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save config to {filename}: {e}")
    
    def print_config_summary(self, config: dict):
        """Print a summary of the current configuration."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {config['codename']}{config['expr_num']}")
        print(f"{'='*60}")
        print(f"Leakage Model: {config['leakage']}")
        print(f"Data Augmentation: {config['data_augmentation']}")
        if config['data_augmentation']:
            print(f"  - Gaussian Noise: {config['aug_gaussian_noise']:.3f}")
            print(f"  - Random Shift: {config['aug_random_shift']}")
        print(f"Model: {config['model'].upper()}")
        print(f"  - Layers: {config['model_args']['layers']}")
        print(f"  - Hidden Dim: {config['model_args']['hidden_dim']}")
        print(f"  - Activation: {config['model_args']['activation']}")
        print(f"  - Output Dim: {config['model_args']['output_dim']}")
        print(f"Optimizer: {config['optimizer']}")
        print(f"Learning Rate: {config['lr']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Epochs: {config['num_epochs']}")
        print(f"{'='*60}")
    
    def run(self):
        """Main training loop that runs forever with random configurations."""
        print("Starting Configurable Random Training Loop...")
        print("Press Ctrl+C to stop the loop.\n")
        
        # Start with the next available experiment number
        expr_num = self.get_next_experiment_number()

        # the dafaframe that records the data of the experiments
        df = pd.DataFrame()
        
        try:
            while True:
                start_time = time.time()
                
                # Generate random configuration
                config_dict = self.generate_random_config(expr_num)
                                
                # Print configuration summary
                self.print_config_summary(config_dict)
                
                try:
                    # Create Config object and run experiment (passing seed=None for randomization)
                    config = Config(config_dict)
                    score = experiment(config, seed=None)
                    
                    end_time = time.time()
                    duration = end_time - start_time

                    print(f"\n✅ Experiment {self.ranges['codename']}{expr_num} completed successfully!")
                    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

                    # record the experiment result to the dataframe
                    config_dict['SCORE'] = score
                    df = pd.concat([df, pd.json_normalize(config_dict)], ignore_index=True)
                    df.to_csv(f"{self.ranges['output_dir']}/random_search_results.csv", index=False)

                    
                except Exception as e:
                    print(f"\n❌ Experiment {self.ranges['codename']}{expr_num} failed with error:")
                    print(f"Error: {str(e)}")
                    print("\nFull traceback:")
                    traceback.print_exc()
                    
                    # Save failed config for debugging
                    self.save_config(config_dict, f"failed_config_RD{expr_num}.json")
                    print(f"Failed configuration saved to failed_config_RD{expr_num}.json")


                # Move to next experiment
                expr_num += 1
                
                # Add delay between experiments
                delay = self.ranges.get("delay_between_experiments", 5)
                print(f"\nWaiting {delay} seconds before starting {self.ranges['codename']}{expr_num}...\n")
                time.sleep(delay)
                
        except KeyboardInterrupt:
            print(f"\n\nTraining loop stopped by user.")
            print(f"Last experiment number: {self.ranges['codename']}{expr_num-1}")
            print("Random configurations were saved for debugging.")

def main():
    """Main entry point."""
    loop = RandomTrainingLoop()
    loop.run()

if __name__ == "__main__":
    main()
