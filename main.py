
from src.experiment import experiment
from src.config import Config, create_config_template
import json

if __name__ == "__main__":

    # Create a configuration
    # !! remember to increse the expr_num !!
    with open("expr_config.json", "r") as f:
        config_template = json.loads(f.read())
    
    # Create a Config object
    config = Config(config_template)
    
    # Train the model with the given configuration
    experiment(config)