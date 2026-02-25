import json
from src.experiment import experiment
from src.config import Config

if __name__ == "__main__":

    with open("expr_config.json", "r") as f:
        config_template = json.loads(f.read())

    config = Config(config_template)

    # Train and evaluate the model
    experiment(config)
