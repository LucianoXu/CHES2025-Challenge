# CHES 2025 Challenge — Artifact

## Setup

- Prepare a `Python 3.11` environment and install dependencies:
```
pip install -r requirements.txt
```

- Download the dataset from https://drive.google.com/drive/folders/1JGbphwZXQvN_tEhpBIbQ-q-pN9wkqKQ- and place the `.h5` file at `./Dataset/CHES_2025/CHES_Challenge.h5`.

## Training

Run `main_training.py` to train and evaluate a model. It reads the configuration from `expr_config.json`:

```
python main_training.py
```

Training results (configuration, model weights, TensorBoard logs) are saved to the `Results/` directory.

## Evaluation

Run `main_analyze.py` to evaluate a previously trained model. It loads the saved configuration and model weights from an experiment output directory:

```
python main_analyze.py ./Results/<experiment_name>/
```

## Monitoring

Use TensorBoard to monitor training and evaluation results:

```
tensorboard --logdir=Results
```

## Repository Structure

```
├── src/                  # Supporting source files
│   ├── config.py         # Configuration management
│   ├── dataloader.py     # Data loading, preprocessing, denoising, augmentation
│   ├── experiment.py     # Training loop and evaluation pipeline
│   ├── model.py          # MLP and CNN model architectures
│   └── utils.py          # AES-Sbox operations, GE/NTGE evaluation metrics
├── main_training.py      # Script to train and evaluate the DNN
├── main_analyze.py       # Script to evaluate a trained DNN
├── expr_config.json      # Experiment configuration
├── requirements.txt      # Python dependencies
└── README.md
```
