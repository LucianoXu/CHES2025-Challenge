# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataloader import load_data, SCA_Dataset
from .utils import evaluate_optimized, key_wise_log_likelihood, key_wise_log_likelihood_plot
from .config import Config

from .model import MLP, WindowedMLP, GatedMLP, CNN

def trainer(config: Config, datasets: dict[str, SCA_Dataset], device) -> tuple[nn.Module, float]:
    '''
    The training will make use of all training data in the dataloader.

    Returns:
        model: the trained model
        score: the score for this competition (upper bounded by 200K)
    '''

    model_type = config["model"]
    num_epochs = config["num_epochs"]
    dataset_sizes = {'train': config['train_size'], 'val': config['val_size']}

    # create the tensorboard writer
    writer = SummaryWriter(log_dir=config.output_folder)

    # Build the model
    model_args = config["model_args"]
    if model_type == "mlp":
        model = MLP(model_args).to(device)
    elif model_type == "window_mlp":
        model = WindowedMLP(model_args).to(device)
    elif model_type == "gated_mlp":
        model = GatedMLP(model_args).to(device)
    elif model_type == "cnn":
        model = CNN(model_args).to(device)
        print("===Completed CNN model Hyperparameters===")
        print(model.model_args)
        print()

    # record the model size
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size} parameters")

    # record the training setting
    training_record = "Model Size: {} parameters\n\n".format(model_size)
    training_record += config.get_json()
    writer.add_text("config", training_record)

    # Creates the optimizer
    lr = config["lr"]

    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    # This is the trainning Loop
    criterion = nn.CrossEntropyLoss()

    # Initialize progress bar
    pbar = tqdm(total=num_epochs, desc="Training Progress", leave=True)

    best_val_loss = float('inf')
    best_model_state = None
    early_stop: bool = False

    for epoch in range(num_epochs):

        desc = ""

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  # ,
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            ds = datasets[phase]  # tqdm(dataloader[phase])

            # augument the dataset (or not) for training
            if phase == 'train':
                ds = ds.augment(device)

            tk0 = DataLoader(
                ds,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=0
            )

            # Iterate over all data (one epoch).
            for (traces, labels) in tk0:
                inputs = traces.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, dim=1)

                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            inputs.detach()
            labels.detach()
            
            desc += '[{} Loss: {:.4f} Acc: {:.4f}]'.format(phase, epoch_loss, epoch_acc)

            # record loss and accuracy
            writer.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/Accuracy', epoch_acc, epoch)

            # compare validation loss with history
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_state = model.state_dict()
                
                elif epoch_loss > best_val_loss * 1.002:
                    print(f"Validation loss increased by more than 5%: {epoch_loss:.4f} > {best_val_loss * 1.05:.4f}. Early stopping.")
                    early_stop = True
    
        writer.flush()
        pbar.set_description(desc)
        pbar.update(1)

        if early_stop:
            print("Early stopping triggered.")
            break

    pbar.close()

    # Load the best model state
    assert best_model_state is not None
    model.load_state_dict(best_model_state)

    ############################################
    # Evaluation Phase

    if hasattr(model, 'free_cache'):
        model.free_cache = True # type: ignore[assignment]

    # evaluate GE and NTGE

    dataset_test = datasets['test']
    correct_key = dataset_test.K[0]

    print("Evaluation GE/NTGE score ...")
    GE, NTGE, test_key_log_prob = evaluate_optimized(
        device, 
        model, 
        dataset_test.X, 
        dataset_test.P, 
        correct_key, 
        leakage_model=config['leakage'], 
        nb_attacks=config['num_attacks'], 
        total_nb_traces_attacks=100_000, 
        attack_trace_usage=100_000,)
    # record the test key log likelihood distribution
    # the values will be large negative numbers because they are probabilities product of joint events (k0,k0, ..., k0) throughout the whole trace
    key_wise_log_likelihood_plot(
        "Test Joint Key Log-Likelihood Distribution",
        test_key_log_prob,
        writer,
        highlight_indices=[correct_key],
        global_step=config["test_size"]-1
    )
    
    # write GE (1D numpy array) to tensorboard writer
    for i, val in enumerate(GE):
        writer.add_scalar(f'GE', val, i)

    # calculate the score
    if NTGE == float('inf'):
        score = 200_000 + round((GE[-1].item()))
    else:
        score = NTGE

    writer.add_scalar(f"Score", score, global_step=0)
    print("Score: ", score)
    print("Results saved to tensorboard.")

    # for validation, calculate key-wise log-likelihood distribution and write to tensorboard
    print("Val: Calculating Key-wise Log-Likelihood Distribution ...")
    val_key_log_prob = key_wise_log_likelihood(
        model, 
        datasets['val'].X, 
        datasets['val'].Y, 
        datasets['val'].K, 
        device=device
    )
    key_wise_log_likelihood_plot(
        "Validation Key-wise Log-Likelihood Distribution", 
        val_key_log_prob, 
        writer
    )
    print("Results saved to tensorboard.")

    # for test, calculate key-wise log-likelihood distribution and write to tensorboard
    print("Test: Calculating Key-wise Log-Likelihood Distribution ...")
    test_key_log_prob = key_wise_log_likelihood(
        model,
        datasets['test'].X,
        datasets['test'].Y,
        datasets['test'].K,
        device=device
    )
    correct_key_log_likelihood = test_key_log_prob[correct_key]
    print(f"Correct key log likelihood: {correct_key_log_likelihood}")
    writer.add_scalar(f'Test Key Log-Likelihood', correct_key_log_likelihood, 0)

    print("Done.")

    return model, score

def experiment(expr_config: Config, seed: int|None = 0) -> float:
    '''
    Do the experiment of training and evaluating the model for the given configuration.
    The results will be saved in the output directory specified in the configuration, including configuration file, model weights, and evaluation results.

    Returns:
        score: the score for this competition (upper bounded by 200K)
    '''

    print("Configuration:\n", expr_config.get_json())

    # Create output directory if it does not exist
    if not os.path.exists(expr_config.output_folder):
        os.makedirs(expr_config.output_folder, exist_ok=True)

    # set the seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load data
    (X_train, Y_train, P_train, K_train), (X_val, Y_val, P_val, K_val), (X_test, Y_test, P_test, K_test)= load_data(expr_config, device)

    # save the configuration
    expr_config.save_config()

    datasets = {
        "train": SCA_Dataset(expr_config, X_train, Y_train, P_train, K_train),
        "val": SCA_Dataset(expr_config, X_val, Y_val, P_val, K_val),
        "test": SCA_Dataset(expr_config, X_test, Y_test, P_test, K_test),
    }

    model, score = trainer(
        config=expr_config, 
        datasets=datasets,
        device=device
    )

    torch.save(model.state_dict(), expr_config.model_path)

    return score