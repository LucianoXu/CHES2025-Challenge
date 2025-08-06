import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from src.net import MLP, CNN
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .dataloader import SCA_Dataset
from src.utils import evaluate_optimized
from .config import Config

def trainer(config: Config, datasets: dict[str, SCA_Dataset], device) -> nn.Module:
    '''
    The training will make use of all training data in the dataloader.
    '''

    model_type = config["model"]
    num_epochs = config["num_epochs"]
    dataset_sizes = {'train': config['train_size'], 'val': config['val_size']}

    # create the tensorboard writer and record the training setting
    writer = SummaryWriter(log_dir=config.output_folder)
    writer.add_text("config", config.get_json())

    # Build the model
    model_args = config["model_args"]
    if model_type == "mlp":
        model = MLP(model_args, model_args["input_dim"], model_args["output_dim"]).to(device)
    elif model_type == "cnn":
        model = CNN(model_args, model_args["input_dim"], model_args["output_dim"]).to(device)

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

            tk0 = torch.utils.data.DataLoader(
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

    # evaluate GE and NTGE

    dataset_test = datasets['test']
    correct_key = dataset_test.K[0]

    print("Evaluation GE/NTGE score ...")
    GE, NTGE = evaluate_optimized(
        device, 
        model, 
        dataset_test.X, 
        dataset_test.P, 
        correct_key, 
        leakage_model=config['leakage'], 
        nb_attacks=1, total_nb_traces_attacks=100_000, nb_traces_attacks=100_000,)
    
    # write GE (1D numpy array) to tensorboard writer
    for i, val in enumerate(GE):
        writer.add_scalar(f'GE', val, i)

    writer.add_scalar(f"NTGE", NTGE, global_step=0)
    print("Done.")

    return model