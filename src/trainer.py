import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from src.net import MLP, CNN
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .dataloader import Custom_Dataset
from src.utils import evaluate_optimized
from .config import Config

def trainer(config: Config, dataloaders: dict[str, DataLoader], dataset_test: Custom_Dataset, device) -> nn.Module:

    model_type = config["model"]
    num_steps = config["num_steps"]
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
    pbar = tqdm(total=num_steps, desc="Training Progress", leave=True)

    for step in range(num_steps):

        desc = ""

        # Each step has a training and validation phase
        for phase in ['train', 'val']:  # ,
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            tk0 = dataloaders[phase]  # tqdm(dataloader[phase])
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

            step_loss = running_loss / dataset_sizes[phase]
            step_acc = running_corrects / dataset_sizes[phase]
            inputs.detach()
            labels.detach()
            
            desc += '[{} Loss: {:.4f} Acc: {:.4f}]'.format(phase, step_loss, step_acc)

            # record loss and accuracy
            writer.add_scalar(f'{phase}/Loss', step_loss, step)
            writer.add_scalar(f'{phase}/Accuracy', step_acc, step)
    
        writer.flush()
        pbar.set_description(desc)
        pbar.update(1)

    pbar.close()

    # evaluate GE and NTGE

    correct_key = dataset_test.K_attack[0]


    print("Evaluation GE/NTGE score ...")
    GE, NTGE = evaluate_optimized(
        device, 
        model, 
        dataset_test.X_attack, 
        dataset_test.P_attack, 
        correct_key, 
        leakage_model=config['leakage'], 
        nb_attacks=1, total_nb_traces_attacks=100_000, nb_traces_attacks=100_000,)
    
    # write GE (1D numpy array) to tensorboard writer
    for i, val in enumerate(GE):
        writer.add_scalar(f'GE', val, i)

    writer.add_scalar(f"NTGE", NTGE, global_step=0)
    print("Done.")

    return model