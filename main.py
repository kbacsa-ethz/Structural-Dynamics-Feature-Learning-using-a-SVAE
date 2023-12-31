# Import necessary libraries and modules
import os
import json
import argparse
import random
import numpy as np
from datetime import datetime

# Import custom modules
from create_model import create_model
from create_dataset import create_dataset
from train import *


# Define the main function, which serves as the entry point of the program
def main(cfg):
    # Fix random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Define the folder to store experiment results
    exp_folder = 'experiments'
    exp_path = os.path.join(cfg.root_path, exp_folder)

    # Check if resuming a previous experiment
    if cfg.resume:
        log_path = os.path.join(exp_path, cfg.resume)
        with open(os.path.join(log_path, "config.txt")) as f:
            config_dict = json.loads(f.read())
    else:
        # Create a new experiment folder if not resuming
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        dt_string += '_' + cfg.model_type + '-' + str(cfg.batch_size)
        log_path = os.path.join(exp_path, dt_string)

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        with open(os.path.join(log_path, 'config.txt'), 'w') as f:
            config_dict = cfg.__dict__
            json.dump(config_dict, f, indent=2)

    save_path = os.path.join(log_path, 'ckpt.pth')

    # Create datasets
    datasets, dataloaders = create_dataset(
        cfg.data_path,
        cfg.n_dof,
        cfg.batch_size,
        cfg.num_workers
    )

    # Create the machine learning model
    model, model_hyperparameters = create_model(
        cfg.model_type,
        cfg.extractor,
        cfg.in_channels,
        cfg.latent_dim,
        cfg.n_classes,
        cfg.num_layers,
        cfg.class_layers,
        cfg.dropout,
        cfg.target,
        cfg.seq_len,
        cfg.batch_size
    )

    # Create a training handle based on the model type
    if model_hyperparameters['model_type'] == 'ae':
        Trainer = TrainerAE(
            model,
            model_hyperparameters,
            cfg.learning_rate,
            cfg.learning_decay,
            cfg.weight_decay,
            save_path,
        )
    elif model_hyperparameters['model_type'] == 'vae':
        Trainer = TrainerVAE(
            model,
            model_hyperparameters,
            cfg.learning_rate,
            cfg.learning_decay,
            cfg.weight_decay,
            save_path,
            cfg.n_epochs // 5
        )
    elif model_hyperparameters['model_type'] == 'cvae':
        Trainer = TrainerCVAE(
            model,
            model_hyperparameters,
            cfg.learning_rate,
            cfg.learning_decay,
            cfg.weight_decay,
            save_path,
            cfg.n_epochs // 5,
            cfg.n_classes,
            cfg.class_weight
        )
    elif model_hyperparameters['model_type'] == 'svae':
        Trainer = TrainerSVAE(
            model,
            model_hyperparameters,
            cfg.learning_rate,
            cfg.learning_decay,
            cfg.weight_decay,
            save_path,
            cfg.n_epochs // 5,
            cfg.n_classes,
            cfg.class_weight,
            torch.nn.BCEWithLogitsLoss(reduction='sum')
        )
    else:
        raise NotImplementedError("Model type not implemented")

    # Training loop
    Trainer.train(cfg.n_epochs, ['train', 'val'], datasets, dataloaders)

    return 0


# Entry point of the script
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SVAE training")

    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data-path', type=str, default='.')
    parser.add_argument('--resume', type=str)

    # Dataset parameters
    parser.add_argument('--n-dof', type=int, default=20)
    parser.add_argument('--seq-len', type=int, default=300)

    # Model parameters
    parser.add_argument('--in-channels', type=int, default=20)
    parser.add_argument('--latent-dim', type=int, default=5)
    parser.add_argument('--n-classes', type=int, default=6)  # Do not set this value to 1
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--class-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--extractor', type=str, default='lstm')
    parser.add_argument('--model-type', type=str, default='ae')
    parser.add_argument('--target', type=str, default='accelerations')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--class-weight', type=float, default=1e1)
    parser.add_argument('--regularization', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--learning-decay', type=float, default=0.984)

    args = parser.parse_args()
    main(args)
