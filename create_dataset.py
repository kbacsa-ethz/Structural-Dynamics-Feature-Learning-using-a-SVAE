import os
from torch.utils.data import DataLoader

from structure_dataset import StructureDataset


def create_dataset(
        data_path,
        n_dof,
        batch_size,
        num_workers
):
    # dataset
    phases = ['train', 'test']

    datasets = {
        "train": StructureDataset(os.path.join(data_path, "train_dataset"), n_dof),
        "test": StructureDataset(os.path.join(data_path, "test_dataset"), n_dof)
    }

    dataloaders = {
        x: DataLoader(dataset=datasets[x], batch_size=batch_size, shuffle=True if x == 'train' else False,
                      num_workers=num_workers, pin_memory=True) for x in
        phases}

    return datasets, dataloaders
