
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class IRDropDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform

        # Load all file names (0.npy, 1.npy, ...)
        self.file_list = sorted(os.listdir(os.path.join(base_dir, "input_power_grid")))
        self.length = len(self.file_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx_str = self.file_list[idx].replace(".npy", "")

        # Load 3 input channels
        power_grid = np.load(os.path.join(self.base_dir, "input_power_grid", idx_str + ".npy"))
        cell_density = np.load(os.path.join(self.base_dir, "input_cell_density", idx_str + ".npy"))
        switching = np.load(os.path.join(self.base_dir, "input_switching", idx_str + ".npy"))

        # Stack inputs -> shape (3, H, W)
        X = np.stack([power_grid, cell_density, switching], axis=0).astype(np.float32)

        # Load output IR-drop map
        Y = np.load(os.path.join(self.base_dir, "labels_ir_drop", idx_str + ".npy")).astype(np.float32)
        Y = np.expand_dims(Y, axis=0)  # shape (1, H, W)

        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        if self.transform:
            X, Y = self.transform(X, Y)

        return X, Y


def get_dataloaders(batch_size=16, base_dir="./dataset"):
    dataset = IRDropDataset(base_dir)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
