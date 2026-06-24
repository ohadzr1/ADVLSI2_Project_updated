# Loading of Training dataset and data augmentation


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import zipfile
from pathlib import Path

COLAB_ROOT = Path(os.environ.get("COLAB_ROOT", "/content"))
TRAINING_ZIP = Path(os.environ.get("TRAINING_ZIP", COLAB_ROOT / "training_dataset.zip"))
DATA_DIR = Path(os.environ.get("TRAINING_DATA_DIR", COLAB_ROOT / "data"))

# Unzip dataset
with zipfile.ZipFile(str(TRAINING_ZIP), 'r') as zip_ref:
    zip_ref.extractall(str(DATA_DIR))

class DRCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['clean', 'dirty']
        self.file_list = []
        for idx, cls in enumerate(self.classes):
            path = os.path.join(self.root_dir, cls)
            for f in os.listdir(path):
                if f.endswith('.npy'):
                    self.file_list.append((os.path.join(path, f), idx))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        matrix = np.load(file_path).astype(np.float32)
        matrix = torch.from_numpy(matrix).unsqueeze(0) # Channel dimension
        if self.transform:
            matrix = self.transform(matrix)
        return matrix, label

# Industry Standard Augmentation + Random Shift
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    # Translation of up to 10% (20px) to prevent spatial bias
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
])