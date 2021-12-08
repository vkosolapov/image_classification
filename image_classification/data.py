import os
import numpy as np
import torch
from torchvision import datasets, transforms

datadir = "data/imagenette2"

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1))
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}


class DataLoader():
    def __init__(self, phase, batch_size):
        image_dataset = datasets.ImageFolder(os.path.join(datadir, phase), data_transforms[phase])
        data_loader = torch.utils.data.DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=(phase=="train"),
            drop_last=(phase=="train"),
            num_workers=1,
            persistent_workers=True,
            pin_memory=True
        )
        self.data_loader = data_loader
        self.dataset_size = len(image_dataset)
        self.class_names = image_dataset.classes
