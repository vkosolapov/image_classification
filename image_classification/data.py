import os
import numpy as np
from functools import partial
import torch
from torchvision import datasets, transforms


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def augment(img, augmentation_pipeline):
    img = np.asarray(img)
    return augmentation_pipeline(image=img)["image"]


class DataLoader:
    def __init__(self, datadir, phase, batch_size, augmentations):
        data_transforms = self.transforms(augmentations)
        image_dataset = datasets.ImageFolder(
            os.path.join(datadir, phase), data_transforms[phase]
        )
        data_loader = torch.utils.data.DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=(phase == "train"),
            drop_last=(phase == "train"),
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )
        self.data_loader = data_loader
        self.dataset_size = len(image_dataset)
        self.class_names = image_dataset.classes

    def transforms(self, augmentations):
        return {
            "train": transforms.Compose(
                [
                    partial(augment, augmentation_pipeline=augmentations),
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }
