import os
import numpy as np
import torch
from torchvision import datasets, transforms
import albumentations as A


datadir = "data/imagenette2"

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

augmentation_pipeline = A.Compose(
    [
        A.OneOf(
            [
                A.ShiftScaleRotate(),
                A.RandomResizedCrop(256, 256),
                A.HorizontalFlip(),
                # A.VerticalFlip(),
                # A.Transpose(),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.RandomGamma(),
                A.RGBShift(),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1),
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20,
                ),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.ElasticTransform(
                    alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                ),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ],
            p=0.5,
        ),
        A.OneOf([A.GaussNoise(p=0.5), A.Blur(p=0.5),], p=0.5),
    ],
    p=1,
)


def augment(img):
    img = np.asarray(img)
    return augmentation_pipeline(image=img)["image"]


data_transforms = {
    "train": transforms.Compose(
        [
            augment,
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


class DataLoader:
    def __init__(self, phase, batch_size):
        image_dataset = datasets.ImageFolder(
            os.path.join(datadir, phase), data_transforms[phase]
        )
        data_loader = torch.utils.data.DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=(phase == "train"),
            drop_last=(phase == "train"),
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
        )
        self.data_loader = data_loader
        self.dataset_size = len(image_dataset)
        self.class_names = image_dataset.classes
