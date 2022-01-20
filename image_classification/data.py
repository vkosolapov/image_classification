import os
import numpy as np
from functools import partial
import torch
from torchvision import datasets, transforms
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Convert, NormalizeImage
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def augment(img, augmentation_pipeline):
    img = np.asarray(img)
    return augmentation_pipeline(image=img)["image"]


class Augmentation(Operation):
    def __init__(self, phase, augmentations):
        super().__init__()
        self.phase = phase
        self.augmentations = augmentations
    
    def generate_code(self):
        if self.phase == "train":
            return partial(augment, augmentation_pipeline=self.augmentations)
        else:
            return lambda x: x
    
    def declare_state_and_memory(self, previous_state):
        return (previous_state, None)
        # return (previous_state, AllocationQuery(shape=previous_state.shape, dtype=np.float32))


class DataLoader:
    def __init__(self, datadir, phase, device, batch_size, augmentations):
        image_dataset = datasets.ImageFolder(
            os.path.join(datadir, phase),
        )
        ffcv_path = f"{datadir}_ffcv/{phase}.beton"
        writer = DatasetWriter(
            ffcv_path, 
            {
                'image': RGBImageField(max_resolution=256, jpeg_quality=90), 
                'label': IntField()
            }
        )
        writer.from_indexed_dataset(image_dataset)
        order = OrderOption.RANDOM if phase == "train" else OrderOption.SEQUENTIAL
        data_loader = Loader(
            ffcv_path, 
            batch_size=batch_size, 
            num_workers=8, 
            order=order,
            pipelines={
                'image': 
                [
                    RandomResizedCropRGBImageDecoder((224, 224)),
                    Augmentation(phase, augmentations),
                    ToTensor(), 
                    ToDevice(device), 
                    ToTorchImage(), 
                    Convert(torch.float16), 
                    NormalizeImage(mean, std, np.float16),
                ]
            }
        )
        self.data_loader = data_loader
        self.dataset_size = len(image_dataset)
        self.class_names = image_dataset.classes
