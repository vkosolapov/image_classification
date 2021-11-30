from math import degrees
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import random
from tqdm import tqdm
import os
import copy

import model

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

batch_size = 64
num_epochs = 20
pretrained = False

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation(degrees=30),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.25))
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

datadir = "data/imagenette2"
sets = ["train", "val"]
image_datasets = {
    x: datasets.ImageFolder(os.path.join(datadir, x), data_transforms[x]) for x in sets
}
data_loaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    ) for x in sets
}
dataset_sizes = {x: len(image_datasets[x]) for x in sets}
class_names = image_datasets["train"].classes


def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs, checkpoint_file=None):
    since = time.time()
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ["train", "val"]:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for i, (inputs, labels) in tqdm(enumerate(data_loaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                acc = torch.sum(preds == labels.data) / len(preds)
                writer.add_scalar(f"{phase} loss", loss.item(), dataset_sizes[phase] // batch_size * epoch + i)
                writer.add_scalar(f"{phase} acc", acc, dataset_sizes[phase] // batch_size * epoch + i)
                writer.close()
            if phase == "train" and scheduler:
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict()
            }
            torch.save(checkpoint, f"checkpoints/checkpoint_{epoch}.pth")
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "checkpoints/final.pt")
    return model


# model_conv = torchvision.models.vgg11(pretrained=pretrained, num_classes=len(class_names))
# if pretrained:
#    for param in model_conv.parameters():
#        param.requires_grad = False
model_conv = model.vgg(num_classes=len(class_names))
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.001)
scheduler = None

train_model(model_conv, criterion, optimizer_conv, scheduler, num_epochs=num_epochs)
