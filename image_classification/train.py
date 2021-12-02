import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time
import random
from tqdm import tqdm
import copy

from data import DataLoader
from model import VGG
from optimizer import Ranger
from scheduler import CyclicCosineDecayLR

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

num_classes = 10
batch_size = 64
num_epochs = 20
pretrained = False

sets = ["train", "val"]
data_loaders = {x: DataLoader(x, batch_size) for x in sets}


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
            for i, (inputs, labels) in tqdm(enumerate(data_loaders[phase].data_loader)):
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
                writer.add_scalar(f"{phase} loss", loss.item(), data_loaders[phase].dataset_size // batch_size * epoch + i)
                writer.add_scalar(f"{phase} acc", acc, data_loaders[phase].dataset_size // batch_size * epoch + i)
                writer.close()
            if phase == "train" and scheduler:
                scheduler.step()
            epoch_loss = running_loss / data_loaders[phase].dataset_size
            epoch_acc = running_corrects.double() / data_loaders[phase].dataset_size
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
model_conv = torchvision.models.resnet18(pretrained=pretrained)
if pretrained:
    for param in model_conv.parameters():
        param.requires_grad = False
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_classes)
# model_conv = model.VGG(num_classes=len(class_names))

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.001)
optimizer_conv = Ranger(model_conv.parameters(), lr=0.01)
scheduler_conv = CyclicCosineDecayLR(
    optimizer_conv, 
    warmup_start_lr=0.005,
    init_decay_epochs=5,
    min_decay_lr=0.001,
    restart_lr=0.01,
    restart_interval=5
)

train_model(model_conv, criterion, optimizer_conv, scheduler_conv, num_epochs=num_epochs)
