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


def evaluate_minibatch(preds, labels, loss, running_loss, running_corrects, inputs_size):
    running_loss += loss.item() * inputs_size
    running_corrects += torch.sum(preds == labels.data)
    acc = torch.sum(preds == labels.data) / len(preds)
    return running_loss, running_corrects, acc


def log_minibatch(phase, loss, acc, dataset_size, epoch, minibatch):
    writer.add_scalar(f"{phase} loss", loss.item(), dataset_size // batch_size * epoch + minibatch)
    writer.add_scalar(f"{phase} acc", acc, dataset_size // batch_size * epoch + minibatch)
    writer.close()


def evaluate_epoch(running_loss, running_corrects, dataset_size):
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    return epoch_loss, epoch_acc


def log_epoch(phase, epoch_loss, epoch_acc):
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


def train_epoch(model, data_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for i, (inputs, labels) in tqdm(enumerate(data_loader.data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        running_loss, running_corrects, acc = evaluate_minibatch(
            preds, 
            labels, 
            loss, 
            running_loss, 
            running_corrects, 
            inputs.size(0), 
        )
        log_minibatch("train", loss, acc, data_loader.dataset_size, epoch, i)
    scheduler.step()
    epoch_loss, epoch_acc = evaluate_epoch(running_loss, running_corrects, data_loader.dataset_size)
    log_epoch("train", epoch_loss, epoch_acc)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
    }
    torch.save(checkpoint, f"checkpoints/checkpoint_{epoch}.pth")


def test_epoch(model, data_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for i, (inputs, labels) in tqdm(enumerate(data_loader.data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        running_loss, running_corrects, acc = evaluate_minibatch(
            preds, 
            labels, 
            loss, 
            running_loss, 
            running_corrects, 
            inputs.size(0), 
        )
        log_minibatch("val", loss, acc, data_loader.dataset_size, epoch, i)
    epoch_loss, epoch_acc = evaluate_epoch(running_loss, running_corrects, data_loader.dataset_size)
    log_epoch("val", epoch_loss, epoch_acc)
    return epoch_acc


def train_model(model, data_loaders, criterion, optimizer, scheduler, num_epochs=num_epochs, checkpoint_file=None):
    since = time.time()
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_epoch(model, data_loaders["train"], criterion, optimizer, scheduler, epoch)
        epoch_acc = test_epoch(model, data_loaders["val"], criterion, epoch)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "checkpoints/final.pt")
    return model


if __name__ == '__main__':
    sets = ["train", "val"]
    data_loaders = {x: DataLoader(x, batch_size) for x in sets}

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

    train_model(model_conv, data_loaders, criterion, optimizer_conv, scheduler_conv, num_epochs=num_epochs)
