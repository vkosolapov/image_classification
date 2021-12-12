import torch
import torch.nn as nn
import torchvision
import torchmetrics
import torch.optim as optim
from torchcontrib.optim import SWA
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time
import random
from tqdm import tqdm
import copy

from data import DataLoader
from model import ResNet
from loss import LabelSmoothingFocalLoss
from optimizer import Ranger
from scheduler import CyclicCosineDecayLR
from gradinit.gradinit_utils import gradinit

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_name = "006_add_gradinit_and_swa"
writer = SummaryWriter(f"./runs/{experiment_name}")

num_classes = 10
batch_size = 64
num_epochs = 200
pretrained = False
accuracy = torchmetrics.Accuracy(num_classes=num_classes)
auroc = torchmetrics.AUROC(num_classes=num_classes, average="macro")


def evaluate_minibatch(preds, probs, labels, loss, running_loss, inputs_size):
    acc = accuracy(preds.to("cpu"), labels.to("cpu"))
    auc = auroc(probs.to("cpu"), labels.to("cpu"))
    running_loss += loss.item() * inputs_size
    return acc, auc, running_loss


def log_minibatch(phase, loss, acc, auc, dataset_size, epoch, minibatch):
    writer.add_scalar(
        f"batch_loss/{phase}",
        loss.item(),
        dataset_size // batch_size * epoch + minibatch,
    )
    writer.add_scalar(
        f"batch_acc/{phase}", acc, dataset_size // batch_size * epoch + minibatch
    )
    writer.add_scalar(
        f"batch_auc/{phase}", auc, dataset_size // batch_size * epoch + minibatch
    )
    writer.close()


def evaluate_epoch(running_loss, dataset_size):
    epoch_loss = running_loss / dataset_size
    epoch_acc = accuracy.compute()
    accuracy.reset()
    epoch_auc = auroc.compute()
    auroc.reset()
    return epoch_loss, epoch_acc, epoch_auc


def log_epoch(phase, epoch_loss, epoch_acc, epoch_auc, epoch):
    writer.add_scalar(f"epoch_loss/{phase}", epoch_loss, epoch)
    writer.add_scalar(f"epoch_acc/{phase}", epoch_acc, epoch)
    writer.add_scalar(f"epoch_auc/{phase}", epoch_auc, epoch)
    writer.close()
    print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))


def log_norm(model, dataset_size, epoch, minibatch):
    total_param_norm = 0
    total_grad_norm = 0
    for p in model.parameters():
        param_norm = p.detach().data.norm(2)
        total_param_norm += param_norm.item() ** 2
        grad_norm = p.grad.detach().data.norm(2)
        total_grad_norm += grad_norm.item() ** 2
    total_param_norm = total_param_norm ** (0.5)
    total_grad_norm = total_grad_norm ** (0.5)
    writer.add_scalar(
        f"norm/param", total_param_norm, dataset_size // batch_size * epoch + minibatch
    )
    writer.add_scalar(
        f"norm/grad", total_grad_norm, dataset_size // batch_size * epoch + minibatch
    )
    writer.close()


def train_epoch(model, data_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in tqdm(enumerate(data_loader.data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc, auc, running_loss = evaluate_minibatch(
            preds, probs, labels, loss, running_loss, inputs.size(0),
        )
        log_minibatch("train", loss, acc, auc, data_loader.dataset_size, epoch, i)
        log_norm(model, data_loader.dataset_size, epoch, i)
    if scheduler:
        scheduler.step()
    if isinstance(optimizer, SWA):
        if isinstance(scheduler, CyclicCosineDecayLR):
            if (
                epoch >= scheduler._warmup_epochs + scheduler._init_decay_epochs
                and (epoch + 1) % scheduler._restart_interval == 0
            ):
                optimizer.update_swa()
        optimizer.swap_swa_sgd()
        optimizer.bn_update(data_loader.data_loader, model, device=device)
    epoch_loss, epoch_acc, epoch_auc = evaluate_epoch(
        running_loss, data_loader.dataset_size
    )
    log_epoch("train", epoch_loss, epoch_acc, epoch_auc, epoch)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(checkpoint, f"checkpoints/checkpoint_{epoch}.pth")


def test_epoch(model, data_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    for i, (inputs, labels) in tqdm(enumerate(data_loader.data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, 1)
        acc, auc, running_loss = evaluate_minibatch(
            preds, probs, labels, loss, running_loss, inputs.size(0),
        )
        log_minibatch("val", loss, acc, auc, data_loader.dataset_size, epoch, i)
    epoch_loss, epoch_acc, epoch_auc = evaluate_epoch(
        running_loss, data_loader.dataset_size
    )
    log_epoch("val", epoch_loss, epoch_acc, epoch_auc, epoch)
    return epoch_acc


def train_model(
    model,
    data_loaders,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=num_epochs,
    early_stopping=None,
    checkpoint_file=None,
):
    since = time.time()
    early_stopping_counter = 0
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        train_epoch(
            model, data_loaders["train"], criterion, optimizer, scheduler, epoch
        )
        epoch_acc = test_epoch(model, data_loaders["val"], criterion, epoch)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping and early_stopping_counter >= early_stopping:
                break
        print()
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f"checkpoints/final_{experiment_name}.pt")
    return model


if __name__ == "__main__":
    sets = ["train", "val"]
    data_loaders = {x: DataLoader(x, batch_size) for x in sets}

    model_conv = ResNet("resnet18", num_classes=num_classes)
    model_conv = model_conv.to(device)

    gradinit(model_conv, data_loaders["train"].data_loader)

    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingFocalLoss(num_classes=num_classes, gamma=2, smoothing=0.1)
    optimizer_conv = Ranger(model_conv.parameters(), lr=0.01, weight_decay=0.0001)
    # swa = SWA(optimizer_conv, swa_start=10, swa_freq=5, swa_lr=0.05)
    swa = SWA(optimizer_conv)
    scheduler_conv = CyclicCosineDecayLR(
        optimizer_conv,
        warmup_epochs=5,
        warmup_start_lr=0.005,
        warmup_linear=False,
        init_decay_epochs=5,
        min_decay_lr=0.001,
        restart_lr=0.01,
        restart_interval=10,
    )

    train_model(
        model_conv,
        data_loaders,
        criterion,
        swa,
        scheduler_conv,
        num_epochs=num_epochs,
        early_stopping=20,
    )

