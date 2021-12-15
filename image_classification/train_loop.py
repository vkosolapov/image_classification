import torch
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter

import time
from tqdm import tqdm
import copy

from data import DataLoader
from gradinit.gradinit_utils import gradinit
from augmentation import cutmixup_data, cutmixup_criterion
from scheduler import CyclicCosineDecayLR
from torchcontrib.optim import SWA


class TrainLoop:
    def __init__(
        self,
        experiment_name,
        device,
        datadir,
        batch_size,
        augmentations,
        model,
        optimizer,
        num_epochs,
        criterion,
        accuracy,
        auroc,
        grad_init=None,
        scheduler=None,
        mixup=False,
        cutmix=False,
        cutmixup_alpha=1.0,
        early_stopping=None,
        checkpoint_file=None,
    ):
        self.experiment_name = experiment_name
        self.writer = SummaryWriter(f"./runs/{experiment_name}")
        self.device = device
        self.batch_size = batch_size
        sets = ["train", "val"]
        self.data_loaders = {
            x: DataLoader(datadir, x, batch_size, augmentations) for x in sets
        }
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.accuracy = accuracy
        self.auroc = auroc
        self.grad_init = grad_init
        self.scheduler = scheduler
        self.mixup = mixup
        self.cutmix = cutmix
        self.cutmixup_alpha = cutmixup_alpha
        self.early_stopping = early_stopping
        self.checkpoint_file = checkpoint_file

    def evaluate_minibatch(self):
        self.acc = self.accuracy(self.preds.to("cpu"), self.labels.to("cpu"))
        self.auc = self.auroc(self.probs.to("cpu"), self.labels.to("cpu"))
        self.running_loss += self.loss.item() * self.inputs_size

    def log_minibatch(self, phase, epoch, minibatch):
        dataset_size = self.data_loaders[phase].dataset_size
        self.writer.add_scalar(
            f"batch_loss/{phase}",
            self.loss.item(),
            dataset_size // self.batch_size * epoch + minibatch,
        )
        self.writer.add_scalar(
            f"batch_acc/{phase}",
            self.acc,
            dataset_size // self.batch_size * epoch + minibatch,
        )
        self.writer.add_scalar(
            f"batch_auc/{phase}",
            self.auc,
            dataset_size // self.batch_size * epoch + minibatch,
        )
        self.writer.close()

    def evaluate_epoch(self, phase):
        dataset_size = self.data_loaders[phase].dataset_size
        self.epoch_loss = self.running_loss / dataset_size
        self.epoch_acc = self.accuracy.compute()
        self.accuracy.reset()
        self.epoch_auc = self.auroc.compute()
        self.auroc.reset()

    def log_epoch(self, phase, epoch):
        self.writer.add_scalar(f"epoch_loss/{phase}", self.epoch_loss, epoch)
        self.writer.add_scalar(f"epoch_acc/{phase}", self.epoch_acc, epoch)
        self.writer.add_scalar(f"epoch_auc/{phase}", self.epoch_auc, epoch)
        self.writer.close()
        print(
            "{} Loss: {:.4f} Acc: {:.4f}".format(phase, self.epoch_loss, self.epoch_acc)
        )

    def log_norm(self, phase, epoch, minibatch):
        dataset_size = self.data_loaders[phase].dataset_size
        total_param_norm = 0
        total_grad_norm = 0
        for p in self.model.parameters():
            param_norm = p.detach().data.norm(2)
            total_param_norm += param_norm.item() ** 2
            grad_norm = p.grad.detach().data.norm(2)
            total_grad_norm += grad_norm.item() ** 2
        total_param_norm = total_param_norm ** (0.5)
        total_grad_norm = total_grad_norm ** (0.5)
        self.writer.add_scalar(
            f"norm/param",
            total_param_norm,
            dataset_size // self.batch_size * epoch + minibatch,
        )
        self.writer.add_scalar(
            f"norm/grad",
            total_grad_norm,
            dataset_size // self.batch_size * epoch + minibatch,
        )
        self.writer.close()

    def train_epoch(self, epoch):
        self.model.train()
        self.running_loss = 0.0
        for i, (inputs, labels) in tqdm(
            enumerate(self.data_loaders["train"].data_loader)
        ):
            self.inputs = inputs.to(self.device)
            self.inputs_size = inputs.size(0)
            self.labels = labels.to(self.device)
            if self.mixup or self.cutmix:
                self.inputs, labels_a, labels_b, lambda_ = cutmixup_data(
                    self.inputs,
                    self.labels,
                    self.mixup,
                    self.cutmix,
                    alpha=self.cutmixup_alpha,
                    device=self.device,
                )
                self.inputs, labels_a, labels_b = map(
                    Variable, (self.inputs, labels_a, labels_b)
                )
            with torch.set_grad_enabled(True):
                outputs = self.model(self.inputs)
                if self.mixup or self.cutmix:
                    self.loss = cutmixup_criterion(
                        self.criterion, outputs, labels_a, labels_b, lambda_
                    )
                else:
                    self.loss = self.criterion(outputs, self.labels)
                _, self.preds = torch.max(outputs, 1)
                self.probs = torch.softmax(outputs, 1)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
            self.evaluate_minibatch()
            self.log_minibatch("train", epoch, i)
            self.log_norm("train", epoch, i)
        if self.scheduler:
            self.scheduler.step()
        if isinstance(self.optimizer, SWA):
            if isinstance(self.scheduler, CyclicCosineDecayLR):
                if self.scheduler._restart_flag == True:
                    self.optimizer.update_swa()
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(
                self.data_loaders["train"].data_loader, self.model, device=self.device
            )
        self.evaluate_epoch("train")
        self.log_epoch("train", epoch)
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/checkpoint_{epoch}.pth")

    def test_epoch(self, epoch):
        self.model.eval()
        self.running_loss = 0.0
        for i, (inputs, labels) in tqdm(
            enumerate(self.data_loaders["val"].data_loader)
        ):
            self.inputs = inputs.to(self.device)
            self.labels = labels.to(self.device)
            with torch.set_grad_enabled(False):
                outputs = self.model(self.inputs)
                self.loss = self.criterion(outputs, self.labels)
                _, self.preds = torch.max(outputs, 1)
                self.probs = torch.softmax(outputs, 1)
            self.evaluate_minibatch()
            self.log_minibatch("val", epoch, i)
        self.evaluate_epoch("val")
        self.log_epoch("val", epoch)
        return self.epoch_acc

    def train_model(self):
        since = time.time()
        if self.grad_init:
            gradinit(
                self.model, self.data_loaders["train"].data_loader, args=self.grad_init
            )
        early_stopping_counter = 0
        if self.checkpoint_file:
            checkpoint = torch.load(self.checkpoint_file)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optim_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            print("Epoch {}/{}".format(epoch, self.num_epochs - 1))
            print("-" * 10)
            self.train_epoch(epoch)
            epoch_acc = self.test_epoch(epoch)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if (
                    self.early_stopping
                    and early_stopping_counter >= self.early_stopping
                ):
                    break
            print()
        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_acc))
        self.model.load_state_dict(best_model_wts)
        torch.save(
            self.model.state_dict(), f"checkpoints/final_{self.experiment_name}.pt"
        )
        return self.model
