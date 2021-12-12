import numpy as np
import torch
import torch.nn as nn


class LabelSmoothingFocalLoss(nn.Module):
    def __init__(
        self,
        num_class,
        gamma=0,
        alpha=None,
        balance_index=-1,
        smooth=None,
        size_average=True,
    ):
        super().__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth
        self.size_average = size_average
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            if len(self.alpha) != self.num_class:
                raise AssertionError
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError("Not support alpha type")
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logits, labels):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)
            logits = logits.permute(0, 2, 1).contiguous()
            logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1, 1)
        epsilon = 1e-10
        alpha = self.alpha.to(logits.device)
        idx = labels.cpu().long()
        one_hot_key = torch.FloatTensor(labels.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key = one_hot_key.to(logits.device)
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (self.num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logits).sum(1) + epsilon
        logpt = pt.log()
        gamma = self.gamma
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
