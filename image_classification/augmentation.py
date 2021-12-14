import numpy as np
import torch


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmixup_data(x, y, mixup=False, cutmix=False, alpha=1.0, device="cuda"):
    if mixup:
        mixup_lambda = np.random.beta(alpha, alpha)
    else:
        mixup_lambda = 1
    if cutmix:
        cutmix_lambda = np.random.beta(alpha, alpha)
    else:
        cutmix_lambda = 1
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), cutmix_lambda)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    if mixup_lambda >= cutmix_lambda:
        mixed_x = mixup_lambda * x + (1 - mixup_lambda) * x[index, :]
        result_lambda = mixup_lambda
    else:
        mixed_x = torch.tensor(x)
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        result_lambda = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2])
        )
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, result_lambda


def cutmixup_criterion(criterion, pred, y_a, y_b, result_lambda):
    return result_lambda * criterion(pred, y_a) + (1 - result_lambda) * criterion(
        pred, y_b
    )
