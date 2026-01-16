import torch
import torch.nn as nn
from hydra.utils import instantiate


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        loss = torch.sqrt((pred - target) ** 2 + self.epsilon**2)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class L1Loss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        return self.loss(pred, target)


class FFTLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred.float(), dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        loss = torch.nn.functional.l1_loss(
            pred_fft, target_fft, reduction=self.reduction
        )
        return loss


class CombinedMDTLoss(nn.Module):
    def __init__(self, l1_weight=1.0, fft_weight=0.1, reduction="mean"):
        super().__init__()
        self.l1_loss = L1Loss(reduction=reduction)
        self.fft_loss = FFTLoss(reduction=reduction)
        self.l1_weight = l1_weight
        self.fft_weight = fft_weight

    def forward(self, pred, target):
        l_pix = self.l1_loss(pred, target)
        l_fft = self.fft_loss(pred, target)
        return self.l1_weight * l_pix + self.fft_weight * l_fft


class LossContainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.losses = nn.ModuleDict()
        self.weights = {}

        for name, entry in cfg.items():
            if name == "_target_":
                continue

            self.losses[name] = entry.loss
            self.weights[name] = entry.weight

    def forward(self, preds: dict, targets: dict):
        """
        preds:   dict[str, Tensor]
        targets: dict[str, Tensor]
        """
        total_loss = 0.0
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            if name not in preds:
                continue

            loss_val = loss_fn(preds[name], targets[name])
            weighted = self.weights[name] * loss_val

            loss_dict[name] = loss_val
            total_loss += weighted

        loss_dict["total"] = total_loss
        return total_loss, loss_dict


def build_criterion(cfg_criterion):
    """Instantiate scheduler from Hydra config"""
    return instantiate(cfg_criterion)
