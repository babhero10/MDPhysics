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


class LossContainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.losses = nn.ModuleDict()
        self.weights = {}

        for name, entry in cfg.items():
            if name == "_target_":
                continue

            self.losses[name] = instantiate(entry.loss)
            self.weights[name] = entry.weight

    def forward(self, preds: dict, targets: dict):
        """
        preds:   dict[str, Tensor]
        targets: dict[str, Tensor]
        """
        total_loss = 0.0
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            loss_val = loss_fn(preds[name], targets[name])
            weighted = self.weights[name] * loss_val

            loss_dict[name] = loss_val
            total_loss += weighted

        loss_dict["total"] = total_loss
        return total_loss, loss_dict


def build_criterion(cfg_criterion):
    """Instantiate scheduler from Hydra config"""
    return instantiate(cfg_criterion)
