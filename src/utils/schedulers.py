import math
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWarmupLR(LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.

    The learning rate schedule:
    1. Warmup phase (0 to warmup_epochs): Linear increase from warmup_lr to base_lr
    2. Cosine phase (warmup_epochs to T_max): Cosine decay from base_lr to eta_min

    Args:
        optimizer: Wrapped optimizer
        T_max: Total number of epochs (including warmup)
        eta_min: Minimum learning rate after cosine decay (default: 1e-6)
        warmup_epochs: Number of warmup epochs (default: 10)
        warmup_lr: Starting learning rate for warmup (default: 1e-6)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 1e-6,
        warmup_epochs: int = 10,
        warmup_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup: interpolate from warmup_lr to base_lr
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_lr + alpha * (base_lr - self.warmup_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            # Adjust epoch count to start from 0 after warmup
            cosine_epoch = self.last_epoch - self.warmup_epochs
            cosine_T_max = self.T_max - self.warmup_epochs

            # Cosine annealing formula
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * cosine_epoch / cosine_T_max)) / 2
                for base_lr in self.base_lrs
            ]
