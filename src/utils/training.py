import torch
from hydra.utils import instantiate
from pathlib import Path


class CheckpointManager:
    def __init__(
        self,
        save_dir: str,
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best_score = None
        self.counter = 0
        self.best_path = None
        self.stop_training = False

    def step(self, model: torch.nn.Module, current_score: float, epoch: int):
        """
        Call this at the end of each epoch.
        Returns True if early stopping is triggered.
        """
        is_improvement = False

        if self.best_score is None:
            is_improvement = True
        else:
            if (
                self.mode == "min"
                and (self.best_score - current_score) > self.min_delta
            ):
                is_improvement = True
            elif (
                self.mode == "max"
                and (current_score - self.best_score) > self.min_delta
            ):
                is_improvement = True

        if is_improvement:
            # save model
            self.best_score = current_score
            self.counter = 0
            path = self.save_dir / f"best_model_epoch{epoch}.pt"
            torch.save(model.state_dict(), path)
            self.best_path = path
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True

        return self.stop_training


def build_metrics(cfg_metrics, device):
    metrics = {}
    for name, metric_cfg in cfg_metrics.items():
        metric = instantiate(metric_cfg)
        metric = metric.to(device)
        metrics[name] = metric

    return metrics


def build_optimizer(model, cfg_opt):
    """Instantiate optimizer from Hydra config"""
    return instantiate(cfg_opt, params=model.parameters())


def build_scheduler(optimizer, cfg_sched):
    """Instantiate scheduler from Hydra config"""
    return instantiate(cfg_sched, optimizer=optimizer)


def train_one_epoch(model, train_loader, optimizer, criterion, device, metrics=None):
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    for metric in (metrics or {}).values():
        metric.reset()

    running_loss = 0.0

    for batch in train_loader:
        blur = batch["blur"].to(device)
        sharp = batch["sharp"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # FP16 forward
            pred = model(blur)
            pred = torch.clamp(pred, 0, 1)
            loss = criterion(pred, sharp)

        # scale gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # update metrics (keep in FP32 for accuracy)
        if metrics:
            for metric in metrics.values():
                metric.update(pred.detach(), sharp)

    results = {
        name: metric.compute().item() for name, metric in (metrics or {}).items()
    }
    avg_loss = running_loss / len(train_loader)

    return avg_loss, results


@torch.no_grad()
def validate(model, val_loader, criterion, device, metrics=None):
    model.eval()
    for metric in (metrics or {}).values():
        metric.reset()

    running_loss = 0.0

    for batch in val_loader:
        blur = batch["blur"].to(device)
        sharp = batch["sharp"].to(device)

        with torch.cuda.amp.autocast():
            pred = model(blur)
            pred = torch.clamp(pred, 0, 1)
            loss = criterion(pred, sharp)

        running_loss += loss.item()

        if metrics:
            for metric in metrics.values():
                metric.update(pred, sharp)

    results = {
        name: metric.compute().item() for name, metric in (metrics or {}).items()
    }
    avg_loss = running_loss / len(val_loader)

    return avg_loss, results
