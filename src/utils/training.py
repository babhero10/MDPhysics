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
        self.best_epoch = -1
        self.stop_training = False

    def step(self, model: torch.nn.Module, current_score: float, epoch: int):
        """
        Call this at the end of each epoch.
        Returns (stop_training, is_saved).
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
            self.best_epoch = epoch
            self.counter = 0
            path = self.save_dir / f"best_model_epoch{epoch}.pt"
            torch.save(model.state_dict(), path)
            self.best_path = path
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True

        return self.stop_training, is_improvement


def build_metrics(cfg_metrics, device):
    metrics = {}
    for name, metric_cfg in cfg_metrics.items():
        metric = instantiate(metric_cfg)
        metric = metric.to(device)
        metrics[name] = metric

    return metrics


def build_optimizer(model, cfg_opt):
    """Instantiate optimizer from Hydra config"""
    return instantiate(
        cfg_opt, params=filter(lambda p: p.requires_grad, model.parameters())
    )


def build_scheduler(optimizer, cfg_sched):
    """Instantiate scheduler from Hydra config"""
    return instantiate(cfg_sched, optimizer=optimizer)


def build_model(cfg, device):
    model = instantiate(cfg.model.arch).to(device)

    if cfg.train.freeze is not None:
        for name, param in model.named_parameters():
            for rule in cfg.train.freeze:
                if name.startswith(rule):
                    param.requires_grad = False

    if cfg.train.last_checkpoint is not None:
        model.load_state_dict(torch.load(cfg.train.last_checkpoint))

    return model


def train_one_epoch(model, train_loader, optimizer, criterion, device, metrics=None):
    model.train()
    scaler = torch.amp.GradScaler("cuda")

    for metric in (metrics or {}).values():
        metric.reset()

    running_loss_dict = {}
    total_samples = 0

    for batch in train_loader:
        blur = batch["blur"].to(device)
        sharp = batch["sharp"].to(device)
        batch_size = blur.size(0)

        optimizer.zero_grad()

        # Construct targets dictionary
        targets = {"blur_image": blur, "sharp_image": sharp}

        with torch.amp.autocast("cuda"):  # FP16 forward
            pred = model(blur)
            loss, loss_dict = criterion(pred, targets)

        # scale gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses (weighted by batch size)
        for k, v in loss_dict.items():
            current = running_loss_dict.get(k, 0.0)
            running_loss_dict[k] = current + v.item() * batch_size
        
        total_samples += batch_size

        # update metrics (keep in FP32 for accuracy)
        if metrics:
            # Metrics usually expect sharp predictions vs sharp targets
            # Check what pred["sharp_image"] contains.
            if "sharp_image" in pred:
                for metric in metrics.values():
                    metric.update(pred["sharp_image"].detach(), sharp)

    results = {
        name: metric.compute().item() for name, metric in (metrics or {}).items()
    }
    
    avg_loss_dict = {k: v / total_samples for k, v in running_loss_dict.items()}

    return avg_loss_dict, results


@torch.no_grad()
def validate(model, val_loader, criterion, device, metrics=None):
    model.eval()
    for metric in (metrics or {}).values():
        metric.reset()

    running_loss_dict = {}
    total_samples = 0

    for batch in val_loader:
        blur = batch["blur"].to(device)
        sharp = batch["sharp"].to(device)
        batch_size = blur.size(0)

        targets = {"blur_image": blur, "sharp_image": sharp}

        with torch.amp.autocast("cuda"):
            pred = model(blur)
            loss, loss_dict = criterion(pred, targets)

        # Accumulate losses (weighted by batch size)
        for k, v in loss_dict.items():
            current = running_loss_dict.get(k, 0.0)
            running_loss_dict[k] = current + v.item() * batch_size
            
        total_samples += batch_size

        if metrics:
            if "sharp_image" in pred:
                for metric in metrics.values():
                    metric.update(pred["sharp_image"], sharp)

    results = {
        name: metric.compute().item() for name, metric in (metrics or {}).items()
    }
    
    avg_loss_dict = {k: v / total_samples for k, v in running_loss_dict.items()}

    return avg_loss_dict, results
