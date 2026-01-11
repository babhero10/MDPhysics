import torch
from hydra.utils import instantiate
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils.vis import visualize_depth, visualize_motion_field

matplotlib.use("Agg")


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
    return instantiate(cfg_opt, params=filter(lambda p: p.requires_grad, model.parameters()))


def build_scheduler(optimizer, cfg_sched):
    """Instantiate scheduler from Hydra config"""
    return instantiate(cfg_sched, optimizer=optimizer)


def train_one_epoch(model, train_loader, optimizer, criterion, device, metrics=None):
    model.train()
    scaler = torch.amp.GradScaler("cuda")

    for metric in (metrics or {}).values():
        metric.reset()

    running_loss = 0.0

    for batch in train_loader:
        blur = batch["blur"].to(device)
        sharp = batch["sharp"].to(device)
        dist = (
            torch.tensor(np.array([0.5, 0.5, 0.5, 0.5]).reshape(1, 4))
            .float()
            .repeat([blur.size(0), 1])
            .to(device)
        )

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):  # FP16 forward
            pred = model(blur, dist)
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


def create_validation_figure(blur_imgs, sharp_imgs, pred_imgs, metrics_dict=None):
    """
    Creates a matplotlib figure comparing Blur, Pred, and Sharp images.
    metrics_dict: dict (global metrics) or list of dicts (per-image metrics).
    """

    def to_numpy(t):
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t.detach().cpu().permute(0, 2, 3, 1).numpy()

    blur_imgs = to_numpy(blur_imgs)
    sharp_imgs = to_numpy(sharp_imgs)
    pred_imgs = to_numpy(pred_imgs)

    batch_size = blur_imgs.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))

    # Handle single sample case (make axes 2D)
    if batch_size == 1:
        axes = np.array([axes])  # Now shape (1, 3)

    for i in range(batch_size):
        # Determine title for this specific image
        title_str = ""
        if metrics_dict:
            if isinstance(metrics_dict, list) and i < len(metrics_dict):
                # Per-image metrics
                metric_strs = [f"{k}: {v:.4f}" for k, v in metrics_dict[i].items()]
                title_str = "\n".join(metric_strs)
            elif isinstance(metrics_dict, dict):
                # Global metrics (fallback)
                metric_strs = [f"{k}: {v:.4f}" for k, v in metrics_dict.items()]
                title_str = " | ".join(metric_strs)

        ax_blur = axes[i][0]
        ax_pred = axes[i][1]
        ax_sharp = axes[i][2]

        ax_blur.imshow(blur_imgs[i])
        ax_blur.set_title("Blurred")
        ax_blur.axis("off")

        ax_pred.imshow(pred_imgs[i])
        ax_pred.set_title(f"Restored\n{title_str}" if title_str else "Restored")
        ax_pred.axis("off")

        ax_sharp.imshow(sharp_imgs[i])
        ax_sharp.set_title("Sharp")
        ax_sharp.axis("off")

    plt.tight_layout()
    return fig


def log_validation_visualizations(
    model, writer, epoch, blur_imgs, sharp_imgs, metrics, device
):
    """
    Runs inference on the provided batch, calculates metrics per image,
    and logs separate figures to TensorBoard.
    """
    model.eval()
    with torch.no_grad():
        # Ensure inputs are on device
        blur_imgs = blur_imgs.to(device)
        sharp_imgs = sharp_imgs.to(device)
        dist = (
            torch.tensor(np.array([0.5, 0.5, 0.5, 0.5]).reshape(1, 4))
            .float()
            .repeat([blur_imgs.size(0), 1])
            .to(device)
        )

        pred_imgs = model(blur_imgs, dist)
        pred_imgs = torch.clamp(pred_imgs, 0, 1)

        # Loop over each image in the batch
        for i in range(blur_imgs.size(0)):
            # Slice to keep dimensions (1, C, H, W)
            b = blur_imgs[i : i + 1]
            s = sharp_imgs[i : i + 1]
            p = pred_imgs[i : i + 1]

            # Calculate metrics for this image
            img_metrics = {}
            for name, metric in metrics.items():
                metric.reset()
                # metric.update expects (preds, target)
                metric.update(p, s)
                img_metrics[name] = metric.compute().item()

            # Create figure for this single image
            # create_validation_figure handles list of dicts for titles
            fig = create_validation_figure(b, s, p, metrics_dict=[img_metrics])

            # Log to TensorBoard with unique tag per sample
            writer.add_figure(f"deblurred_image_{i}", fig, epoch)
            plt.close(fig)


def log_mdphysics_visualizations(
    model, writer, epoch, blur_imgs, sharp_imgs, metrics, device
):
    """
    Custom visualization logger for MDPhysics model.
    Handles dictionary output and logs depth + motion field.
    """
    model.eval()
    with torch.no_grad():
        blur_imgs = blur_imgs.to(device)
        sharp_imgs = sharp_imgs.to(device)

        # MDPhysics.forward(img) - does not take dist argument
        outputs = model(blur_imgs)

        # Extract outputs
        pred_imgs = outputs["enhanced_img"]
        pred_imgs = torch.clamp(pred_imgs, 0, 1)
        depth = outputs["depth"]
        motion_field = outputs["motion_field"]

        # Loop over each image in the batch
        for i in range(blur_imgs.size(0)):
            b = blur_imgs[i : i + 1]
            s = sharp_imgs[i : i + 1]
            p = pred_imgs[i : i + 1]
            d = depth[i : i + 1]
            mf = motion_field[i : i + 1]

            # Calculate metrics for this image
            img_metrics = {}
            for name, metric in metrics.items():
                metric.reset()
                metric.update(p, s)
                img_metrics[name] = metric.compute().item()

            # 1. Log Restored Image (Standard)
            fig = create_validation_figure(b, s, p, metrics_dict=[img_metrics])
            writer.add_figure(f"deblurred_image_{i}", fig, epoch)
            plt.close(fig)

            # 2. Log Depth Map
            # visualize_depth returns RGB numpy array (H, W, 3)
            d_vis = visualize_depth(d)
            writer.add_image(f"depth_map_{i}", d_vis, epoch, dataformats="HWC")

            # 3. Log Motion Field
            # visualize_motion_field returns RGB numpy array (H, W, 3)
            mf_vis = visualize_motion_field(mf)
            writer.add_image(f"motion_field_{i}", mf_vis, epoch, dataformats="HWC")


@torch.no_grad()
def validate(model, val_loader, criterion, device, metrics=None):
    model.eval()
    for metric in (metrics or {}).values():
        metric.reset()

    running_loss = 0.0

    for batch in val_loader:
        blur = batch["blur"].to(device)
        sharp = batch["sharp"].to(device)
        dist = (
            torch.tensor(np.array([0.5, 0.5, 0.5, 0.5]).reshape(1, 4))
            .float()
            .repeat([blur.size(0), 1])
            .to(device)
        )

        with torch.amp.autocast("cuda"):
            pred = model(blur, dist)
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
