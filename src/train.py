import hydra
from omegaconf import DictConfig, OmegaConf
from dataset.Dataset import BlurDataset
from utils.seed import set_seed
import torch
from utils.training import (
    train_one_epoch,
    validate,
    build_optimizer,
    build_scheduler,
    build_metrics,
    build_model,
    CheckpointManager,
)
from utils.losses import build_criterion
from utils.logger import Logger
from utils.visuals import log_visualizations
from torchinfo import summary


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Optimize for modern GPUs
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    set_seed(cfg.seed)
    # Initialize Logger
    _ = Logger(cfg.logger)
    logger = Logger.get_logger()
    writer = Logger.get_writer()

    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    train_dataset = BlurDataset(cfg.dataset, "train")
    val_dataset = BlurDataset(cfg.dataset, "val")

    logger.info("Data loaded successfully!")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=cfg.dataset.num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size_val,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=cfg.dataset.num_workers,
    )

    device = torch.device(cfg.device)

    fixed_batch = next(iter(val_loader))

    # Select 4 random indices from the batch
    batch_size = fixed_batch["blur"].size(0)
    indices = torch.randperm(batch_size)[:4]

    fixed_blur = fixed_batch["blur"][indices].to(device)
    fixed_sharp = fixed_batch["sharp"][indices].to(device)

    # Instantiate model from 'arch' sub-config
    model = build_model(cfg, device)

    # Get summary as string
    model_summary = summary(
        model,
        input_size=(
            cfg.train.batch_size,
            3,
            cfg.dataset.img_size[0],
            cfg.dataset.img_size[1],
        ),
        depth=4,
        col_names=["input_size", "output_size", "num_params", "trainable"],
    )

    logger.info("Model initalized successfully!")
    logger.info(f"Device used: {device}")
    logger.info(str(model_summary))

    criterion = build_criterion(cfg.train.criterion)

    # optimizer and scheduler
    optimizer = build_optimizer(model, cfg.train.optimizer)
    scheduler = build_scheduler(optimizer, cfg.train.scheduler)

    # metrics
    metrics = build_metrics(cfg.metrics, device)
    metrics_blur = None

    # Check for blur metrics requirement
    model_cfg = getattr(model, "cfg", None)
    if model_cfg:
        use_blurring = getattr(model_cfg, "use_blurring_block", False)
        blur_type = getattr(model_cfg, "used_image_blurring_block", "")
        if use_blurring and blur_type == "GT":
            metrics_blur = build_metrics(cfg.metrics, device)
            logger.info("Initialized metrics for Blur Image (Stage 1 monitoring).")

    checkpoint_mgr = CheckpointManager(
        save_dir=cfg.train.checkpoint.dir,
        mode=cfg.train.checkpoint.mode,
        patience=cfg.train.checkpoint.patience,
        min_delta=cfg.train.checkpoint.min_delta,
    )

    # Optimization
    scaler = torch.amp.GradScaler("cuda")

    logger.info("Training Started!")

    for epoch in range(cfg.train.epochs):
        logger.info(f"Epoch {epoch + 1} | Training...")
        train_losses, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            metrics,
            metrics_blur,
        )

        logger.info(f"Epoch {epoch + 1} | Validating...")
        val_losses, val_metrics = validate(
            model, val_loader, criterion, device, metrics, metrics_blur
        )

        # Logging to TensorBoard
        # Log total loss
        writer.add_scalars(
            "metrics/loss",
            {
                "train": train_losses.get("total", 0.0),
                "val": val_losses.get("total", 0.0),
            },
            epoch + 1,
        )

        # Log individual losses
        for loss_name in train_losses:
            if loss_name != "total":
                writer.add_scalars(
                    f"losses/{loss_name}",
                    {
                        "train": train_losses[loss_name],
                        "val": val_losses.get(loss_name, 0.0),
                    },
                    epoch + 1,
                )

        for metric_name, train_val in train_metrics.items():
            val_val = val_metrics.get(metric_name)
            if val_val is not None:
                writer.add_scalars(
                    f"metrics/{metric_name}",
                    {"train": train_val, "val": val_val},
                    epoch + 1,
                )

        # Log to console
        metrics_str = ", ".join(
            [
                f"train_{k}={v:.4f}, val_{k}={val_metrics[k]:.4f}"
                for k, v in train_metrics.items()
            ]
        )

        train_total = train_losses.get("total", 0.0)
        val_total = val_losses.get("total", 0.0)

        logger.info(
            f"Epoch {epoch + 1}: train_loss={train_total:.4f}, val_loss={val_total:.4f}, {metrics_str}"
        )

        # step scheduler if needed
        scheduler.step()

        current_score = val_metrics[cfg.train.checkpoint.monitor]
        stop_training, is_saved = checkpoint_mgr.step(model, current_score, epoch + 1)

        if is_saved:
            logger.info(f"New best model saved! Score: {current_score:.4f}")

            # Log visualizations
            log_visualizations(
                model, writer, epoch + 1, fixed_blur, fixed_sharp, metrics, device
            )

        if stop_training:
            logger.warning(f"Early stopping triggered at epoch {epoch + 1}")
            break

    logger.info(
        f"Training finished. Best model saved at epoch {checkpoint_mgr.best_epoch} with score {checkpoint_mgr.best_score:.4f}"
    )
    logger.info(f"Best model path: {checkpoint_mgr.best_path}")


if __name__ == "__main__":
    main()
