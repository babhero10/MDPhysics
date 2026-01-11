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
    CheckpointManager,
    log_validation_visualizations,
    log_mdphysics_visualizations,
)

from utils.logger import Logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
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

    fixed_blur = fixed_batch["blur"].to(device)[:2]
    fixed_sharp = fixed_batch["sharp"].to(device)[:2]

    # Instantiate model from 'arch' sub-config
    model = hydra.utils.instantiate(cfg.model.arch).to(device)

    logger.info("Model initalized successfully!")
    logger.info(f"Device used: {device}")

    criterion = torch.nn.MSELoss()

    # optimizer and scheduler
    optimizer = build_optimizer(model, cfg.train.optimizer)
    scheduler = build_scheduler(optimizer, cfg.train.scheduler)

    # metrics
    metrics = build_metrics(cfg.metrics, device)

    checkpoint_mgr = CheckpointManager(
        save_dir=cfg.train.checkpoint.dir,
        mode=cfg.train.checkpoint.mode,
        patience=cfg.train.checkpoint.patience,
        min_delta=cfg.train.checkpoint.min_delta,
    )

    logger.info("Training Started!")

    for epoch in range(cfg.train.epochs):
        logger.info(f"Epoch {epoch + 1} | Training...")
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, metrics
        )

        logger.info(f"Epoch {epoch + 1} | Validating...")
        val_loss, val_metrics = validate(model, val_loader, criterion, device, metrics)

        # Logging to TensorBoard
        writer.add_scalars(
            "metrics/loss", {"train": train_loss, "val": val_loss}, epoch + 1
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
        logger.info(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, {metrics_str}"
        )

        # step scheduler if needed
        scheduler.step()

        current_score = val_metrics[cfg.train.checkpoint.monitor]
        stop_training, is_saved = checkpoint_mgr.step(model, current_score, epoch + 1)

        if is_saved:
            logger.info(f"New best model saved! Score: {current_score:.4f}")

            # Log visualizations
            if cfg.model.get("name") == "MDPhysics":
                log_mdphysics_visualizations(
                    model, writer, epoch + 1, fixed_blur, fixed_sharp, metrics, device
                )
            else:
                log_validation_visualizations(
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
