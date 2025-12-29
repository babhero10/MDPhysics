import hydra
from omegaconf import DictConfig
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
)

# from utils.logger import Logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    train_dataset = BlurDataset(cfg.dataset, "train")
    val_dataset = BlurDataset(cfg.dataset, "val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=8,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=8,
    )

    model = None
    device = torch.device(cfg.device)

    criterion = torch.nn.MSELoss()

    # optimizer and scheduler
    optimizer = build_optimizer(model, cfg.train.optimizer)
    scheduler = build_scheduler(optimizer, cfg.train.scheduler)

    # metrics
    metrics = build_metrics(cfg.metrics, device)

    checkpoint_mgr = CheckpointManager(
        save_dir=cfg.checkpoint.dir,
        mode=cfg.checkpoint.mode,
        patience=cfg.checkpoint.patience,
        min_delta=cfg.checkpoint.min_delta,
    )

    for epoch in range(cfg.train.epochs):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, metrics
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device, metrics)

        # step scheduler if needed
        scheduler.step()

        current_score = val_metrics[cfg.checkpoint.monitor]
        if checkpoint_mgr.step(model, current_score, epoch):
            break


if __name__ == "__main__":
    main()
