import hydra
from omegaconf import DictConfig, OmegaConf
from dataset.Dataset import BlurDataset
from utils.training import build_model, build_metrics
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import Logger
import matplotlib.pyplot as plt
from pathlib import Path
import csv


@hydra.main(version_base=None, config_path="../configs", config_name="test")
def main(cfg: DictConfig):
    # Set device
    device = torch.device(cfg.device)

    # Initialize Logger
    _ = Logger(cfg.logger)
    logger = Logger.get_logger()

    logger.info("Test Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Load Test Dataset
    test_dataset = BlurDataset(cfg.dataset, "test")

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
    )

    # Build Model
    model = build_model(cfg, device)
    model.eval()

    # Metrics
    metrics = build_metrics(cfg.metrics, device)

    # Check if checkpoint was loaded
    last_ckpt = OmegaConf.select(cfg, "train.last_checkpoint")
    if not last_ckpt:
        logger.warning("No checkpoint loaded! Testing with random weights.")
    else:
        logger.info(f"Loaded checkpoint: {last_ckpt}")

    save_dir = Path("images")
    save_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving visualized results to: {save_dir.resolve()}")

    # CSV File Setup
    csv_path = Path("results.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    # Write Header
    csv_header = ["Image_ID"] + list(metrics.keys())
    csv_writer.writerow(csv_header)

    logger.info("Starting Testing...")

    # Test Loop
    device_type = "cuda" if "cuda" in cfg.device else "cpu"

    image_counter = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            blur = batch["blur"].to(device)
            sharp = batch["sharp"].to(device)

            with torch.amp.autocast(device_type):
                outputs = model(blur)

            if "sharp_image" in outputs:
                pred = outputs["sharp_image"]
                pred = torch.clamp(pred, 0.0, 1.0)

                # Convert to numpy for plotting (Batch, C, H, W) -> (Batch, H, W, C)
                blur_np = blur.permute(0, 2, 3, 1).cpu().numpy()
                pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()
                sharp_np = sharp.permute(0, 2, 3, 1).cpu().numpy()

                batch_size = blur.size(0)
                for i in range(batch_size):
                    # Prepare per-image tensors (keep 4D for metrics: 1, C, H, W)
                    pred_tensor = pred[i : i + 1]
                    sharp_tensor = sharp[i : i + 1]

                    # Calculate metrics for this image AND accumulate to global
                    current_metrics = {}
                    for name, metric in metrics.items():
                        # Calling metric(pred, target) updates global state AND returns current batch value
                        val = metric(pred_tensor, sharp_tensor)
                        current_metrics[name] = val.item()

                    # Write to CSV
                    csv_row = [f"{image_counter:06d}"] + [
                        current_metrics[k] for k in metrics.keys()
                    ]

                    csv_writer.writerow(csv_row)

                    # Create Title String
                    metric_str = " | ".join(
                        [f"{k}: {v:.4f}" for k, v in current_metrics.items()]
                    )

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    fig.suptitle(
                        f"Image {image_counter:06d}\n{metric_str}", fontsize=12
                    )

                    # Blur
                    axes[0].imshow(blur_np[i])
                    axes[0].set_title("Blur")
                    axes[0].axis("off")

                    # Pred Sharp
                    axes[1].imshow(pred_np[i])
                    axes[1].set_title("Pred Sharp")
                    axes[1].axis("off")

                    # GT
                    axes[2].imshow(sharp_np[i])
                    axes[2].set_title("GT")
                    axes[2].axis("off")

                    plt.tight_layout()
                    plt.savefig(save_dir / f"{image_counter:06d}.png")
                    plt.close(fig)

                    image_counter += 1

    # Close CSV
    csv_file.close()

    # Compute and Print Final Average Metrics
    results = {name: metric.compute().item() for name, metric in metrics.items()}

    logger.info("Test Results (Average):")
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info(f"Individual results saved to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
