import gdown
import shutil
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import random

logger = logging.getLogger(__name__)


def download_gopro(path):
    try:
        # Download the zip file
        result = gdown.download(
            "https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view",
            path,
            quiet=False,
            fuzzy=True,
        )
        if result is None:
            logger.error("GoPro dataset downloading failed!")
            return False
        logger.info("GoPro dataset downloaded successfully!")

        # Unzip the file
        zip_path = Path(path)
        extract_dir = zip_path.parent

        logger.info(f"Extracting {zip_path} to {extract_dir}...")
        shutil.unpack_archive(zip_path, extract_dir)
        logger.info("Extraction completed successfully!")

        # Remove the zip file
        logger.info(f"Removing zip file {zip_path}...")
        zip_path.unlink()
        logger.info("Zip file removed!")

        return True
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return False


def flatten_dataset(cfg):
    src_root = Path(cfg.src_root)
    dst_root = Path(cfg.dst_root)

    if not src_root.exists():
        logger.error(f"The following path {src_root} doesn't exist")
        return False

    dst_root.mkdir(parents=True, exist_ok=True)

    for split in cfg.splits:
        src_split = src_root / split
        dst_split = dst_root / split

        for mode in cfg.modes:
            (dst_split / mode).mkdir(parents=True, exist_ok=True)

        for seq_dir in src_split.iterdir():
            if not seq_dir.is_dir():
                continue

            seq_name = seq_dir.name

            for mode in cfg.modes:
                src_mode_dir = seq_dir / mode
                if not src_mode_dir.exists():
                    continue

                for img_path in src_mode_dir.iterdir():
                    if img_path.suffix.lower() != ".png":
                        continue

                    new_name = f"{seq_name}_{img_path.name}"
                    dst_path = dst_split / mode / new_name

                    if cfg.copy:
                        shutil.copy2(img_path, dst_path)
                    else:
                        shutil.move(img_path, dst_path)


def split_train_val(cfg):
    """
    Split train folder into train and validation sets

    Args:
        cfg: Configuration with dst_root, modes, val_split, and optional seed
    """
    dst_root = Path(cfg.dst_root)
    train_dir = dst_root / "train"
    val_dir = dst_root / "val"

    if not train_dir.exists():
        logger.error(f"Train directory {train_dir} doesn't exist")
        return False

    val_split = cfg.val_split

    if val_split <= 0.0 or val_split >= 1.0:
        logger.info(f"Invalid or zero val_split={val_split}, skipping validation split")
        return True

    # Set random seed for reproducibility
    seed = cfg.seed
    random.seed(seed)

    logger.info(f"Splitting train set: {val_split*100:.1f}% for validation")

    # Create validation directories
    for mode in cfg.modes:
        (val_dir / mode).mkdir(parents=True, exist_ok=True)

    # Use the first mode to determine the list of files
    first_mode = cfg.modes[0]
    ref_train_dir = train_dir / first_mode

    if not ref_train_dir.exists():
        logger.error(f"Reference mode directory {ref_train_dir} doesn't exist")
        return False

    # Get all images and sort them to ensure deterministic shuffling
    images = sorted(
        [img.name for img in ref_train_dir.iterdir() if img.suffix.lower() == ".png"]
    )

    if len(images) == 0:
        logger.warning(f"No images found in {ref_train_dir}")
        return False

    # Shuffle and split
    random.shuffle(images)
    num_val = int(len(images) * val_split)
    val_filenames = images[:num_val]

    logger.info(f"Moving {num_val} images from train to val for modes: {cfg.modes}")

    # Move validation images for ALL modes
    for filename in val_filenames:
        for mode in cfg.modes:
            src_path = train_dir / mode / filename
            dst_path = val_dir / mode / filename

            if src_path.exists():
                shutil.move(str(src_path), str(dst_path))
            else:
                logger.warning(f"File not found: {src_path}")

    logger.info("Train/Val split completed successfully!")
    return True


@hydra.main(
    version_base=None, config_path="../../configs/dataset", config_name="gopro_download"
)
def main(cfg: DictConfig):
    if cfg.download:
        if not download_gopro(cfg.src_root):
            exit(-1)

    flatten_dataset(cfg)

    if cfg.get("val_split", 0.0) > 0.0:
        split_train_val(cfg)

    print("GoPro dataset re-organized successfully!")


if __name__ == "__main__":
    main()
