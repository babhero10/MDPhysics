import gdown
import shutil
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import random

logger = logging.getLogger(__name__)


def download_dataset(url: str, path: str, dataset_name: str = "Dataset"):
    """
    Generic download function for datasets from Google Drive.

    Args:
        url: Google Drive URL or file ID
        path: Destination path for the zip file
        dataset_name: Name of the dataset for logging
    """
    try:
        # Download the zip file
        result = gdown.download(
            url,
            path,
            quiet=False,
            fuzzy=True,
        )
        if result is None:
            logger.error(f"{dataset_name} downloading failed!")
            return False
        logger.info(f"{dataset_name} downloaded successfully!")

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


# Default download URLs
DOWNLOAD_URLS = {
    "gopro": "https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view",
    "realblur": "https://drive.google.com/uc?id=17v5Dj0M2WExgxJgsGpEWc-6UyhK1IWWK",
}


def download_gopro(path):
    """Download GoPro dataset (legacy function for backward compatibility)."""
    return download_dataset(DOWNLOAD_URLS["gopro"], path, "GoPro")


def flatten_dataset(cfg):
    """Flatten nested dataset structure (e.g., GoPro format)."""
    src_root = Path(cfg.src_root)
    dst_root = Path(cfg.dst_root)

    if not src_root.exists():
        logger.error(f"The following path {src_root} doesn't exist")
        return False

    dst_root.mkdir(parents=True, exist_ok=True)

    # Mode mapping (e.g., gt -> sharp)
    mode_mapping = cfg.get("mode_mapping", {})

    for split in cfg.splits:
        src_split = src_root / split
        dst_split = dst_root / split

        # Create destination directories with mapped names
        for mode in cfg.modes:
            dst_mode = mode_mapping.get(mode, mode)
            (dst_split / dst_mode).mkdir(parents=True, exist_ok=True)

        for seq_dir in src_split.iterdir():
            if not seq_dir.is_dir():
                continue

            seq_name = seq_dir.name

            for mode in cfg.modes:
                src_mode_dir = seq_dir / mode
                if not src_mode_dir.exists():
                    continue

                dst_mode = mode_mapping.get(mode, mode)

                for img_path in src_mode_dir.iterdir():
                    if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                        continue

                    new_name = f"{seq_name}_{img_path.name}"
                    dst_path = dst_split / dst_mode / new_name

                    if cfg.copy:
                        shutil.copy2(img_path, dst_path)
                    else:
                        shutil.move(img_path, dst_path)


def flatten_realblur(cfg):
    """
    Flatten RealBlur dataset using text file lists.

    Expected text file format (space-separated):
        gt_path blur_path
    """
    src_root = Path(cfg.src_root)
    dst_root = Path(cfg.dst_root)

    if not src_root.exists():
        logger.error(f"The following path {src_root} doesn't exist")
        return False

    dst_root.mkdir(parents=True, exist_ok=True)

    # Mode mapping (e.g., gt -> sharp)
    mode_mapping = cfg.get("mode_mapping", {"gt": "sharp", "blur": "blur"})

    # Process each split using its list file
    list_files = cfg.get("list_files", {})

    for split, list_file in list_files.items():
        list_path = src_root / list_file
        if not list_path.exists():
            logger.warning(f"List file not found: {list_path}")
            continue

        dst_split = dst_root / split

        # Create destination directories
        for src_mode, dst_mode in mode_mapping.items():
            (dst_split / dst_mode).mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {split} split from {list_file}...")

        with open(list_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                logger.warning(f"Invalid line format: {line}")
                continue

            gt_rel_path, blur_rel_path = parts

            # Full paths
            gt_path = src_root / gt_rel_path
            blur_path = src_root / blur_rel_path

            if not gt_path.exists():
                logger.warning(f"GT file not found: {gt_path}")
                continue
            if not blur_path.exists():
                logger.warning(f"Blur file not found: {blur_path}")
                continue

            # Extract scene name and image name for unique naming
            # e.g., RealBlur-R_.../scene183/gt/gt_7.png -> scene183_gt_7.png
            scene_name = gt_path.parent.parent.name
            img_name = gt_path.name

            # Create unique filename
            new_gt_name = f"{scene_name}_{img_name}"
            new_blur_name = f"{scene_name}_{blur_path.name}"

            # Destination paths
            dst_gt = dst_split / mode_mapping.get("gt", "sharp") / new_gt_name
            dst_blur = dst_split / mode_mapping.get("blur", "blur") / new_blur_name

            if cfg.copy:
                shutil.copy2(gt_path, dst_gt)
                shutil.copy2(blur_path, dst_blur)
            else:
                shutil.move(str(gt_path), str(dst_gt))
                shutil.move(str(blur_path), str(dst_blur))

        logger.info(f"Processed {len(lines)} pairs for {split} split")


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

    # Get the actual mode names in destination (after mapping for RealBlur)
    mode_mapping = cfg.get("mode_mapping", {})
    if cfg.get("dataset_type") == "realblur":
        # For RealBlur, use the mapped names (sharp, blur)
        modes = list(set(mode_mapping.values()))
    else:
        # For GoPro, use the original modes
        modes = list(cfg.modes)

    # Create validation directories
    for mode in modes:
        (val_dir / mode).mkdir(parents=True, exist_ok=True)

    # Use the first mode to determine the list of files
    first_mode = modes[0]
    ref_train_dir = train_dir / first_mode

    if not ref_train_dir.exists():
        logger.error(f"Reference mode directory {ref_train_dir} doesn't exist")
        return False

    # Get all images and sort them to ensure deterministic shuffling
    images = sorted(
        [img.name for img in ref_train_dir.iterdir() if img.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    if len(images) == 0:
        logger.warning(f"No images found in {ref_train_dir}")
        return False

    # Shuffle and split
    random.shuffle(images)
    num_val = int(len(images) * val_split)
    val_filenames = images[:num_val]

    logger.info(f"Moving {num_val} images from train to val for modes: {modes}")

    # For RealBlur, filenames differ between modes (gt_X.png vs blur_X.png)
    is_realblur = cfg.get("dataset_type") == "realblur"

    # Move validation images for ALL modes
    for filename in val_filenames:
        for mode in modes:
            # For RealBlur, convert filename between gt/blur naming
            if is_realblur:
                if mode == "blur":
                    mode_filename = filename.replace("_gt_", "_blur_")
                else:  # sharp
                    mode_filename = filename
            else:
                mode_filename = filename

            src_path = train_dir / mode / mode_filename
            dst_path = val_dir / mode / mode_filename

            if src_path.exists():
                shutil.move(str(src_path), str(dst_path))
            else:
                logger.warning(f"File not found: {src_path}")

    logger.info("Train/Val split completed successfully!")
    return True


@hydra.main(version_base=None, config_path="../../configs/download", config_name="gopro")
def main(cfg: DictConfig):
    dataset_type = cfg.get("dataset_type", "gopro")

    if cfg.get("download", False):
        # Use custom download_url from config, or fall back to default URLs
        url = cfg.get("download_url", DOWNLOAD_URLS.get(dataset_type))
        download_path = cfg.get("download_path", f"{cfg.src_root}.zip")

        if url is None:
            logger.error(f"No download URL found for dataset type: {dataset_type}")
            exit(-1)

        if not download_dataset(url, download_path, dataset_type.upper()):
            exit(-1)

    # Choose flatten method based on dataset type
    if dataset_type == "realblur":
        flatten_realblur(cfg)
    else:
        flatten_dataset(cfg)

    if cfg.get("val_split", 0.0) > 0.0:
        split_train_val(cfg)

    print(f"{dataset_type.upper()} dataset re-organized successfully!")


if __name__ == "__main__":
    main()
