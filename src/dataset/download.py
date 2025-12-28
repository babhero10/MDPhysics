import gdown
import shutil
from pathlib import Path
import hydra
from omegaconf import DictConfig


def download_gopro(path):
    gdown(
        "https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view",
        path,
        quiet=False,
        fuzzy=True
    )


def flatten_dataset(cfg):
    src_root = Path(cfg.dataset.src_root)
    dst_root = Path(cfg.dataset.dst_root)

    for split in cfg.dataset.splits:
        src_split = src_root / split
        dst_split = dst_root / split

        for mode in cfg.dataset.modes:
            (dst_split / mode).mkdir(parents=True, exist_ok=True)

        for seq_dir in src_split.iterdir():
            if not seq_dir.is_dir():
                continue

            seq_name = seq_dir.name

            for mode in cfg.dataset.modes:
                src_mode_dir = seq_dir / mode
                if not src_mode_dir.exists():
                    continue

                for img_path in src_mode_dir.iterdir():
                    if img_path.suffix.lower() not in cfg.dataset.image_exts:
                        continue

                    new_name = f"{seq_name}_{img_path.name}"
                    dst_path = dst_split / mode / new_name

                    if cfg.dataset.copy:
                        shutil.copy2(img_path, dst_path)
                    else:
                        shutil.move(img_path, dst_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.dataset.download:
        download_gopro(cfg.dataset.src_root)

    flatten_dataset(cfg.dataset)
    print("GoPro dataset downloaded successfully")


if __name__ == "__main__":
    main()
