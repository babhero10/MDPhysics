import torch
from PIL import Image
from pathlib import Path
import numpy as np
import lmdb
from tqdm import tqdm
from torch.utils.data import Dataset
import hydra


class LMDBDataset:
    def __init__(self, path):
        self.__env = None
        self.__path = str(path)

    def open(self, write: bool):
        if self.__env is not None:
            self.close()

        if write:
            self.__env = lmdb.open(
                self.__path,
                map_size=12 * 1024**3,
                subdir=True,
                readonly=False,
                lock=True,
                readahead=True,
                meminit=True,
            )
        else:
            self.__env = lmdb.open(
                self.__path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )

    def get_env(self):
        return self.__env

    def close(self):
        self.__env.close()
        self.__env = None


class BlurDataset(Dataset):
    def __init__(self, dataset_cfg: dict, split="train"):
        self.__raw_path = Path(dataset_cfg.src_root)
        self.__output_path = Path(dataset_cfg.dst_root) / split
        self.__output_path.mkdir(parents=True, exist_ok=True)

        self.__lmdb_env = LMDBDataset(self.__output_path)
        self.__patch_size = dataset_cfg.patch_size
        self.__split = split

        self.transform = None
        if split == "train" and "augmentations" in dataset_cfg:
            self.transform = hydra.utils.instantiate(dataset_cfg.augmentations)

        if not self.__check_downloaded():
            raise RuntimeError("Dataset not downloaded or corrupted")

        if not self.__check_processed():
            self.__process()

    def __load_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        img = np.asarray(img, dtype=np.uint8)
        return img

    def __process(self):
        self.__lmdb_env.open(write=True)

        gopro_root = self.__raw_path / self.__split

        blur_dir = gopro_root / "blur"
        sharp_dir = gopro_root / "sharp"

        blur_files = sorted(blur_dir.iterdir())
        sharp_files = sorted(sharp_dir.iterdir())

        assert len(blur_files) == len(sharp_files), "Blur/Sharp count mismatch"

        length = len(blur_files)

        txn = self.__lmdb_env.get_env().begin(write=True)
        for idx, (b_path, s_path) in tqdm(
            enumerate(zip(blur_files, sharp_files)),
            total=length,
            desc=f"Processing {self.__split} split to LMDB",
        ):
            blur = self.__load_image(b_path)
            sharp = self.__load_image(s_path)

            assert blur.shape == sharp.shape

            if idx == 0:
                ref_shape = sharp.shape
                txn.put(
                    b"__shape__",
                    np.array(sharp.shape, dtype=np.int64).tobytes()
                )
            elif sharp.shape != ref_shape:
                txn.put(
                    f"{idx}_shape".encode(),
                    np.array(sharp.shape, dtype=np.int64).tobytes(),
                )

            txn.put(f"{idx}_blur".encode(), blur.tobytes())
            txn.put(f"{idx}_sharp".encode(), sharp.tobytes())

            if idx % 50 == 0:
                txn.commit()
                txn = self.__lmdb_env.get_env().begin(write=True)

        txn.put(b"__len__", str(length).encode())
        txn.put(b"__complete__", b"1")

        txn.commit()
        self.__lmdb_env.close()

    def __ensure_env(self):
        if self.__lmdb_env.get_env() is None:
            self.__lmdb_env.open(write=False)

    def __getitem__(self, idx):
        self.__ensure_env()
        env = self.__lmdb_env.get_env()

        with env.begin() as txn:
            shape = np.frombuffer(
                txn.get(f"{idx}_shape".encode()) or txn.get(b"__shape__"),
                dtype=np.int64,
            )
            blur = np.frombuffer(
                txn.get(f"{idx}_blur".encode()), dtype=np.uint8
            ).reshape(shape)
            sharp = np.frombuffer(
                txn.get(f"{idx}_sharp".encode()), dtype=np.uint8
            ).reshape(shape)

        if self.__split == 'train' and self.transform:
            augmented = self.transform(image=blur, sharp=sharp)
            blur = augmented['image']
            sharp = augmented['sharp']

        # Convert to float
        blur = torch.tensor(np.transpose(blur, (2, 0, 1))).float().div_(255.0)
        sharp = torch.tensor(np.transpose(sharp, (2, 0, 1))).float().div_(255.0)

        return blur, sharp

    def __len__(self):
        self.__ensure_env()
        with self.__lmdb_env.get_env().begin() as txn:
            return int(txn.get(b"__len__").decode())

    def __del__(self):
        if self.__lmdb_env.get_env() is not None:
            self.__lmdb_env.close()

    def __check_downloaded(self) -> bool:
        gopro_root = self.__raw_path
        if not gopro_root.is_dir():
            return False

        for split in ["train", "val", "test"]:
            blur_dir = gopro_root / split / "blur"
            sharp_dir = gopro_root / split / "sharp"

            if not blur_dir.is_dir():
                return False
            if not sharp_dir.is_dir():
                return False

            # At least one image must exist
            if not any(blur_dir.iterdir()):
                return False
            if not any(sharp_dir.iterdir()):
                return False

        return True

    def __check_processed(self) -> bool:
        if not self.__output_path.is_dir():
            return False
        try:
            self.__lmdb_env.open(write=False)
        except lmdb.Error:
            return False

        try:
            with self.__lmdb_env.get_env().begin() as txn:
                completed = txn.get(b"__complete__")
                if completed is None or completed.decode() != "1":
                    return False
        except Exception:
            return False
        finally:
            self.__lmdb_env.close()

        return True
