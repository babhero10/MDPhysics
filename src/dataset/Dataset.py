import os
import torch
from PIL import Image
from pathlib import Path
import numpy as np
import lmdb
from tqdm import tqdm
from torch.utils.data import Dataset


class LMDBDataset():
    def __init__(self, path):
        self.__env = None
        self.__path = path

    def open(self, write: bool):
        if self.__env is not None:
            self.close()

        if write:
            self.__env = lmdb.open(
                self.__path,
                map_size=8 * 1024**3,
                subdir=True,
                readonly=False,
                lock=True,
                readahead=True,
                meminit=True
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
        self.__env.sync()
        self.__env.close()
        self.__env = None


class BlurDataset(Dataset):
    def __init__(self, dataset_cfg: dict, split="train"):
        self.__raw_path = dataset_cfg.path
        self.__output_path = dataset_cfg.processed_path
        self.__lmdb_env = LMDBDataset(self.__output_path)
        self.__split = split

        if not self.__check_processed():
            self.__process()

        self.__lmdb_env.open(write=False)

    def __load_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def __process(self):
        self.__lmdb_env.open(write=True)

        gopro_root = (
            Path(self.__raw_path)
            / self.__split
        )

        blur_dir = gopro_root / "blur"
        sharp_dir = gopro_root / "sharp"

        blur_files = sorted(blur_dir.iterdir())
        sharp_files = sorted(sharp_dir.iterdir())

        assert len(blur_files) == len(sharp_files), "Blur/Sharp count mismatch"

        length = len(blur_files)

        with self.__lmdb_env.get_env().begin(write=True) as txn:
            for idx, (b_path, s_path) in tqdm(
                enumerate(zip(blur_files, sharp_files)),
                total=length,
                desc="Processing to LMDB"
            ):
                blur = self.__load_image(b_path)
                sharp = self._load_image(s_path)

                if idx == 0:
                    txn.put(
                        b"__shape__",
                        np.array(blur.shape, dtype=np.int64).tobytes()
                    )

                txn.put(f"{idx}_blur".encode(), blur.tobytes())
                txn.put(f"{idx}_sharp".encode(), sharp.tobytes())

            txn.put(b"__len__", str(length).encode())
            txn.put(b"__complete__", b"1")

        self.__lmdb_env.close()

    def __getitem__(self, idx):
        if self.__lmdb_env.get_env() is None:
            self.__lmdb_env.open(write=False)

        env = self.__lmdb_env.get_env()

        with env.begin() as txn:
            shape = np.frombuffer(txn.get(b"__shape__"), dtype=np.int64)

            blur = np.frombuffer(txn.get(f"{idx}_blur".encode()), dtype=np.float32).reshape(shape)
            sharp = np.frombuffer(txn.get(f"{idx}_sharp".encode()), dtype=np.float32).reshape(shape)

        return torch.as_tensor(blur), torch.as_tensor(sharp)

    def __len__(self):
        with self.__lmdb_env.get_env().begin() as txn:
            return int(txn.get(b"__len__").decode())

    def __check_downloaded(self) -> bool:
        if not os.path.isdir(self.__raw_path):
            return False

        gopro_root = os.path.join(self.__raw_path, "GOPRO_Large")
        if not os.path.isdir(gopro_root):
            return False

        for split in ["train", "test"]:
            blur_dir = os.path.join(gopro_root, split, "blur")
            sharp_dir = os.path.join(gopro_root, split, "sharp")

            if not os.path.isdir(blur_dir):
                return False
            if not os.path.isdir(sharp_dir):
                return False

            # At least one image must exist
            if len(os.listdir(blur_dir)) == 0:
                return False
            if len(os.listdir(sharp_dir)) == 0:
                return False

        return True

    def __check_processed(self) -> bool:
        if not os.path.isdir(self.__output_path):
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
