import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import numpy as np
import lmdb
from tqdm import tqdm
from torch.utils.data import Dataset
import hydra
from transformers import AutoModelForDepthEstimation


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
                map_size=24 * 1024**3,
                subdir=True,
                readonly=False,
                lock=True,
                readahead=True,
                meminit=True,
            )
        else:
            self.__env = lmdb.open(
                self.__path, readonly=True, lock=False, readahead=False, meminit=False
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
        self.__split = split

        # Depth settings
        self.__use_depth = dataset_cfg.get("use_depth", False)
        self.__depth_model_repo = dataset_cfg.get(
            "depth_model_repo", "depth-anything/Depth-Anything-V2-Small-hf"
        )

        # RAW image settings (for RealBlur-R)
        self.__is_raw = dataset_cfg.get("is_raw", False)

        # Configurable splits to check (default: train, val, test)
        self.__expected_splits = dataset_cfg.get("expected_splits", ["train", "val", "test"])

        # Separate spatial and color transforms
        self.spatial_transform = None
        self.color_transform = None
        if split == "train":
            if "spatial_augmentations" in dataset_cfg:
                self.spatial_transform = hydra.utils.instantiate(
                    dataset_cfg.spatial_augmentations
                )
            if "color_augmentations" in dataset_cfg:
                self.color_transform = hydra.utils.instantiate(
                    dataset_cfg.color_augmentations
                )

        if not self.__check_downloaded():
            raise RuntimeError("Dataset not downloaded or corrupted")

        if not self.__check_processed():
            self.__process()

        self.__ensure_env()
        with self.__lmdb_env.get_env().begin() as txn:
            self.__len = int(txn.get(b"__len__").decode())

        self.__lmdb_env.close()
        self.__lmdb_env = None

    def __load_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        img = np.asarray(img, dtype=np.uint8)

        # Apply RAW preprocessing if enabled (for RealBlur-R)
        if self.__is_raw:
            img = self.__preprocess_raw(img)

        return img

    def __preprocess_raw(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess RAW/linear images (e.g., RealBlur-R).
        Applies simple gamma correction to convert from linear to sRGB space.
        """
        # Convert to float [0, 1]
        img_float = img.astype(np.float32) / 255.0

        # Apply gamma correction (linear to sRGB)
        # sRGB uses gamma ~2.2, but the standard formula is:
        # if x <= 0.0031308: 12.92 * x
        # else: 1.055 * x^(1/2.4) - 0.055
        gamma = 2.2
        img_float = np.power(np.clip(img_float, 0, 1), 1.0 / gamma)

        # Convert back to uint8
        img = (img_float * 255.0).astype(np.uint8)
        return img

    def __process(self):
        self.__lmdb_env.open(write=True)

        gopro_root = self.__raw_path / self.__split

        blur_dir = gopro_root / "blur"
        sharp_dir = gopro_root / "sharp"

        blur_names = {p.name for p in blur_dir.iterdir()}
        sharp_names = {p.name for p in sharp_dir.iterdir()}

        common_names = sorted(list(blur_names.intersection(sharp_names)))

        if len(common_names) != len(blur_names) or len(common_names) != len(
            sharp_names
        ):
            print(
                f"Warning: Dataset mismatch in {self.__split} split. "
                f"Blur: {len(blur_names)}, Sharp: {len(sharp_names)}, Common: {len(common_names)}"
            )

        blur_files = [blur_dir / name for name in common_names]
        sharp_files = [sharp_dir / name for name in common_names]

        length = len(blur_files)

        # Load depth model if use_depth is enabled
        depth_model = None
        if self.__use_depth:
            print(f"Loading depth model: {self.__depth_model_repo}")
            depth_model = AutoModelForDepthEstimation.from_pretrained(
                self.__depth_model_repo
            )
            depth_model.eval()
            if torch.cuda.is_available():
                depth_model = depth_model.cuda()

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
                txn.put(b"__shape__", np.array(sharp.shape, dtype=np.int64).tobytes())
            elif sharp.shape != ref_shape:
                txn.put(
                    f"{idx}_shape".encode(),
                    np.array(sharp.shape, dtype=np.int64).tobytes(),
                )

            txn.put(f"{idx}_blur".encode(), blur.tobytes())
            txn.put(f"{idx}_sharp".encode(), sharp.tobytes())

            # Compute and store depth map if use_depth is enabled
            if self.__use_depth and depth_model is not None:
                depth = self.__compute_depth(blur, depth_model)
                txn.put(f"{idx}_depth".encode(), depth.tobytes())

            if idx % 50 == 0:
                txn.commit()
                txn = self.__lmdb_env.get_env().begin(write=True)

        txn.put(b"__len__", str(length).encode())
        txn.put(b"__use_depth__", b"1" if self.__use_depth else b"0")
        txn.put(b"__complete__", b"1")

        txn.commit()
        self.__lmdb_env.close()

    def __compute_depth(self, blur_img: np.ndarray, depth_model) -> np.ndarray:
        """Compute depth map from blur image using DepthAnythingV2."""
        H, W = blur_img.shape[:2]

        # Convert to tensor and normalize to [0, 1]
        img_tensor = torch.tensor(blur_img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            depth_out = depth_model(pixel_values=img_tensor)
            depth_map = depth_out.predicted_depth

            # Interpolate to original size
            depth_map = F.interpolate(
                depth_map.unsqueeze(1),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            depth_map = depth_map.squeeze().cpu().numpy()

        # Store as float32
        return depth_map.astype(np.float32)

    def __ensure_env(self):
        if self.__lmdb_env.get_env() is None:
            self.__lmdb_env.open(write=False)

    def __getitem__(self, idx):
        # Fork safe.
        if self.__lmdb_env is None:
            self.__lmdb_env = LMDBDataset(self.__output_path)

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

            # Load depth if available
            depth = None
            if self.__use_depth:
                depth_bytes = txn.get(f"{idx}_depth".encode())
                if depth_bytes is not None:
                    depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(
                        shape[:2]
                    )
                    # Convert depth to 3-channel for albumentations compatibility
                    depth = np.stack([depth, depth, depth], axis=-1)

        if self.__split == "train":
            # Apply spatial transforms to all (blur, sharp, depth)
            if self.spatial_transform:
                if depth is not None:
                    augmented = self.spatial_transform(
                        image=blur, sharp=sharp, depth=depth
                    )
                    depth = augmented["depth"]
                else:
                    augmented = self.spatial_transform(image=blur, sharp=sharp)
                blur = augmented["image"]
                sharp = augmented["sharp"]

            # Apply color transforms only to blur and sharp (not depth)
            if self.color_transform:
                augmented = self.color_transform(image=blur, sharp=sharp)
                blur = augmented["image"]
                sharp = augmented["sharp"]

        # Convert to float tensors
        blur = torch.tensor(np.transpose(blur, (2, 0, 1))).float().div_(255.0)
        sharp = torch.tensor(np.transpose(sharp, (2, 0, 1))).float().div_(255.0)

        result = {"blur": blur, "sharp": sharp}

        # Convert depth to single-channel tensor
        if depth is not None:
            depth = torch.tensor(depth[:, :, 0]).float().unsqueeze(0)
            result["depth"] = depth

        return result

    def __len__(self):
        return self.__len

    def __del__(self):
        if self.__lmdb_env is not None and self.__lmdb_env.get_env() is not None:
            self.__lmdb_env.close()

    def __check_downloaded(self) -> bool:
        root = self.__raw_path
        if not root.is_dir():
            return False

        for split in self.__expected_splits:
            blur_dir = root / split / "blur"
            sharp_dir = root / split / "sharp"

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
