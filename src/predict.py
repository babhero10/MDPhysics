"""
Standalone prediction script for image deblurring.

Example usage:
    # Single image
    python src/predict.py --input /path/to/blur.jpg --output /path/to/output/

    # Folder of images
    python src/predict.py --input /path/to/blur_folder/ --output /path/to/output/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_config(config_path: str = "configs/web.yaml"):
    """Load config using Hydra compose."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_path = Path(config_path).resolve()
    config_dir = str(config_path.parent)
    config_name = config_path.stem

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


def load_models(cfg, device: torch.device):
    """Load deblur model and optionally depth model from config."""
    from omegaconf import OmegaConf
    from src.utils.training import build_model

    # Build and load deblur model
    model = build_model(cfg, device)
    model.eval()

    # Load depth model if enabled in config (check dataset or model config)
    depth_model = None
    use_depth = OmegaConf.select(cfg, "dataset.use_depth", default=False)
    if use_depth:
        depth_repo = OmegaConf.select(
            cfg, "dataset.depth_model_repo", default="depth-anything/Depth-Anything-V2-Large-hf"
        )
        depth_model = load_depth_model(device, depth_repo)

    return model, depth_model


def load_depth_model(device: torch.device, model_repo: str = None):
    """Load DepthAnything-V2 model for depth estimation."""
    from transformers import AutoModelForDepthEstimation

    if model_repo is None:
        model_repo = "depth-anything/Depth-Anything-V2-Large-hf"

    print(f"Loading depth model: {model_repo}")
    depth_model = AutoModelForDepthEstimation.from_pretrained(model_repo)
    depth_model.eval()
    depth_model = depth_model.to(device)
    return depth_model


def compute_depth(image_tensor: torch.Tensor, depth_model, device: torch.device):
    """Compute depth map from image tensor.

    Args:
        image_tensor: (1, 3, H, W) tensor in [0, 1] range
        depth_model: DepthAnything model
        device: torch device

    Returns:
        depth_tensor: (1, 1, H, W) tensor
    """
    H, W = image_tensor.shape[2:]

    with torch.no_grad():
        depth_out = depth_model(pixel_values=image_tensor)
        depth_map = depth_out.predicted_depth

        depth_map = F.interpolate(
            depth_map.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

    return depth_map


def preprocess_image(image_path: Path, device: torch.device):
    """Load and preprocess single image.

    Returns:
        tensor: (1, 3, H, W) float tensor in [0, 1] range
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().div_(255.0)
    tensor = tensor.unsqueeze(0).to(device)
    return tensor


def postprocess_output(tensor: torch.Tensor) -> Image.Image:
    """Convert output tensor to PIL Image.

    Args:
        tensor: (1, 3, H, W) or (3, H, W) tensor in [0, 1] range

    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    pred_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(pred_np)


def predict_single(
    model,
    image_path: Path,
    output_path: Path,
    device: torch.device,
    depth_model=None,
):
    """Run prediction on single image."""
    device_type = "cuda" if device.type == "cuda" else "cpu"

    input_tensor = preprocess_image(image_path, device)

    depth = None
    if depth_model is not None:
        depth = compute_depth(input_tensor, depth_model, device)

    with torch.no_grad():
        with torch.amp.autocast(device_type):
            output = model(input_tensor, depth=depth)
            pred = torch.clamp(output["sharp_image"], 0.0, 1.0)

    output_image = postprocess_output(pred)
    output_image.save(output_path)


def get_image_files(input_path: Path) -> list:
    """Get list of image files from path (file or directory)."""
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [input_path]
        else:
            raise ValueError(f"Not a supported image file: {input_path}")
    elif input_path.is_dir():
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(set(files))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def main():
    """Main entry point with argparse."""
    parser = argparse.ArgumentParser(
        description="Deblur images using trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to single image or folder of images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for deblurred images",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input path not found: {args.input}")

    # Load config
    cfg = load_config()

    # Setup device from config
    device = torch.device(cfg.device)
    if "cuda" in cfg.device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load models from config
    print("Loading models...")
    model, depth_model = load_models(cfg, device)

    # Get image files
    image_files = get_image_files(args.input)
    if not image_files:
        print("No image files found in input path")
        return

    print(f"Found {len(image_files)} image(s) to process")

    # Process images
    successful = 0
    failed = 0

    for image_path in tqdm(image_files, desc="Processing"):
        output_path = args.output / image_path.name
        try:
            predict_single(model, image_path, output_path, device, depth_model)
            successful += 1
        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
            failed += 1

    print(f"\nProcessing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {args.output.resolve()}")


if __name__ == "__main__":
    main()
