"""
Standalone prediction script for image deblurring.

Example usage:
    # Single image
    python src/predict.py \
        --input /path/to/blur.jpg \
        --output /path/to/output/ \
        --checkpoint outputs/mdt_edited/.../best_model_epoch493.pt

    # Folder of images
    python src/predict.py \
        --input /path/to/blur_folder/ \
        --output /path/to/output/ \
        --checkpoint outputs/mdt_edited/.../best_model_epoch493.pt

    # With depth estimation
    python src/predict.py \
        --input /path/to/blur_folder/ \
        --output /path/to/output/ \
        --checkpoint outputs/mdt_edited/.../best_model_epoch493.pt \
        --use-depth
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Load model from config and checkpoint."""
    from hydra.utils import instantiate

    cfg = OmegaConf.load(config_path)

    # Handle img_size interpolation - set a default if not resolved
    if OmegaConf.is_missing(cfg, "arch.cfg.img_size"):
        OmegaConf.update(cfg, "arch.cfg.img_size", 256)
    elif "${" in str(OmegaConf.select(cfg, "arch.cfg.img_size", default="")):
        # img_size references dataset.patch_size which isn't available standalone
        OmegaConf.update(cfg, "arch.cfg.img_size", 256)

    model = instantiate(cfg.arch).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=True, map_location=device)
    )
    model.eval()
    return model


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

        # Interpolate to original size
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

    # Load and preprocess
    input_tensor = preprocess_image(image_path, device)

    # Compute depth if model provided
    depth = None
    if depth_model is not None:
        depth = compute_depth(input_tensor, depth_model, device)

    # Run inference
    with torch.no_grad():
        with torch.amp.autocast(device_type):
            output = model(input_tensor, depth=depth)
            pred = torch.clamp(output["sharp_image"], 0.0, 1.0)

    # Save output
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
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=True,
        help="Path to model checkpoint .pt file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/mdt_edited.yaml"),
        help="Path to config file (default: configs/model/mdt_edited.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Enable depth estimation with DepthAnything-V2",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default=None,
        help="HuggingFace repo for depth model (default: depth-anything/Depth-Anything-V2-Large-hf)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        raise FileNotFoundError(f"Input path not found: {args.input}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(str(args.config), str(args.checkpoint), device)

    # Load depth model if requested
    depth_model = None
    if args.use_depth:
        depth_model = load_depth_model(device, args.depth_model)

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

    # Print summary
    print(f"\nProcessing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {args.output.resolve()}")


if __name__ == "__main__":
    main()
