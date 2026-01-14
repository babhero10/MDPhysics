import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin

# Add local DepthAnything3 to path
current_dir = Path(__file__).resolve().parent
da3_path = current_dir / "Depth-Anything-3" / "src"
if str(da3_path) not in sys.path:
    sys.path.append(str(da3_path))

# Import internals directly, skipping api.py to avoid broken torchvision import
from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from typing import Dict, Optional


class DepthAnything3Lite(nn.Module, PyTorchModelHubMixin):
    """
    Lightweight version of DepthAnything3 API that avoids torchvision dependency.
    """

    def __init__(self, model_name: str = "da3mono-large", **kwargs):
        super().__init__()
        self.model_name = model_name

        # Resolve model_name to registry key
        # Handle cases where model_name might be a repo ID like "depth-anything/da3mono-large"
        registry_name = model_name.split("/")[-1]

        if registry_name not in MODEL_REGISTRY:
            print(
                f"Warning: {registry_name} not found in registry. Available: {list(MODEL_REGISTRY.keys())}"
            )
            # Fallback to a safe default if possible, or assume it matches one
            # If exact match fails, try simple key matching
            for k in MODEL_REGISTRY.keys():
                if k in registry_name:
                    registry_name = k
                    break

        if registry_name in MODEL_REGISTRY:
            self.config = load_config(MODEL_REGISTRY[registry_name])
            self.model = create_object(self.config)
            self.model.eval()
        else:
            raise ValueError(f"Could not resolve model config for {model_name}")

    def forward(self, x, *args, **kwargs):
        # Forward directly to the underlying model
        out = self.model(x, *args, **kwargs)
        
        # Convert addict.Dict to regular dict to prevent torchinfo recursion
        # torchinfo tries to access .tensors, which addict creates dynamically, causing infinite loop
        if hasattr(out, "to_dict"):
            return out.to_dict()
        if isinstance(out, dict):
            return dict(out)
        return out


class ExternalDepthModel(nn.Module):
    """
    Wrapper for official Depth Anything V3 API (Lite version).
    """

    def __init__(
        self,
        checkpoint: str = "depth-anything/da3mono-large",
        freeze_backbone: bool = True,
        fine_tune_head: bool = False,
        **kwargs,  # Ignore obsolete params like default_focal_length
    ):
        super().__init__()
        self.checkpoint = checkpoint

        print(f"Loading official Depth Anything V3 (Lite): {checkpoint}...")
        # Use our Lite wrapper
        self.model = DepthAnything3Lite.from_pretrained(checkpoint)

        if freeze_backbone:
            self.freeze_parameters(fine_tune_head)

    def freeze_parameters(self, fine_tune_head: bool):
        """
        Freezes parameters. If fine_tune_head is True, unfreezes the head.
        """
        for param in self.model.parameters():
            param.requires_grad = False

        if fine_tune_head:
            # Unfreeze the Dual-DPT head
            head_found = False
            for name, module in self.model.named_modules():
                if "head" in name.lower():
                    print(f"Unfreezing DA3 head: {name}")
                    for param in module.parameters():
                        param.requires_grad = True
                    head_found = True

            if not head_found:
                print(
                    "Warning: Could not identify DA3 head. Model remains fully frozen."
                )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Depth Anything V3.
        Auto-pads input to multiple of 14 and crops output.

        Args:
            x: Input image (B, 3, H, W).
               Assumed [0, 1] range. DA3 usually handles internal normalization.

        Returns:
            Dict with 'depth', 'intrinsics', 'extrinsics'.
        """
        B, C, H, W = x.shape

        # 1. Pad to multiple of 14 (DA3 patch size)
        patch_size = 14
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size

        x_padded = x
        if pad_h > 0 or pad_w > 0:
            # Pad right and bottom
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

        # DA3 expects (B, N, 3, H, W) where N is number of views.
        # We treat each image in the batch as a single view (N=1).
        x_view = x_padded.unsqueeze(1)

        # Call the model directly
        # prediction is now a standard dict, safe for torchinfo
        prediction = self.model(x_view)

        results = {}

        # Helper to crop back
        def unpad(tensor):
            # tensor is [B, ..., H_pad, W_pad]
            if pad_h > 0 or pad_w > 0:
                return tensor[..., :H, :W]
            return tensor

        # Access outputs from prediction dict
        # prediction['depth']: [B, N, H, W] -> [B, 1, H, W]

        # Extract depth
        if "depth" in prediction:
            depth = prediction["depth"]
            if (
                depth.dim() == 3
            ):  # Handle case if it returns [N, H, W] for single sample
                depth = depth.unsqueeze(1)
            # If it's [B, N, H, W], we take N=1 logic if needed, but keeping shape is safer

            # Crop depth back to original size
            results["depth"] = unpad(depth)

        # Confidence maps
        if "depth_conf" in prediction:
            results["conf"] = unpad(prediction["depth_conf"])
        elif "conf" in prediction:
            results["conf"] = unpad(prediction["conf"])

        # Extrinsics [B, N, 3, 4] - No spatial dimensions, no unpad needed
        if "extrinsics" in prediction:
            results["extrinsics"] = prediction["extrinsics"]

        # Intrinsics [B, N, 3, 3]
        if "intrinsics" in prediction:
            intrinsics = prediction["intrinsics"]
            # If present, we take the intrinsics for the single view
            # Standard motion_blur expects (B, 3, 3)
            if intrinsics is not None and intrinsics.dim() == 4:
                results["intrinsics"] = intrinsics[:, 0, :, :]
            else:
                results["intrinsics"] = intrinsics

        return results