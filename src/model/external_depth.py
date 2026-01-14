import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Dict

# Add local DepthAnything3 to path
current_dir = Path(__file__).resolve().parent
da3_path = current_dir / "Depth-Anything-3" / "src"
if str(da3_path) not in sys.path:
    sys.path.append(str(da3_path))

from depth_anything_3.api import DepthAnything3


class ExternalDepthModel(nn.Module):
    """
    Wrapper for official Depth Anything V3 API.
    Refactored to use depth_anything_3.api.DepthAnything3.
    """

    def __init__(
        self,
        checkpoint: str = "depth-anything/da3mono-large",
        freeze_backbone: bool = True,
        **kwargs,  # Ignore obsolete params like default_focal_length
    ):
        super().__init__()
        self.checkpoint = checkpoint

        # Use the official loading mechanism
        self.model = DepthAnything3.from_pretrained(checkpoint)

        if freeze_backbone:
            self.freeze_parameters()

    def freeze_parameters(self):
        """
        Freezes parameters.
        """
        for param in self.model.parameters():
            param.requires_grad = False

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

        # Call the model directly to maintain gradients for fine-tuning
        # (Using .inference() would likely break the computational graph)
        prediction = self.model(x_view)

        results = {}

        # Helper to crop back
        def unpad(tensor):
            # tensor is [B, ..., H_pad, W_pad]
            if pad_h > 0 or pad_w > 0:
                return tensor[..., :H, :W]
            return tensor

        # Access outputs from prediction object
        # prediction.depth: [B, N, H, W] -> [B, 1, H, W]
        if hasattr(prediction, "depth"):
            depth = prediction.depth
            if (
                depth.dim() == 3
            ):  # Handle case if it returns [N, H, W] for single sample
                depth = depth.unsqueeze(1)
            # If it's [B, N, H, W], we take N=1
            if depth.dim() == 4 and depth.shape[1] > 1:
                pass  # Keep views if present, but usually N=1 for us

            # Crop depth back to original size
            results["depth"] = unpad(depth)

        # Confidence maps
        if hasattr(prediction, "conf"):
            results["conf"] = unpad(prediction.conf)

        # Extrinsics [B, N, 3, 4] - No spatial dimensions, no unpad needed
        if hasattr(prediction, "extrinsics"):
            results["extrinsics"] = prediction.extrinsics

        # Intrinsics [B, N, 3, 3]
        if hasattr(prediction, "intrinsics"):
            intrinsics = prediction.intrinsics
            # If present, we take the intrinsics for the single view
            # Standard motion_blur expects (B, 3, 3)
            if intrinsics is not None and intrinsics.dim() == 4:
                results["intrinsics"] = intrinsics[:, 0, :, :]
            else:
                results["intrinsics"] = intrinsics

        return results
