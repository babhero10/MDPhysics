import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add local DepthAnything3 to path
current_dir = Path(__file__).resolve().parent
da3_path = current_dir / "Depth-Anything-3" / "src"
if str(da3_path) not in sys.path:
    sys.path.append(str(da3_path))

from depth_anything_3.api import DepthAnything3
from typing import Dict, Optional


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

        Args:
            x: Input image (B, 3, H, W).
               Assumed [0, 1] range. DA3 usually handles internal normalization.

        Returns:
            Dict with 'depth', 'intrinsics', 'extrinsics'.
        """
        B, C, H, W = x.shape

        # DA3 expects (B, N, 3, H, W) where N is number of views.
        # We treat each image in the batch as a single view (N=1).
        # We can reshape x to (B, 1, 3, H, W) to process images independently but view-aware.
        x_view = x.unsqueeze(1)

        # Call the model directly to maintain gradients for fine-tuning
        # (Using .inference() would likely break the computational graph)
        prediction = self.model(x_view)

        results = {}

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

            results["depth"] = depth

        # Confidence maps
        if hasattr(prediction, "conf"):
            results["conf"] = prediction.conf

        # Extrinsics [B, N, 3, 4]
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
