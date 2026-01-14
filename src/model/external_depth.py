import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict


class ExternalDepthModel(nn.Module):
    """
    Wrapper for Depth Anything V3 (and V2) models to provide depth priors.
    Handles different model variants (Large, Mono, Metric) and their specific outputs.
    """

    def __init__(
        self,
        checkpoint: str = "depth-anything/Depth-Anything-V2-Small-hf",
        freeze_backbone: bool = True,
        fine_tune_head: bool = False,
        default_focal_length: float = 700.0,  # For metric scaling if intrinsics unknown
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.default_focal_length = default_focal_length

        print(f"Loading External Depth Model: {checkpoint}...")
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

        # Determine model type/capabilities based on checkpoint name
        self.is_v3 = "da3" in checkpoint.lower() or "v3" in checkpoint.lower()
        self.is_metric = "metric" in checkpoint.lower()
        self.is_mono = "mono" in checkpoint.lower()

        # Freeze parameters
        if freeze_backbone:
            self.freeze_backbone(fine_tune_head)

    def freeze_backbone(self, fine_tune_head: bool):
        """
        Freezes backbone, optionally unfreezes the head.
        """
        for param in self.model.parameters():
            param.requires_grad = False

        if fine_tune_head:
            head_found = False
            # Common head names in DPT/DepthAnything architectures
            head_keywords = ["head", "decode_head", "depth_head"]

            for name, module in self.model.named_modules():
                # We want to unfreeze the highest level module that looks like a head
                # but avoid unfreezing tiny sub-modules individually if parent is frozen.
                # Simplest strategy: check name against keywords.
                if any(k in name for k in head_keywords):
                    # Check if it's a top-level head or close to it
                    print(f"Unfreezing head module: {name}")
                    for param in module.parameters():
                        param.requires_grad = True
                    head_found = True

            if not head_found:
                print(
                    "Warning: Could not identify depth head automatically. Keeping full model frozen."
                )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning differentiable dictionary.

        Args:
            x: Input image (B, 3, H, W). Normalized expected?
               Usually DA models expect ImageNet mean/std normalized inputs.
               We assume x is [0, 1] RGB. Caller might need to normalize.

        Returns:
            Dict with:
            - 'depth': (B, 1, H, W) - Metric or Relative Depth (Distance, not Disparity)
            - 'intrinsics': (B, 3, 3) Optional
            - 'extrinsics': (B, 3, 4) Optional
        """
        # Run model
        # Note: DPT models often return a specific output class.
        # We access attributes directly if possible, or fallback to dict/tuple.
        outputs = self.model(x)

        results = {}

        # 1. Extract Raw Depth
        raw_depth = None
        if hasattr(outputs, "predicted_depth"):
            raw_depth = outputs.predicted_depth
        elif hasattr(outputs, "depth"):
            raw_depth = outputs.depth
        elif isinstance(outputs, dict) and "predicted_depth" in outputs:
            raw_depth = outputs["predicted_depth"]
        elif isinstance(outputs, torch.Tensor):
            raw_depth = outputs
        elif isinstance(outputs, (list, tuple)):
            raw_depth = outputs[0]

        if raw_depth is None:
            # Fallback for some HuggingFace DPT implementations
            if hasattr(outputs, "last_hidden_state"):
                # This would require a manual head, which we assume is part of 'model'
                # If we are here, 'model' might be just a backbone?
                # AutoModel usually loads the full Architecture if mapped correctly.
                pass
            raise ValueError(
                f"Could not extract depth from model output type: {type(outputs)}"
            )

        # Ensure (B, 1, H, W)
        if raw_depth.dim() == 3:
            raw_depth = raw_depth.unsqueeze(1)

        # 2. Extract Intrinsics/Extrinsics (DA3 Large specific)
        if hasattr(outputs, "predicted_intrinsics"):
            results["intrinsics"] = outputs.predicted_intrinsics
        elif hasattr(outputs, "intrinsics"):
            results["intrinsics"] = outputs.intrinsics

        if hasattr(outputs, "predicted_extrinsics"):
            results["extrinsics"] = outputs.predicted_extrinsics
        elif hasattr(outputs, "extrinsics"):
            results["extrinsics"] = outputs.extrinsics

        # 3. Apply Scaling / Normalization Logic
        if self.is_metric:
            # DA3 Metric: Output is raw units. Needs scaling.
            # Formula: depth_meters = focal * raw / 300.0

            # Get focal length
            if "intrinsics" in results:
                # Intrinsics K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                # Shape (B, 3, 3)
                fx = results["intrinsics"][:, 0, 0].view(-1, 1, 1, 1)
                fy = results["intrinsics"][:, 1, 1].view(-1, 1, 1, 1)
                focal = (fx + fy) / 2.0
            else:
                # Use default/estimated focal length
                focal = self.default_focal_length

            # Apply scaling
            # Note: raw output of DA3 Metric might be negative? No, depth is usually positive.
            # We apply ReLU just in case to avoid physics errors.
            results["depth"] = (F.relu(raw_depth) * focal) / 300.0

        else:
            # Relative Depth (DA3 Large / Mono / V2)
            # Docs: "Range 0 to 1 (relative depth, nearer = lower values)"
            # Wait, user docs said "nearer = lower values".
            # Standard Depth: Near = Small Z, Far = Large Z.
            # So raw_depth is proportional to Distance Z.

            # However, typically relative depth models output something that needs scaling.
            # And they often use Sigmoid at the end, so range is [0, 1].
            # 0.0 -> Near (0 meters), 1.0 -> Far (Arbitrary max).
            # We might want to scale this to a reasonable physical range (e.g. 0-100m)
            # so the physics engine gradients work well.
            # If we just leave it 0-1, velocities will need to be very small.
            # We assume the "fine-tuned head" or downstream adapter will handle this,
            # OR we apply a heuristic scale factor.
            # Let's apply a heuristic scale factor of 10.0 to map 0..1 to 0..10m range.
            # This makes "1.0" (Far) = 10 meters.
            results["depth"] = raw_depth * 10.0

        return results


import torch.nn.functional as F
