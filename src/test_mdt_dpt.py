import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model.MDT_DPT import MDT_DPT
from omegaconf import OmegaConf


def test_mdt_dpt():
    cfg = OmegaConf.create(
        {
            "dim": 48,
            "num_blocks": [2, 2, 2, 2],
            "num_refinement_blocks": 1,
            "ffn_expansion_factor": 3,
            "patch_size": 128,  # Configured patch size
            "backbone_name": "facebook/dinov2-base",
            "depth_backbone_name": "depth-anything/Depth-Anything-V2-Small-hf",
            "out_indices": [8, 9, 10, 11],
            "freeze_backbones": True,
        }
    )

    print("Instantiating MDT_DPT...")
    model = MDT_DPT(cfg)
    model.eval()

    # Test 1: Training resolution (128x128)
    print("\n--- Test 1: 128x128 Crop ---")
    x1 = torch.randn(1, 3, 128, 128)
    try:
        with torch.no_grad():
            out1 = model(x1)
        print("Success. Output:", out1["sharp_image"].shape)
    except Exception as e:
        print("FAILED:", e)
        import traceback

        traceback.print_exc()

    # Test 2: Full Resolution (1280x720) - Mismatch with patch_size
    print("\n--- Test 2: 1280x720 Full Res ---")
    x2 = torch.randn(1, 3, 720, 1280)  # PyTorch is (C, H, W)
    try:
        with torch.no_grad():
            out2 = model(x2)
        print("Success. Output:", out2["sharp_image"].shape)
        if out2["sharp_image"].shape[-2:] == (720, 1280):
            print("RESOLUTION PRESERVED CORRECTLY")
        else:
            print("RESOLUTION MISMATCH")
    except Exception as e:
        print("FAILED:", e)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_mdt_dpt()
