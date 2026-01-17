import torch
from model.MDT_Edited import MDT_Edited
from omegaconf import DictConfig
from torchinfo import summary


def debug_with_torchinfo(input_res=(256, 256)):
    print(f"\n--- Torchinfo Summary for: {input_res} ---")

    # 1. Setup minimal config
    # MDT architecture uses patch_size to determine the number of cuts.
    # We'll use 128 as a default patch_size for the model structure.
    cfg = DictConfig(
        {
            "dim": 48,
            "num_blocks": [2, 2, 2, 2],
            "num_refinement_blocks": 2,
            "ffn_expansion_factor": 3,
            "patch_size": 128,
        }
    )

    # 2. Initialize on CPU
    device = torch.device("cpu")
    model = MDT_Edited(cfg).to(device)
    model.eval()

    # 3. Use summary
    stats = summary(
        model,
        input_size=(1, 3, input_res[0], input_res[1]),
        device=device,
        depth=4,
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        verbose=0,
    )
    print(stats)


if __name__ == "__main__":
    try:
        import torchinfo
    except ImportError:
        print("torchinfo not found. Installing...")
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo"])

    # Test some resolutions
    debug_with_torchinfo(input_res=(128, 128))
    debug_with_torchinfo(input_res=(1280, 720))
