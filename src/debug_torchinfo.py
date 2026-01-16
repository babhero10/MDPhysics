import torch
from model.MDPhysics import DPT
from omegaconf import DictConfig
from torchinfo import summary

def debug_with_torchinfo(input_res=(256, 256)):
    print(f"\n--- Torchinfo Summary for: {input_res} ---")
    
    # 1. Setup minimal config
    cfg = DictConfig({
        "dim": 48,
        "num_blocks": [2, 2, 2, 2],
        "num_refinement_blocks": 2,
        "ffn_expansion_factor": 3,
        "img_size": input_res,
        "backbone": {
            "name": "facebook/dinov2-small",
            "patch_size": 14
        }
    })

    # 2. Initialize on CPU
    device = torch.device('cpu')
    model = DPT(cfg).to(device)
    model.eval()

    # 3. Use summary
    # depth=4 allows you to see inside the TransformerBlocks
    # col_names includes input/output size and params
    stats = summary(
        model, 
        input_size=(1, 3, input_res[0], input_res[1]),
        device=device,
        depth=4,
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        verbose=0 # Set to 1 if you want to see the progress
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
    
    # Test 256x256
    debug_with_torchinfo(input_res=(340, 640))
    
    # Test a custom resolution that used to crash
    debug_with_torchinfo(input_res=(1280, 740))
