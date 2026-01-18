import torch.nn as nn
from .archs.mdt_dpt_arch import MDT_DPT_Impl

class MDT_DPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Extract cfg params just like MDT_Edited
        dim = cfg.get("dim", 48)
        num_blocks = cfg.get("num_blocks", [6, 6, 12, 8])
        num_refinement_blocks = cfg.get("num_refinement_blocks", 4)
        ffn_expansion_factor = cfg.get("ffn_expansion_factor", 3)
        patch_size = cfg.get("patch_size", 128)
        
        self.net = MDT_DPT_Impl(
            cfg=cfg,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            ffn_expansion_factor=ffn_expansion_factor,
            patch_size=patch_size
        )
    
    def forward(self, x):
        out_list = self.net(x)
        return {"sharp_image": out_list[0]}