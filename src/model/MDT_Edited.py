from .archs.mdt_arch import mdt


class MDT_Edited(mdt):
    def __init__(self, cfg):
        dim = cfg.get("dim", 48)
        num_blocks = cfg.get("num_blocks", [6, 6, 12, 8])
        num_refinement_blocks = cfg.get("num_refinement_blocks", 4)
        ffn_expansion_factor = cfg.get("ffn_expansion_factor", 3)
        patch_size = cfg.get("patch_size", 128)

        super().__init__(
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            ffn_expansion_factor=ffn_expansion_factor,
            patch_size=patch_size,
        )

    def forward(self, x):
        out_list = super().forward(x, dist=None)
        return {"sharp_image": out_list[0]}
