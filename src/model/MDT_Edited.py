from .archs.mdt_arch import mdt


class MDT_Edited(mdt):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x, depth=None):
        out_list = super().forward(x, dist=None, depth=depth)
        return {"sharp_image": out_list[0]}
