from .archs.mdt_arch import mdt


class MDT_Edited(mdt):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        out_list = super().forward(x, dist=None)
        return {"sharp_image": out_list[0]}
