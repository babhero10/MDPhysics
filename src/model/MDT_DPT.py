import torch.nn as nn
from .archs.mdt_dpt_arch import mdt


class MDT_DPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = mdt(cfg)

    def forward(self, x):
        out = self.net(x)
        return {"sharp_image": out[0]}
