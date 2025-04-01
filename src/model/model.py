import torch.nn as nn


class SAM_Maps(nn.Module):
    def __init__(self, rge, rs, rc, config):
        self.config = config
        self.rge = rge
        self.rs = rs
        self.rc = rc

    def forward(self, x):
        rg = self.rge(x)
        rs = self.rs(x, rg)
        rc = self.rc(rg, rs)

        return rg, rs, rc
