import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config, rge, rs, rc):
        super(Model, self).__init__()

        self.config = config

        self.rge = rge
        self.rs = rs
        self.rc = rc

    def forward(self, inputs):
        rg = self.rge(inputs)
        rs = self.rs(inputs, rg)
        rc = self.rc(rg, rs)

        return rg, rs, rc
