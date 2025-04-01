import torch.nn as nn


class RC(nn.Module):
    def __init__(self, config):
        super(RC, self).__init__()
        self.config = config

    def forward(self, inputs):
        raise NotImplementedError
