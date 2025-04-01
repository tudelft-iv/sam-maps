import torch.nn as nn


class RS(nn.Module):
    def __init__(self, config):
        super(RS, self).__init__()
        self.config = config

    def forward(self, inputs):
        raise NotImplementedError
