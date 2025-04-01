from torch.nn import nn


class RGE(nn.Module):
    def __init__(self, config):
        super(RGE, self).__init__()
        self.config = config

    def forward(self, inputs):
        raise NotImplementedError
