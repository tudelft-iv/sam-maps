import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config, rge, rs, rc):
        super(Model, self).__init__()

        self.config = config

        self.rge = rge
        self.rs = rs
        self.rc = rc

    def forward(self, inputs):
        rg = self.get_roadgraph(inputs)
        rs = self.get_semantic_mask(inputs, rg)
        rc = self.get_connected_map(rg, rs)

        return rg, rs, rc

    def get_roadgraph(self, inputs):
        return self.rge(inputs)

    def get_semantic_mask(self, inputs, roadgraph):
        return self.rs(inputs, roadgraph)

    def get_connected_map(self, roadgraph, semantic_mask):
        return self.rc(roadgraph, semantic_mask)
