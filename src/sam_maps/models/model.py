import torch.nn as nn
import torch
import gc

class Model(nn.Module):
    def __init__(self, config, rge, rs, rc, config_path=None):
        super(Model, self).__init__()

        self.config = config

        self.rge = rge
        self.rs = rs
        self.rc = rc
        self.config_path = config_path

    def forward(self, inputs):
        rg = self.get_roadgraph(inputs)
        if self.config.rge_module.model_name == "samroad":   
            if self.config.manual and ((self.config.graph.edges is None) or (self.config.graph.nodes is None)):
                print("Graph is ready to be manually adapted.")
                return
        gc.collect()
        torch.cuda.empty_cache()
        rs = self.get_semantic_mask(inputs, rg)
        rc = self.get_connected_map(rg, rs)

        return rg, rs, rc

    def get_roadgraph(self, inputs):
        return self.rge(inputs)

    def get_semantic_mask(self, inputs, roadgraph):
        return self.rs(inputs, roadgraph)

    def get_connected_map(self, roadgraph, semantic_mask):
        return self.rc(roadgraph, semantic_mask)
