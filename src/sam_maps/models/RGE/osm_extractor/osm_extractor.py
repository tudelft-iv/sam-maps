from sam_maps.models.RGE.rge import RGE
import ast
import osmnx as ox
import networkx as nx
from .utils.utils import OSMGraphPreprocessor



class OSMExtractor(RGE):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, inputs):
        if self.config.area_selection_method in ["annot_edges", "annot_bbox"]:
            G = OSMGraphPreprocessor.generate_graph_from_annotations(self.config.annot_file, self.config.from_crs, self.config.target_crs,
                                                                     network_type=self.config.network_type, method=self.config.area_selection_method)
            
        elif self.config.area_selection_method == "place":
            G = ox.graph_from_place(self.config.place, network_type=self.config.network_type)
        
        elif self.config.area_selection_method == "bbox":
            for i, bbox in enumerate(self.config.bbox):
                G_i = ox.graph_from_bbox(bbox, network_type=self.config.network_type)
                if i==0:
                    G = G_i
                else:
                    G = nx.disjoint_union(G, G_i)
        else:
            raise ValueError("Invalid Area Selection Method")
        
        graph = OSMGraphPreprocessor.graph_2_network_data(G, self.config.from_crs, self.config.target_crs, 
                                                          default_width=self.config.w_init, default_lanes=1)
        return graph
