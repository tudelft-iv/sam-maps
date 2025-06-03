from sam_maps.models.RGE.rge import RGE
import ast
import os
from .utils.utils import samroad_graph_generator, SAMRoadGraphPreprocessor
from sam_maps.models.RGE.utils.graph_conversion import convert_geojson_to_graph



class SAMRoad(RGE):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, inputs):
        output_dir = os.path.join(self.config.pred_datapath, self.config.experiment_name, "RGE")
        os.makedirs(output_dir, exist_ok=True)
        if self.config.manual and ((self.config.graph.edges is not None) and (self.config.graph.nodes is not None)):
            graph = convert_geojson_to_graph(self.config.graph.nodes, self.config.graph.edges, output_dir)
        else:
            if self.config.bbox is None:
                raise ValueError("Bboxes or a graph must be provided in the config.")
            bboxes = self.config.bbox

            graph = samroad_graph_generator(self.config, bboxes, output_dir, gpu_id = self.config.gpus[0])
        if (not self.config.manual) or (self.config.graph is not None):
            graph = SAMRoadGraphPreprocessor.graph_2_network_data(graph, self.config.trim_dist, self.config.max_part_trim, self.config.w_init)
        return graph
