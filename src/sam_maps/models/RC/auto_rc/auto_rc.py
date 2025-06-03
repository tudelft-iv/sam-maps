import numpy as np
from shapely.geometry import LineString, Polygon
from scipy.spatial import ConvexHull
from copy import deepcopy
import os


from sam_maps.models.RC.utils.utils import RoadMapFinalizer
from sam_maps.models.RC.rc import RC
from sam_maps.utils.utils import multi_line_string_2_line_string
from sam_maps.models.RC.utils.map_conversion import generate_vod_format_map


class AutoRC(RC):
    def __init__(self, config):
        super().__init__(config)
    

    @staticmethod
    def _add_connector(network_data, edge_key, successor_key, intersection_id, predecessor=False, flipped=False):
        connector_id = f"{edge_key}_{successor_key}"
        if connector_id in network_data.keys():
            return network_data

        source_data = network_data["original_edges"][edge_key]
        dest_data = network_data["original_edges"][successor_key]

        source_left = source_data["boundary_left"][::-1]
        source_right = source_data["boundary_right"]
        dest_left = dest_data["boundary_left"][::-1]
        dest_right = dest_data["boundary_right"]
        source_line = np.array(source_data["geometry"].coords)
        dest_line = np.array(dest_data["geometry"].coords)

        if predecessor: 
            source_left_end = source_left[-1] if not flipped else source_left[0]
            source_right_end = source_right[-1] if not flipped else source_right[0]
            dest_left_end = dest_left[0] 
            dest_right_end = dest_right[0] 
            start = source_line[-1] if not flipped else source_line[0]
            end = dest_line[0]
        else:
            source_left_end = source_left[-1]
            source_right_end = source_right[-1]
            dest_left_end = dest_left[0] if not flipped else dest_left[-1]
            dest_right_end = dest_right[0] if not flipped else dest_right[-1]
            start = source_line[-1] 
            end = dest_line[0] if not flipped else dest_line[-1]

        
        left_boundary = np.array([source_left_end, dest_left_end])[::-1]
        right_boundary = np.array([source_right_end, dest_right_end])
        polygon = np.vstack((right_boundary, left_boundary))

        centerline_geom = LineString([start, end])
        
        network_data["connectors"][connector_id] = {
            "connector_id": connector_id,
            "source": edge_key,
            "dest": successor_key,
            "intersection_id": intersection_id,
            "left_boundary": left_boundary,
            "right_boundary": right_boundary,
            "polygon": polygon,
            "geometry": centerline_geom
        }

        return network_data
    
    @classmethod
    def build_connectors(cls, network_data: dict) -> dict:
        network_data["connectors"] = {}
        for edge_key, edge_data in network_data["original_edges"].items():
            for successor_key, suc_flip in zip(edge_data["successors"], edge_data["suc_flip"]):
                intersection_id = edge_data["to_object"]
                network_data = cls._add_connector(network_data, edge_key, str(successor_key), intersection_id, predecessor=False, flipped=suc_flip)

            for predecessor_key, pre_flip in zip(edge_data["predecessors"], edge_data["pre_flip"]):
                intersection_id = edge_data["from_object"]
                network_data = cls._add_connector(network_data, str(predecessor_key), edge_key, intersection_id, predecessor=True, flipped=pre_flip)

        return network_data


    @classmethod
    def build_polygons(cls, network_data: dict, extend_dist: float = 2.5) -> dict:
        network_data["polygons"] = {}
        empty_intersections = []
        for i, intersect in enumerate(network_data["intersections"].keys()):
            pts = np.empty((0,2))
            for connector_key, connector in network_data["connectors"].items():
                if (connector["intersection_id"])==intersect:
                    pts = np.concatenate((pts, connector["left_boundary"], connector["right_boundary"]))
            pts = np.unique(pts, axis=0)
            if len(pts) == 0:
                empty_intersections.append(intersect)
                continue  

            elif len(pts) > 2:
                polygon_type = "intersection"
                hull = ConvexHull(pts)
                hull_points = pts[hull.vertices]
                polygon = Polygon(hull_points)
 
            else:
                polygon_type = "terminal"
                objs = [data_object for data_object in network_data["original_edges"].values() if data_object["from_object"]==intersect]
                flip = False
                if len(objs) == 0:
                    objs = [data_object for data_object in network_data["original_edges"].values() if data_object["to_object"]==intersect]
                    flip = True
                obj = objs[0]
                if flip:
                    idx1 = -1
                    idx2 = 0
                    dist = extend_dist
                else:
                    dist = -extend_dist
                    idx1 = 0
                    idx2 = -1
                rps = np.array(obj["boundary_right"])
                lps = np.array(obj["boundary_left"])
                pts = np.array([rps[idx1], lps[idx2]])
                rd = - (rps[idx1]-rps[int(idx1 + np.sign(idx1+0.5))]) * np.sign(idx1+0.5)
                ld = - (rps[idx2]-rps[int(idx2 + np.sign(idx2+0.5))]) * np.sign(idx2+0.5)
                rcs = np.array([rd/(np.linalg.norm(rd)), ld/np.linalg.norm(ld)])
                new_pts = pts + rcs * dist
                pts = np.array([pts[0], new_pts[0], new_pts[1], pts[1], pts[0]])
                polygon = Polygon(pts)
            network_data["polygons"][str(i)] = {"element_id": i, "type": polygon_type, "road_type": None, "geometry": polygon}

        
        # Remove empty intersections
        for intersection in empty_intersections:
            network_data["intersections"].pop(intersection)
        return network_data

    
    def finalize_map_generation(self, network_data: dict, results: dict, save_dir: str):
        adapted_network_data = deepcopy(network_data)
        adapted_network_data["original_edges"] = self._format_lanes(results, network_data)
        adapted_network_data = self.build_connectors(adapted_network_data)
        adapted_network_data = self.build_polygons(adapted_network_data)
        RoadMapFinalizer.generate_annotation_geopackages(adapted_network_data, save_dir, self.config.target_crs)
        self.create_prediction_map(save_dir, os.path.join(self.config.pred_datapath, self.config.experiment_name))
    
    def create_prediction_map(self, map_dir: str, save_dir: str):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        generate_vod_format_map(map_dir, save_dir)


    @staticmethod
    def _format_lanes(warped_results: dict, network_data: dict):
        formatted_lanes = deepcopy(network_data["original_edges"])
        for key in formatted_lanes.keys():
            formatted_lanes[key]["geometry"] = multi_line_string_2_line_string(warped_results[key]["combined_lanes"]["geometry"])
            formatted_lanes[key]["boundary_left"] = warped_results[key]["combined_lanes"]["boundary_left"][::-1]
            # formatted_lanes[key]["boundary_left"] = warped_results[key]["combined_lanes"]["boundary_left"]
            formatted_lanes[key]["boundary_right"] = warped_results[key]["combined_lanes"]["boundary_right"]
            formatted_lanes[key]["length"] = warped_results[key]["combined_lanes"]["length"]
            # formatted_lanes[key]["width"] = warped_results[key]["combined_lanes"]["width"]
        return formatted_lanes
    
    def forward(self, road_graph, road_segmentation):
        save_dir = os.path.join(self.config.pred_datapath, self.config.experiment_name, "RC")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.finalize_map_generation(road_graph, road_segmentation, save_dir)
