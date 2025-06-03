import os
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd

class RoadMapFinalizer:
    def __init__(self):
        pass

    @staticmethod
    def _create_lanes_gpkg(lanes, output_folder, crs="EPSG:32631"):
        geo_dict = {"element_id": [], "road_type": [], "lane_id": [], "from_object": [], "to_object": [], 
                "predecessors": [], "successors": [], "boundary_left": [], "boundary_right": [], "centerline": [],"geometry": []}
        for key, edge in lanes.items():
                geo_dict["element_id"].append(int(key))
                geo_dict["road_type"].append(1)
                geo_dict["lane_id"].append(int(key))
                geo_dict["from_object"].append(edge["from_object"])
                geo_dict["to_object"].append(edge["to_object"])
                geo_dict["predecessors"].append(repr(edge["predecessors"])[1:-1])
                geo_dict["successors"].append(repr(edge["successors"])[1:-1])
                boundary_left = tuple(tuple(arr) for arr in edge["boundary_left"])
                geo_dict["boundary_left"].append(repr(boundary_left)[1:-1])
                boundary_right = tuple(tuple(arr) for arr in edge["boundary_right"])
                geo_dict["boundary_right"].append(repr(boundary_right)[1:-1])
                geo_dict["centerline"].append(repr(list(edge["geometry"].coords))[1:-1])
                geo_dict["geometry"].append(Polygon(np.concatenate((edge["boundary_right"], edge["boundary_left"]))))
        
        gdf = gpd.GeoDataFrame(geo_dict, geometry='geometry')
        gdf.set_crs(crs, inplace=True, allow_override=True)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        gdf.to_file((os.path.join(output_folder, "lanes.gpkg")), layer='generated', driver='GPKG')
        return geo_dict

    @staticmethod
    def _create_connectors_gpkg(connectors, output_folder, crs="EPSG:32631"):
        geo_dict = {"connector_id": [], "source": [], "dest": [], "intersection_id": [], "left_boundary": [], 
                    "right_boundary": [], "lane_type": [], "legal": [], "polygon": [], "geometry": []}

        for key, connector in connectors.items():
            geo_dict["connector_id"].append(key)
            geo_dict["source"].append(connector["source"])
            geo_dict["dest"].append(connector["dest"])
            geo_dict["intersection_id"].append(connector["intersection_id"])
            geo_dict["left_boundary"].append(repr(tuple(tuple(c) for c in connector["left_boundary"]))[1:-1])
            geo_dict["right_boundary"].append(repr(tuple(tuple(c) for c in connector["right_boundary"]))[1:-1])
            geo_dict["lane_type"].append(1)
            geo_dict["legal"].append(True)
            geo_dict["polygon"].append(repr(tuple(tuple(c) for c in connector["polygon"]))[1:-1])
            geo_dict["geometry"].append(connector["geometry"])
        gdf = gpd.GeoDataFrame(geo_dict, geometry='geometry')
        
        gdf.set_crs(crs, inplace=True, allow_override=True)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        gdf.to_file((os.path.join(output_folder, "connectors.gpkg")), layer='generated', driver='GPKG')
        return geo_dict      
    
    @staticmethod
    def _create_polygons_gpkg(polygons, output_folder, crs="EPSG:32631"):
        geo_dict = {"element_id": [], "type": [], "road_type": [], "geometry": []}
        for key, polygon in polygons.items():
            geo_dict["element_id"].append(polygon["element_id"])
            geo_dict["type"].append(polygon["type"])
            geo_dict["road_type"].append(polygon["road_type"])
            geo_dict["geometry"].append(polygon["geometry"])
        gdf = gpd.GeoDataFrame(geo_dict, geometry="geometry")
        
        gdf.set_crs(crs, inplace=True, allow_override=True)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        gdf.to_file((os.path.join(output_folder, "polygons.gpkg")), layer='generated', driver='GPKG')
        return geo_dict

    @classmethod
    def generate_annotation_geopackages(cls, network_data: dict, save_dir: str, crs: str):
        cls._create_lanes_gpkg(network_data["original_edges"], save_dir, crs)
        cls._create_connectors_gpkg(network_data["connectors"], save_dir, crs)
        cls._create_polygons_gpkg(network_data["polygons"], save_dir, crs)