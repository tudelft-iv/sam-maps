import os

import numpy as np
import torch
import cv2 as cv
import ast 
from samgeo import tms_to_geotiff
import tifffile as tifi
import rasterio
from rasterio.enums import Resampling
import pickle
from collections import defaultdict
from addict import Dict
from shapely.geometry import LineString
import geopandas as gpd
from typing import Optional
import sys

samroad_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "sam_road")
if samroad_path not in sys.path:
    sys.path.append(samroad_path)


from pyproj import Transformer

from triage import visualize_image_and_graph
from dataset import get_patch_info_one_img
from utils import load_config
from model import SAMRoad
from inferencer import infer_one_img
from graph_utils import convert_to_sat2graph_format
from sam_maps.models.RGE.utils.graph_utils import add_connections, trim_edge
from sam_maps.models.RGE.utils.graph_conversion import convert_graph_to_geojson, filter_samroad_graph


class SAMRoadGraphPreprocessor:
    def __init__(self):
        pass


    @classmethod
    def _get_next_node(cls, nodes: list, G: dict, intersections: list) -> list:
        final_node = nodes[-1]
        if final_node in intersections:
            return nodes
        new_node = np.argwhere([n not in nodes for n in G[final_node]])   
        if len(new_node) != 0:
            nodes.append(G[final_node][new_node.flatten()[0]])
            return cls._get_next_node(nodes, G, intersections)
        else:
            if nodes[0] in G[final_node] and len(nodes) > 2:

                nodes.append(nodes[0])
            return nodes


    @classmethod
    def graph_2_network_data(cls, G: dict, trim_dist: float = 2.5, max_part_trim: float = 0.5, default_width: float = 3.5) -> dict:
        network_data = {}
        intersections = []
        crs = G["crs"]
        G = G["geolocalized_graph"]

        intersection_counter = 0
        network_data = {"intersections": {}, "original_edges": {}, "connectors": {}}
        intersection_to_key = {}
        for key, connections in G.items():
            if  len(connections) > 2 or len(connections)==1:
                intersections.append(key)
                intersection_id = f"i{intersection_counter}"
                network_data["intersections"][intersection_id] = {"point": key}
                intersection_to_key[key] = intersection_id
                intersection_counter +=1
        
        roads = {}
        all_roads = []
        node_to_edge_ids = defaultdict(list)
        edge_step = 0
        for i, intersection in enumerate(intersections):
            for j, n in enumerate(G[intersection]):
                pts = cls._get_next_node([intersection, n], G, intersections)
                roads_list = [list(pt) for pt in pts]
                
                if LineString(roads_list[::-1]) in all_roads:
                    continue
                from_obj = intersection_to_key[tuple(roads_list[0])]
                to_obj = intersection_to_key[tuple(roads_list[-1])]
                all_roads.append(LineString(roads_list))
                edge = LineString(roads_list) # roads[f"({', '.join(map(str, roads_list[0]))})_({', '.join(map(str, roads_list[-1]))})"]
                trimmed_edge = trim_edge(edge, trim_dist, max_part_trim)
                length = float(trimmed_edge.length)
                width = default_width
                lanes = 1 # default_lanes
                # boundary_left, boundary_right, lane_geometries, width, lanes = calculate_lane_bounds(trimmed_edge, width, lanes, mode="variable_width_one_lane")

                for lane_idx in range(lanes):
                    lane_key = f"{edge_step + lane_idx}"  # Unique key for each lane
                    lane_group = edge_step # Assign the lane group to the original edge ID

                    network_data["original_edges"][lane_key] = {
                                                            "from_object": from_obj,
                                                            "to_object": to_obj,
                                                            "geometry": trimmed_edge,
                                                            "original_edge": edge,
                                                            # "geometry": lane_geometries[lane_idx],
                                                            "crs": crs,
                                                            "width": width,
                                                            "length": length,
                                                            "lanes": lanes,
                                                            # "boundary_left": boundary_left[lane_idx],
                                                            # "boundary_right": boundary_right[lane_idx],
                                                            "lane_group": lane_group,
                                                            "predecessors": [],
                                                            "successors": [],
                                                            "pre_flip": [],
                                                            "suc_flip": [],
                                                            }
                    node_to_edge_ids[int(from_obj[1:])].append(lane_key)
                    node_to_edge_ids[int(to_obj[1:])].append(lane_key)
                edge_step += lanes
        network_data = add_connections(network_data, node_to_edge_ids)
        return network_data



def resample_tif_to_resolution(input_tif: str, output_tif: str, target_resolution: float = 1.0, image_size: int = 2048):
    """
    Resample a GeoTIFF to a specified resolution.

    Parameters:
        input_tif (str): Path to the input GeoTIFF file.
        output_tif (str): Path to save the resampled GeoTIFF.
        target_resolution (float): Desired resolution in meters per pixel.
    """
    with rasterio.open(input_tif) as src:
        # Extract original resolution
        original_resolution_x = src.transform[0]
        original_resolution_y = abs(src.transform[4])
        print(f"Original resolution: {original_resolution_x} x {original_resolution_y}")

        # Calculate scaling factors
        scale_x = original_resolution_x / target_resolution
        scale_y = original_resolution_y / target_resolution

        # Calculate new dimensions
        new_width = int(src.width * scale_x)
        new_height = int(src.height * scale_y)
        print(f"New dimensions: {new_width} x {new_height}")

        # Validate dimensions
        if new_width <= 0 or new_height <= 0:
            raise ValueError("Calculated dimensions are invalid. Check the target resolution or input file.")

        transform = src.transform * src.transform.scale(
            src.width / new_width,
            src.height / new_height
        )

        # Resample the data
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear
        )

        # Update metadata
        profile = src.profile
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': transform
        })

        # Save the resampled image
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(data)

def transform_graph_2_coords(G, tif_file_path, target_crs = "EPSG:32631") -> dict:
    """
    transform the graph from pixel coordinates to geolocalized coordinates
    """
    with rasterio.open(tif_file_path) as src:
        # Define pixel coordinates (row, col) you want to convert
        transformer = Transformer.from_crs(src.crs, target_crs, always_xy=True)
        G_transformed = {}
        for key, val in G.items():
            key_new = src.transform * (key[1], key[0])
            key_new = transformer.transform(key_new[0], key_new[1])

            G_transformed[key_new] = []
            for pt in val:
                pt_new = src.transform * (pt[1], pt[0])
                pt_new = transformer.transform(pt_new[0], pt_new[1])
                G_transformed[key_new].append(pt_new)
            
    return G_transformed


def samroad_graph_generator(config: Dict, bboxes: list, output_dir: str, gpu_id: Optional[int] = None) -> dict:
    """
    For a list of bounding boxes, generate the SAMRoad graph and combine the graphs and save it as a pickle file.
    """
    # Obtain TIF
    # config = load_config(config)
    config_samroad = load_config(config.samroad_config)
    out_dir = output_dir
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")   
    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    net = SAMRoad(config_samroad)

    # load checkpoint
    checkpoint = torch.load(config.samroad_checkpoint, map_location="cpu")
    
    print(f'##### Loading Trained CKPT {config.samroad_checkpoint} #####')
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    net.to(device)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    print("Generate the Geo-localized Satellite Image")
    complete_graph = {}
    bboxes = list(bboxes)
    for i, bbox in enumerate(bboxes):
        bbox = list(bbox)

        output_dir = os.path.join(out_dir, f"_{i}")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        satellite_file = os.path.join(output_dir, "satellite.tif")
        tms_to_geotiff(output=os.path.join(output_dir, "satellite.tif"), bbox=bbox, zoom=config.zoom, source="Satellite", overwrite=True, quiet=True)
        
        resampled_satellite_file =  os.path.join(output_dir, "satellite_resampled.tif")
        resample_tif_to_resolution(satellite_file, resampled_satellite_file, target_resolution=config.resolution, image_size=config.image_size)


        img = tifi.imread(resampled_satellite_file)

        if (img.shape[0] > config.image_size) or (img.shape[1] > config.image_size):
            print(f"WARNING: the bounding box is divided into equal parts to fit the image size: {config.image_size}x{config.image_size}, this could lead to discontinuities in the graph")
       
        print("Predict SAMRoad graph")
        img, pred_nodes, pred_edges = pred_samroad_graph(img, net, config, config_samroad, device)
        
        viz_img = np.copy(img)
        img_size = viz_img.shape[0]

        viz_img = visualize_image_and_graph(viz_img, pred_nodes / max(img.shape[0], img.shape[1]), pred_edges, img_size)
        cv.imwrite(os.path.join(output_dir, 'graph_prediction.png'), viz_img)
        
        if config_samroad.DATASET == 'spacenet':
            pred_nodes = np.stack([400 - pred_nodes[:, 0], pred_nodes[:, 1]], axis=1)
        print("Generate Graph")
        samroad_graph_pixels = convert_to_sat2graph_format(pred_nodes, pred_edges)
        samroad_graph_coords = transform_graph_2_coords(samroad_graph_pixels, resampled_satellite_file, target_crs=config.target_crs)
        combined_graph = {"crs": config.target_crs, "pixel_graph": samroad_graph_pixels, "geolocalized_graph": samroad_graph_coords}
        complete_graph.update(combined_graph["geolocalized_graph"])
        graph_save_path = os.path.join(output_dir, 'graph.pkl')
        with open(graph_save_path, 'wb') as file:
            pickle.dump(combined_graph, file)
    final_graph = {"crs": config.target_crs, "geolocalized_graph": complete_graph}
    final_graph_save_path = os.path.join(out_dir, 'final_graph.pkl')
    with open(final_graph_save_path, 'wb') as file:
        pickle.dump(final_graph, file)
    if config.apply_mask and os.path.isfile(config.annot_file):
        lanes = gpd.read_file(config.annot_file)
        final_graph = filter_samroad_graph(final_graph, lanes, out_dir)
    # Convert the graph to GeoJSON
    convert_graph_to_geojson(final_graph, out_dir)
    del net
    return final_graph

def pred_samroad_graph(img, net, config, config_samroad, device):
    imgs_x = np.ceil(img.shape[0]/int(config.image_size)).astype(int) 
    imgs_y = np.ceil(img.shape[1]/int(config.image_size)).astype(int)
    img_xsize = img.shape[0]//imgs_x
    img_ysize = img.shape[1]//imgs_y
    final_pred_nodes = []
    final_pred_edges = []
    img_final = np.zeros((max(img.shape[0], img.shape[1], config.image_size), max(img.shape[0], img.shape[1], config.image_size), 3), dtype=np.uint8)
    img_final[:img.shape[0], :img.shape[1]] = img
    tot_nodes = 0
    for i in range(imgs_x):
        for j in range(imgs_y):
            img_patch = np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8)
            img_patch[:img_xsize, :img_ysize] = img[i*img_xsize:(i+1)*img_xsize, j*img_ysize:(j+1)*img_ysize]
            pred_nodes, pred_edges, _, _ = infer_one_img(net, img_patch, config_samroad, device)
            
            # Transform the results properly to the original image
            pred_nodes[:, 0] += i*img_xsize
            pred_nodes[:, 1] += j*img_ysize
            
            pred_edges[:, 0] += tot_nodes
            pred_edges[:, 1] += tot_nodes
            tot_nodes+= pred_nodes.shape[0]
            final_pred_nodes.append(pred_nodes)
            final_pred_edges.append(pred_edges)

    final_pred_nodes = np.concatenate(final_pred_nodes, axis=0)
    final_pred_edges = np.concatenate(final_pred_edges, axis=0)

    return img_final, final_pred_nodes, final_pred_edges

    
