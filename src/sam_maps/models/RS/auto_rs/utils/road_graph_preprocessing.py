import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import rasterio
from pyproj import Transformer
from .road_annotation_generator import RoadGenerator
import pickle
from collections import defaultdict


import osmnx as ox
from shapely.geometry import LineString, MultiLineString, Polygon, Point
from collections import defaultdict
import numpy as np
from pyproj import Transformer
from tqdm import tqdm
import networkx
import pickle
import argparse
import ast
import geopandas as gpd
from .tif_operator import TIFOperator
from .road_annotation_generator import RoadGenerator
from .utils import multi_line_string_2_line_string


def add_connections(network_data: dict, node_to_edge_ids: dict) -> dict:
    for edge_key, edge_data in network_data["original_edges"].items():
        from_node = int(edge_data["from_object"][1:])
        to_node = int(edge_data["to_object"][1:])
        reversed_edge_geom = np.array(edge_data["geometry"].coords)[::-1]

        # Finding predecessors
        for connected_edge in node_to_edge_ids[from_node]:
            if connected_edge != edge_key:  # Avoid self-references
                connected_edge_data = network_data["original_edges"][connected_edge]
                connected_edge_geom = np.array(connected_edge_data["geometry"].coords)
                if reversed_edge_geom.shape[0] == connected_edge_geom.shape[0]:
                    if np.allclose(reversed_edge_geom, connected_edge_geom):
                        continue

                # Exclude edges within the same lane group (same road)
                if connected_edge_data["lane_group"] != edge_data["lane_group"]:
                    if connected_edge_data["to_object"] == edge_data["from_object"]:
                        edge_data["pre_flip"].append(False)
                        edge_data["predecessors"].append(int(connected_edge))
                    elif connected_edge_data["from_object"] == edge_data["from_object"]:
                        edge_data["pre_flip"].append(True)
                        edge_data["predecessors"].append(int(connected_edge))

        # Finding successors
        for connected_edge in node_to_edge_ids[to_node]:
            if connected_edge != edge_key:
                connected_edge_data = network_data["original_edges"][connected_edge]
                connected_edge_geom = np.array(connected_edge_data["geometry"].coords)
                if reversed_edge_geom.shape[0] == connected_edge_geom.shape[0]:
                    if np.allclose(reversed_edge_geom, connected_edge_geom):
                        continue
                if connected_edge_data["lane_group"] != edge_data["lane_group"]:
                    if connected_edge_data["from_object"] == edge_data["to_object"]:
                        edge_data["successors"].append(int(connected_edge))
                        edge_data["suc_flip"].append(False)
                    elif connected_edge_data["to_object"] == edge_data["to_object"]:
                        edge_data["successors"].append(int(connected_edge))
                        edge_data["suc_flip"].append(True)
    return network_data

def trim_edge(geometry: LineString, trim_dist: float = 2.5, max_part_trim: float = 0.25) -> np.ndarray:
    """
    Trims `trim_dist` from both ends of a LineString geometry by following its points.
    If a point is fully within the trim distance, it is removed.
    The function interpolates a new point if the trim distance lies between two points.
    """
    # if max_part_trim * geometry.length <= 2 * min_trim_dist:
    #     return None  # Disregard edges shorter than 2 * trim_dist

    def trim_from_end(coords, trim_dist):
        accumulated_dist = 0
        trimmed_coords = []

        for i in range(len(coords) - 1):
            p1, p2 = np.array(coords[i]), np.array(coords[i + 1])
            segment_length = np.linalg.norm(p2 - p1)

            if accumulated_dist + segment_length > trim_dist:
                # Calculate interpolation factor to reach exact trim distance
                remaining_dist = trim_dist - accumulated_dist
                interpolation_factor = remaining_dist / segment_length
                new_point = p1 + interpolation_factor * (p2 - p1)
                trimmed_coords.append(new_point)
                trimmed_coords.extend(coords[i + 1:])
                break

            accumulated_dist += segment_length

        return trimmed_coords

    # Trim from both the start and end
    original_coords = list(geometry.coords)
    trim_dist = min(trim_dist, max_part_trim * geometry.length)
    trimmed_start = trim_from_end(original_coords, trim_dist)
    trimmed_end = trim_from_end(trimmed_start[::-1], trim_dist)[::-1]

    return LineString(trimmed_end)



def calculate_lane_bounds(geometry: LineString | MultiLineString, width: float, lanes: int, 
                          mode: str = "constant_width") -> tuple[list, list, list, float, int]:
    """
    Calculate the bounds of lanes based on the given method
    'constant_width': creates one lane with one common width for all lanes
    'variable_width': creates multiple lanes with the prescribed width of the lanes given by osm
    'variable_width_one_lane': uses amount of lanes and given width to compute one common lane
    """

    left_bounds = []
    right_bounds = []
    lane_geometries = []

    if mode == "constant_width":
        lanes = 1

    elif mode =="variable_width_one_lane":
        width = width * lanes
        lanes = 1

    for lane_idx in range(lanes):
        offset = (lane_idx - (lanes - 1)/2) * width
        # offset = (lane_idx + 0.5) * width  # Shift centerline for each lane

        lane_geometry =  multi_line_string_2_line_string(geometry.parallel_offset(offset, side='left'))
        lane_geometries.append(lane_geometry)  # Store the lane geometry
        
        lb = multi_line_string_2_line_string(geometry.parallel_offset(width/2, side='left'))
        rb = multi_line_string_2_line_string(geometry.parallel_offset(width/2, side='right'))

        left_bound = np.array(lb.coords)[::-1]
        right_bound = np.array(rb.coords)
        left_bounds.append(left_bound)
        right_bounds.append(right_bound)

    return left_bounds, right_bounds, lane_geometries, width, lanes


class OSMGraphPreprocessor:
    def __init__(self):
        pass


    @classmethod
    def generate_graph_from_annotations(cls, annot: str, from_crs: str, target_crs: str, network_type: str = "drive", method: str = "annot", extend_factor: float = 0.05) -> networkx.classes.multidigraph.MultiDiGraph:
            """
            This method uses Polygon annotations to extract edges within those annotations.
            There are 2 approaches: 
            'annot': This takes the edges within the polygons
            'bbox': This takes all edges within the bounding box within which all the annotations are
            """
            # Extract the Annotations
            gdf = gpd.read_file(annot)
            gdf.to_crs(from_crs)
            annotated_pts = []
            for g in gdf.geometry:
                if g.geom_type == 'Polygon':
                    annotated_pts.append(list(np.array(list(g.exterior.coords))))
            
            # Transform to long, lat coordinates
            transformer = Transformer.from_crs(target_crs, from_crs, always_xy=True)
            transformed_pts = []
            all_pts = []
            for edge in annotated_pts:
                new_pts = TIFOperator.transform_edge_to_tif(np.array(edge), transformer)
                transformed_pts.append(new_pts)

                all_pts += new_pts.tolist()
            all_pts = np.array(all_pts)

            # Get minimal rectangle box that encapsulates all the annotations
            dlat = np.max(all_pts[:,0]) - np.min(all_pts[:,0])
            dlong = np.max(all_pts[:,1]) - np.min(all_pts[:,1])
            bounds = np.array([np.min(all_pts[:,0])-dlat*extend_factor, np.min(all_pts[:,1])-dlong*extend_factor, np.max(all_pts[:,0])+dlat*extend_factor, np.max(all_pts[:,1])+dlong*extend_factor])
            # bbox = np.array([bounds[3], bounds[1], bounds[2], bounds[0]])


            # Get all the roads within the given Bounding box
            # custom_filter = (
            #     '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|'
            #     'secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|'
            #     'living_street|service|track|road"]'
            #     '["access"!~"no"]["motor_vehicle"!~"no"]["vehicle"!~"no"]'
            # )
            G = ox.graph_from_bbox(bounds, network_type=network_type) #custom_filter=custom_filter) #
            ox.plot_graph(G)
            if method == "annot_bbox":
                return G
            edges_within_polygon = []

            for polygon_road in tqdm(transformed_pts):
                polygon = Polygon(polygon_road)
                for u, v, key, data in G.edges(keys=True, data=True):
                    # Check if the edge has geometry attribute
                    if 'geometry' in data:
                        # Extract the LineString geometry of the edge
                        line = data['geometry']
                        # Check if any point in the line falls within the polygon
                        if any(polygon.contains(Point(coord)) for coord in line.coords):
                            # Add this edge if it is within the annotated polygon
                            edges_within_polygon.append((u, v, key))
                    else:
                        # Check the start and end nodes of the edge (in case there is no additional geometry data)
                        start = Point(G.nodes[u]['x'], G.nodes[u]['y'])
                        end = Point(G.nodes[v]['x'], G.nodes[v]['y'])
                        if polygon.contains(start) or polygon.contains(end):
                            # Add this edge if it is within the annotated polygon
                            edges_within_polygon.append((u, v, key))    
            # Create graph of the relevant edges
            subgraph = G.edge_subgraph(edges_within_polygon)
            return subgraph

    @classmethod
    def graph_2_network_data(cls, G: networkx.classes.multidigraph.MultiDiGraph, from_crs: str, target_crs: str, default_width: float = 3.5, 
                            default_lanes: int = 1, mode: str = "constant_width", trim_dist: float = 2.5, max_part_trim: float = 0.4) -> dict:
        """
        This function is the main function of using a graph from OSM
        to generate the data required to form road masks
        We generate a dictionary with the keys:
        - 'intersections': containing information of the which lanes intersect
        - 'original_edges': containing all the relevant info off the lanes
        - 'connectors': provides the actual connections made between the lanes at the intersections 
        """
        network_data = {
            "intersections": {},
            "original_edges": {}
        }

        transformer = Transformer.from_crs(from_crs, target_crs, always_xy=True)

        for node_id, node_data in G.nodes(data=True):
            intersection_key = f"i{node_id}"
            network_data["intersections"][intersection_key] = {
                "point": transformer.transform(node_data["x"], node_data["y"])
            }

        node_to_edge_ids = defaultdict(list)
        geoms_lst = []
        edge_step = 0
        
        # Manual step used for matching boundary detection
        # relevant_edge_ids = [605,604,741,482,481,603,716,441,440,715,449,711,716,714,713,712,710,
        #                      514,707,709,708,706,705,386,516,393,405,392,704,382,703,702,701,
        #                      494,493,700,391,390,402,383,615,810,743,747,655,399,387,653,374,613,
        #                      364,350,352,351,344,343,342,347,503,338,502,914,913,919,650,920,921,
        #                      771,875,976,876,952,970,969,955,953,976,980,992,954,957,311,330,325,
        #                      310,956,994,982,978,981,879,977,887,888,889,979,877,995,929,878,930,
        #                      598,301,597,596,599,222,600,601,302,224,223,234,235,642,263,237,247,
        #                      260,246,602,261,262,256,279,278,288,281,594,752,79,489,490,894,24,26,
        #                      80,56,48,57,759,474,760,759,474,1052,1050,1051,1064,1062,1027,1061,1055,
        #                      1063,1032,968,967,966,964,965,962,663,4,963,17,25,274,498,662]

        for edge_id, (u, v, key, data) in tqdm(enumerate(G.edges(keys=True, data=True))):

            from_obj = f"i{u}"
            to_obj = f"i{v}"
            
            if "geometry" in data:
                geometry = np.array(data["geometry"].coords)
            else:
                start_point = (G.nodes[u]["x"], G.nodes[u]["y"])
                end_point = (G.nodes[v]["x"], G.nodes[v]["y"])
                geometry = np.array([start_point, end_point])

            transformed_coords_x, transformed_coords_y = transformer.transform(geometry[:,0], geometry[:,1])
            transformed_coords = np.vstack((transformed_coords_x, transformed_coords_y)).T
            transformed_geometry = LineString(transformed_coords)
            rev_lane = LineString(transformed_geometry.coords[::-1]) 
            if rev_lane in geoms_lst:
                continue
            geoms_lst.append(transformed_geometry)

            transformed_geometry = trim_edge(transformed_geometry, trim_dist=trim_dist, max_part_trim=max_part_trim)
            if transformed_geometry is None:
                continue  

            length = float(transformed_geometry.length)
            width = data.get("width", default_width) if mode != "constant_width" else default_width
            width = float(width[0]) if isinstance(width, list) else float(width)
            lanes = data.get("lanes", default_lanes)
            lanes = int(lanes[0]) if isinstance(lanes, list) else int(lanes)

            boundary_left, boundary_right, lane_geometries, width, lanes = calculate_lane_bounds(transformed_geometry, width, lanes, mode)
            
            # if edge_step not in relevant_edge_ids:
            #     edge_step+=1
            #     continue

            for lane_idx in range(lanes):
                lane_key = f"{edge_step + lane_idx}"  # Unique key for each lane
                lane_group = edge_step # Assign the lane group to the original edge ID
                # rev_lane = LineString(lane_geometries[lane_idx].coords[::-1]) 
                # if (rev_lane in geoms_lst):
                #     unique_edge_group = np.argwhere(np.array(geoms_lst)==rev_lane).flatten()[0]
                # else:
                #     unique_edge_group = lane_key
                network_data["original_edges"][lane_key] = {
                    "from_object": from_obj,
                    "to_object": to_obj,
                    "geometry": lane_geometries[lane_idx],
                    "crs": target_crs,
                    "width": width,
                    "length": length,
                    "lanes": lanes,
                    "boundary_left": boundary_left[lane_idx],
                    "boundary_right": boundary_right[lane_idx],
                    "lane_group": lane_group,
                    "predecessors": [],
                    "successors": [],
                    "pre_flip": [],
                    "suc_flip": [],
                }
                node_to_edge_ids[u].append(lane_key)
                node_to_edge_ids[v].append(lane_key)
            edge_step += lanes
        network_data = add_connections(network_data, node_to_edge_ids)
        
        # network_data = RoadGenerator.build_connectors(network_data)    
        # network_data = RoadGenerator.build_polygons(network_data, extend_dist=trim_dist)
        return network_data




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
    def graph_2_network_data(cls, G: dict, trim_dist: float = 2.5, max_part_trim: float = 0.5,
                          default_width: float = 3.5, default_lanes: int = 1) -> dict:
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
                lanes = default_lanes
                boundary_left, boundary_right, lane_geometries, width, lanes = calculate_lane_bounds(trimmed_edge, width, lanes, mode="variable_width_one_lane")

                for lane_idx in range(lanes):
                    lane_key = f"{edge_step + lane_idx}"  # Unique key for each lane
                    lane_group = edge_step # Assign the lane group to the original edge ID

                    network_data["original_edges"][lane_key] = {
                                                            "from_object": from_obj,
                                                            "to_object": to_obj,
                                                            "geometry": lane_geometries[lane_idx],
                                                            "crs": crs,
                                                            "width": width,
                                                            "length": length,
                                                            "lanes": lanes,
                                                            "boundary_left": boundary_left[lane_idx],
                                                            "boundary_right": boundary_right[lane_idx],
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
        network_data = RoadGenerator.build_connectors(network_data) 
        network_data = RoadGenerator.build_polygons(network_data, extend_dist=trim_dist)
        return network_data


