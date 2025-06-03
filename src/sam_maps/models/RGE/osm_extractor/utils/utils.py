from collections import defaultdict
import os
import numpy as np
import geopandas as gpd
import networkx
from shapely.geometry import LineString, Point, Polygon
import osmnx as ox
import rasterio
from tqdm import tqdm
from pyproj import Transformer
from sam_maps.models.RGE.utils.graph_utils import add_connections, trim_edge
from sam_maps.utils.tif_operator import TIFOperator



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
            transformer = Transformer.from_crs(gdf.crs, from_crs, always_xy=True)
            transformed_pts = []
            all_pts = []
            for edge in annotated_pts:
                new_pts = TIFOperator.transform_edge_to_tif(np.array(edge), transformer) # todo: TIFOperator
                transformed_pts.append(new_pts)

                all_pts += new_pts.tolist()
            all_pts = np.array(all_pts)

            # Get minimal rectangle box that encapsulates all the annotations
            dlat = np.max(all_pts[:,0]) - np.min(all_pts[:,0])
            dlong = np.max(all_pts[:,1]) - np.min(all_pts[:,1])
            bounds = np.array([np.min(all_pts[:,0])-dlat*extend_factor, np.min(all_pts[:,1])-dlong*extend_factor, np.max(all_pts[:,0])+dlat*extend_factor, np.max(all_pts[:,1])+dlong*extend_factor])
            # bbox = np.array([bounds[3], bounds[1], bounds[2], bounds[0]])


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
                            default_lanes: int = 1, trim_dist: float = 2.5, max_part_trim: float = 0.4) -> dict:
        """
        This function is the main function of using a graph from OSM
        to generate the data required to form road masks
        We generate a dictionary with the keys:
        - 'intersections': containing information of the which lanes intersect
        - 'original_edges': containing all the relevant info off the lanes
        - 'connectors': provides the actual connections made between the lanes at the intersections 
        """
        network_data = {
            "original_edges": {},
            "intersections": {},
            "connectors": {},
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

            transformed_geometry_trimmed = trim_edge(transformed_geometry, trim_dist=trim_dist, max_part_trim=max_part_trim)
            if transformed_geometry is None:
                continue  

            length = float(transformed_geometry.length)
            width = data.get("width", default_width)
            width = float(width[0]) if isinstance(width, list) else float(width)
            lanes = data.get("lanes", default_lanes)
            lanes = int(lanes[0]) if isinstance(lanes, list) else int(lanes)
            width, lanes = lanes * width, 1
            

            lane_key = f"{edge_step}"  # Unique key for each lane
            lane_group = edge_step # Assign the lane group to the original edge ID

            network_data["original_edges"][lane_key] = {
                "from_object": from_obj,
                "to_object": to_obj,
                "geometry": transformed_geometry_trimmed,
                "original_edge": transformed_geometry,
                "crs": target_crs,
                "width": width,
                "length": length,
                "lanes": lanes,
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
        return network_data
