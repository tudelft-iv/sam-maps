"""
Create dataset tables for View-of-Delft Prediction maps
"""

import argparse
import collections
import json
import os
import random
import sys

random.seed(43)
import string

import numpy as np
from map_annotation import Connectors, Lanes, Polygons
from scipy.spatial.transform import Rotation as R

# sys.path.append("..")
# import rosds
# from dsutils import camproj
from pyquaternion import Quaternion
from tqdm import tqdm


def write_table(table, fpath):
    with open(fpath, "w") as f:
        json.dump(table, f, indent=2)


def generate_and_store_token(tokens, length):
    token = generate_token(length)
    tokens.append(token)

    return token, tokens


def generate_token(length=32):
    characters = string.ascii_lowercase + string.digits
    token = "".join(random.choice(characters) for i in range(length))
    return token


def is_token_unique(token, tokens):
    return not set(tokens).intersection(set([token]))


def dict_to_list(dict_):
    return [value for _, value in dict_.items()]


def rotmat_to_quat(rotmat):
    quat_xyzw = R.from_matrix(rotmat).as_quat()
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    return quat_wxyz


def create_unique_token(token_len, tokens):
    token = generate_token(token_len)
    while not is_token_unique(token, tokens):
        token = generate_token(token_len)

    return token


def create_line_dict(coords, tokens, node_lookup, token_len=32):
    line_token, tokens = generate_and_store_token(tokens, token_len)
    node_tokens = [node_lookup[tuple(coord.tolist())] for coord in coords]
    line = {"token": line_token, "node_tokens": node_tokens}
    return line, tokens


def update_canvas_edge(canvas_edge, coord):
    if len(canvas_edge) == 0:
        canvas_edge = [coord[0], coord[0], coord[1], coord[1]]

    # x-coordinate
    if coord[0] < canvas_edge[0]:
        canvas_edge[0] = coord[0]
    elif coord[0] > canvas_edge[1]:
        canvas_edge[1] = coord[0]

    # y-coordinate
    if coord[1] < canvas_edge[2]:
        canvas_edge[2] = coord[1]
    elif coord[1] > canvas_edge[3]:
        canvas_edge[3] = coord[1]

    return canvas_edge

def generate_vod_format_map(input_dir: str, output_dir: str):
    vod_prediction_dir = output_dir
    vod_prediction_maps_dir = os.path.join(vod_prediction_dir, "maps")
    expansion_dir = os.path.join(vod_prediction_maps_dir, "expansion")
    if not os.path.exists(vod_prediction_maps_dir):
        os.mkdir(vod_prediction_maps_dir)
    if not os.path.exists(expansion_dir):
        os.mkdir(expansion_dir)

    # ensure all tokens are unique
    token_len = 32
    tokens = []

    # prepare metadata containers
    canvas_edge = []
    connectivity = {}
    centerline_path = {}
    id_to_token_mapping = {}

    # Load annotations
    print("Loading lanes...")
    lanes_processed = Lanes().load(os.path.join(input_dir, "lanes.gpkg"))
    print(input_dir)

    print("Loading polygons...")
    polygons_processed = Polygons().load(os.path.join(input_dir, "polygons.gpkg"))
    print("Loading connectors...")
    lane_connectors_processed = Connectors().load(
        os.path.join(input_dir, "connectors.gpkg")
    )
    print("Finished loading.")

    # Create tables
    maps_table = {}
    maps_table["version"] = "1.0"

    # Prepare tables
    nodes = {}
    lines = {}
    polygons = {}
    lanes = {}
    lane_connectors = {}
    node_lookup = {}
    road_segments = {}
    ped_crossings = {}
    walkways = {}
    drivable_areas = {}

    drivable_area_token, tokens = generate_and_store_token(tokens, token_len)
    drivable_area = {"token": drivable_area_token, "polygon_tokens": [], "holes": []}

    print("Processing lanes...")
    for lane in tqdm(lanes_processed):
        lane_polygon = lane.polygon

        polygon_token, tokens = generate_and_store_token(tokens, token_len)
        polygon_dict = {"token": polygon_token, "exterior_node_tokens": [], "holes": []}
        for x, y in lane_polygon:
            reverse_key = (x, y)

            if reverse_key not in node_lookup.keys():
                canvas_edge = update_canvas_edge(canvas_edge, (x, y))

                node_token, tokens = generate_and_store_token(tokens, token_len)
                node = {"token": node_token, "x": x, "y": y}
                nodes[node_token] = node

                node_lookup[reverse_key] = node_token
            else:
                node_token = node_lookup[reverse_key]

            polygon_dict["exterior_node_tokens"].append(node_token)
            # TODO manage polygon holes

        drivable_area["polygon_tokens"].append(polygon_token)

        # store the boundaries as lines
        left_nodes = lane.left_boundary.nodes
        left_line, tokens = create_line_dict(left_nodes, tokens, node_lookup, token_len)

        right_nodes = lane.right_boundary.nodes
        right_line, tokens = create_line_dict(
            right_nodes, tokens, node_lookup, token_len
        )

        # Now repeat for the start and end lines
        start_nodes = lane.start_line.nodes
        start_line, tokens = create_line_dict(
            start_nodes, tokens, node_lookup, token_len
        )

        end_nodes = lane.end_line.nodes
        end_line, tokens = create_line_dict(end_nodes, tokens, node_lookup, token_len)

        for line_dict in [left_line, right_line, start_line, end_line]:
            lines[line_dict["token"]] = line_dict

        lane_token, tokens = generate_and_store_token(tokens, token_len)
        lane_dict = {
            "token": lane_token,
            "polygon_token": polygon_token,
            "lane_type": lane.type.value,
            "from_edge_line_token": start_line["token"],
            "to_edge_line_token": end_line["token"],
            "left_lane_divider_segments": [],  # TODO
            "right_lane_divider_segments": [],  # TODO
        }
        print(lane.centerline)
        t, c, k = lane.centerline.get_spline_parameters(s=0)
        centerline_path[lane_token] = (t.tolist(), np.array(c).T.tolist(), k)

        id_to_token_mapping[str(lane.id)] = lane_token

        connectivity[lane_token] = {
            "incoming": [str(id_) for id_ in lane.predecessors],
            "outgoing": [str(id_) for id_ in lane.successors],
        }

        polygons[polygon_token] = polygon_dict
        lanes[lane_token] = lane_dict

    print("Finished processing lanes.")
    print("Processing lane connectors...")
    for connector in tqdm(lane_connectors_processed):
        if not connector.legal:
            continue

        polygon_token, tokens = generate_and_store_token(tokens, token_len)
        polygon_dict = {"token": polygon_token, "exterior_node_tokens": [], "holes": []}
        for x, y in connector.polygon:
            reverse_key = (x, y)

            if reverse_key not in node_lookup.keys():
                canvas_edge = update_canvas_edge(canvas_edge, (x, y))

                node_token, tokens = generate_and_store_token(tokens, token_len)
                node = {"token": node_token, "x": x, "y": y}
                nodes[node_token] = node

                node_lookup[reverse_key] = node_token
            else:
                node_token = node_lookup[reverse_key]

            polygon_dict["exterior_node_tokens"].append(node_token)
            # TODO manage polygon holes

        connector_token, tokens = generate_and_store_token(tokens, token_len)
        connector_dict = {"token": connector_token, "polygon_token": polygon_token}

        t, c, k = connector.centerline.get_spline_parameters()
        centerline_path[connector_token] = (t.tolist(), np.array(c).T.tolist(), k)

        id_to_token_mapping[str(connector.id)] = connector_token
        connectivity[connector_token] = {
            "incoming": [str(id_) for id_ in connector.predecessors],
            "outgoing": [str(id_) for id_ in connector.successors],
        }

        polygons[polygon_token] = polygon_dict
        lane_connectors[connector_token] = connector_dict
    print("Finished processing lane connectors.")

    print("Processing polygons...")
    for polygon_obj in tqdm(polygons_processed):
        # store polygon geometry
        polygon_token, tokens = generate_and_store_token(tokens, token_len)
        polygon_dict = {"token": polygon_token, "exterior_node_tokens": [], "holes": []}
        for x, y in polygon_obj.geometry.exterior.coords:
            reverse_key = (x, y)

            if reverse_key not in node_lookup.keys():
                canvas_edge = update_canvas_edge(canvas_edge, (x, y))

                node_token, tokens = generate_and_store_token(tokens, token_len)
                node = {"token": node_token, "x": x, "y": y}
                nodes[node_token] = node

                node_lookup[reverse_key] = node_token
            else:
                node_token = node_lookup[reverse_key]

            polygon_dict["exterior_node_tokens"].append(node_token)
            # TODO manage polygon holes

        polygons[polygon_token] = polygon_dict

        polygon_type = polygon_obj.type
        if polygon_type == "intersection":
            # add to drivable area
            drivable_area["polygon_tokens"].append(polygon_token)

            # create a 'road_segment'
            road_segment_token, tokens = generate_and_store_token(tokens, token_len)
            road_segment = {
                "token": road_segment_token,
                "polygon_token": polygon_token,
                "is_intersection": True,
                "drivable_area_token": drivable_area_token,
            }
            road_segments[road_segment_token] = road_segment
        elif polygon_type == "crosswalk":
            # create a 'ped_crossing'
            ped_crossing_token, tokens = generate_and_store_token(tokens, token_len)
            ped_crossing = {
                "token": ped_crossing_token,
                "polygon_token": polygon_token,
                "road_segment_token": "",
            }
            ped_crossings[ped_crossing_token] = ped_crossing
        elif polygon_type == "offroad":
            # create a "walkway"
            walkway_token, tokens = generate_and_store_token(tokens, token_len)
            walkway = {
                "token": walkway_token,
                "polygon_token": polygon_token,
            }
            walkways[walkway_token] = walkway

    print("Finished processing polygons.")

    # replace connectivity dict IDs with tokens
    connectivity_new = {}
    for lane_token, lane_connections in connectivity.items():
        lane_connections_new = {}
        for key, lanes_ids in lane_connections.items():
            lane_connections_new[key] = [
                id_to_token_mapping[str(lane_id)] for lane_id in lanes_ids
            ]
        connectivity_new[lane_token] = lane_connections_new

    drivable_areas[drivable_area_token] = drivable_area

    # fill in tables
    maps_table["node"] = dict_to_list(nodes)
    maps_table["line"] = dict_to_list(lines)
    maps_table["polygon"] = dict_to_list(polygons)
    maps_table["lane"] = dict_to_list(lanes)
    maps_table["lane_connector"] = dict_to_list(lane_connectors)
    maps_table["road_segment"] = dict_to_list(road_segments)
    maps_table["ped_crossing"] = dict_to_list(ped_crossings)
    maps_table["walkway"] = dict_to_list(walkways)
    maps_table["drivable_area"] = dict_to_list(drivable_areas)

    # make dummy tables for unannotated fields
    traffic_lights = []
    maps_table["traffic_light"] = traffic_lights

    stop_lines = []
    maps_table["stop_line"] = stop_lines

    road_dividers = []
    maps_table["road_divider"] = road_dividers

    # fill in the metadata
    maps_table["canvas_edge"] = canvas_edge
    maps_table["connectivity"] = connectivity_new
    maps_table["arcline_path_3"] = centerline_path
    write_table(maps_table, os.path.join(expansion_dir, "delft.json"))
