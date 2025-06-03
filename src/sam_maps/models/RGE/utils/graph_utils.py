import numpy as np
from shapely.geometry import LineString

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


