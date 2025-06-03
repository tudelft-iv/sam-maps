import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import split
import yaml
from addict import Dict

def get_distance_coordinates(coord1: np.ndarray, coord2: np.ndarray) -> float:
    """
    Get the distance in meters between two coordinates
    """
    R = 6371000.0
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.power(np.sin(delta_phi / 2.0), 2.0) + np.cos(phi1) * np.cos(phi2) * np.power(np.sin(delta_lambda / 2.0), 2.0)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance
    

def get_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Get Intersection over Union
    """
    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))
    return intersection / union
    
def find_intersection(l1_coords: np.ndarray, l2_coords: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Find the intersection between two lines defined each by two coordinates
    And determine if the intersection finds place between the domain of the coordinates or outside it
    """
    p1, p2 = l1_coords
    p3, p4 = l2_coords

    d1 = p2 - p1
    d2 = p4 - p3

    A = np.array([d1, -d2]).T
    b = p3 - p1

    if np.linalg.det(A) == 0:
        return None, False

    t, u = np.linalg.solve(A, b)

    intersection_within_segments = (0 <= t <= 1) and (0 <= u <= 1)

    intersection_point = np.array([p1 + t * d1])
    
    return intersection_point, intersection_within_segments
    

def get_angle(cls, vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Get the angle between two vectors
    """
    return np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))


def get_relevant_edges(edges: list, bounds: np.ndarray) -> list:
            """
            Get all edges that feature with a part of the edge inside given bounds
            """
            relevant_edges = []
            for edge in edges:
                    edge = np.array(edge)
                    left_bottom_check = (edge - bounds[:2])
                    right_top_check = (edge - bounds[2:])
                    left_bottom_bool = ((left_bottom_check[:, 0] > 0) * (left_bottom_check[:, 1] > 0))
                    right_top_bool = ((right_top_check[:, 0] < 0) * (right_top_check[:, 1] < 0))
                    mask_edge = left_bottom_bool * right_top_bool
                    if np.any(mask_edge):
                            relevant_edges.append(edge)
            return relevant_edges

def split_linestring(line: LineString, n_parts: int) -> tuple[list[LineString], float]:
    
    if n_parts <= 0:
        raise ValueError("n_parts must be a positive integer.")
    if n_parts == 1:
        return [line], line.length

    segment_length = line.length / n_parts
    segments = []
    line_arr = np.array(line.coords)
    line_lst = list(line.coords)
    cum_dist = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(line_arr,axis=0), axis=1))))

    new_pts_dist = np.arange(1, n_parts+1) * segment_length
    sps = []
    for dist in new_pts_dist:
        sps.append(np.argwhere(dist >= cum_dist).flatten()[-1])
    sps[-1] = len(line_arr) - 1
    ls_lst_prev = []
    segments = []
    for i, sp in enumerate(sps):
        if i !=0:
            ls_lst = [ls_lst_prev[-1]]
            start_idx = sps[i-1]+1
        else:
            ls_lst = [line_lst[0]]
            start_idx = 1
        for j in range(start_idx, min(sp+1, len(line_arr) - 1)):
            ls_lst.append(line_lst[j])
        if i!=len(sps)-1: 
            final_point = (new_pts_dist[i] - cum_dist[sp]) / (cum_dist[sp+1] - cum_dist[sp]) * (line_arr[sp+1] - line_arr[sp]) + line_arr[sp]
        else:
            final_point = line_arr[-1]
        ls_lst.append(final_point.tolist())
        ls_lst_prev = ls_lst
        segments.append(LineString(ls_lst))

    return segments, segment_length


from shapely.geometry import LineString, Point

def point_side_shapely(line_points, test_point):
    """
    Determine whether a point is to the left, right, or on a series of line segments using Shapely.

    Args:
        line_points (list of tuples): List of (x, y) points defining the line.
        test_point (tuple): The (x, y) point to compare.

    Returns:
        list of int: For each line segment:
                     -1 for right of the line,
                      1 for left of the line,
                      0 for on the line.
    """
    line = LineString(line_points)  # Create a Shapely LineString
    point = Point(test_point)       # Create a Shapely Point
    sides = []
    
    for i in range(len(line_points) - 1):
        segment = LineString(line_points[i:i+2])  # Create each line segment
        dx, dy = segment.coords[1][0] - segment.coords[0][0], segment.coords[1][1] - segment.coords[0][1]
        px, py = test_point[0] - segment.coords[0][0], test_point[1] - segment.coords[0][1]
        cross = dx * py - dy * px  # Cross product
        
        if cross > 0:
            sides.append(1)  # Left
        elif cross < 0:
            sides.append(-1)  # Right
        else:
            sides.append(0)  # On the line

    return sides

def multi_line_string_2_line_string(multi_line_string: LineString | MultiLineString) -> LineString:
    """
    Combine MultiLineString to a single LineString
    """
    if isinstance(multi_line_string, LineString):
        return multi_line_string
    elif isinstance(multi_line_string, MultiLineString):
        combined_coords = []
        
        for line in multi_line_string.geoms:
            combined_coords.extend(line.coords)

        combined_line_string = LineString(combined_coords)
        
        return combined_line_string
    else:
        raise TypeError("Wrong type of input")


def load_config(path):
    with open(path) as file:
        config_dict = yaml.safe_load(file)
    return Dict(config_dict)