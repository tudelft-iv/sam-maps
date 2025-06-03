import torch
import numpy as np
import random
import yaml
from addict import Dict
from shapely.geometry import LineString, MultiLineString


def set_seed(seed_value=42):
    """
    Set seed for reproducibility.

    Args:
    seed_value (int): The seed value to be set for random number generators.
    """
    # Set the random seed for PyTorch
    torch.manual_seed(seed_value)

    # If using CUDA (PyTorch with GPU)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU

    # Set the random seed for numpy (if using numpy in the project)
    np.random.seed(seed_value)

    # Set the random seed for Python's `random`
    random.seed(seed_value)

# def load_config(config):
#     """
#     Load the configuration from a YAML file.

#     Args:
#     config (str): Path to the YAML configuration file.

#     Returns:
#     dict: Loaded configuration as addict.Dict object.
#     """
#     with open(config, 'r') as file:
#         cfg = yaml.safe_load(file)
#     return Dict(cfg)

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
