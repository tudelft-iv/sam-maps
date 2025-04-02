import os
from typing import Optional
import hydra
from omegaconf import OmegaConf

import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, LineString


from sam_maps.models import build_model
from sam_maps.utils.utils import set_seed

# Constants
CYCLE_LANE_ID = 3
RELEVANT_POLYGON_TYPE = "intersection"


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    set_seed(config.seed)
    OmegaConf.set_struct(config, False)
    config["eval"] = True

    # Load and preprocess ground truth data
    gt_lanes, gt_polygons = load_geodataframes(config.gt_datapath)
    gdf_gt = preprocess_ground_truth(gt_lanes, gt_polygons)
    print("GT area:", gdf_gt.geometry.area.sum())

    # Load and merge prediction data
    pred_lanes, pred_polygons = load_geodataframes(config.pred_datapath)
    gdf_pred = merge_geometries(pred_lanes, pred_polygons)
    print("Pred area:", gdf_pred.geometry.area.sum())

    # Ensure coordinate reference systems match
    # gdf_gt = gdf_gt.to_crs(gdf_pred.crs)

    BUFFER = 2
    mask = gdf_gt.geometry.union_all().buffer(BUFFER)

    # Compute and print coverage metrics
    metrics = compute_coverage_metrics(gdf_gt, gdf_pred, mask)
    print(
        f"IoU: {metrics['iou']:.3f}\nRecall: {metrics['recall']:.3f}\nPrecision: {metrics['precision']:.3f}"
    )
    if "buffer_covered" in metrics:
        print(f"Buffer covered: {100*metrics['buffer_covered']:.01f}%")


def merge_geometries(
    lanes: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Merge lane and polygon geometries, preserving all geometry types."""
    return gpd.overlay(lanes, polygons, how="union", keep_geom_type=True)


def compute_coverage_metrics(
    gpd_A: gpd.GeoDataFrame,
    gpd_B: gpd.GeoDataFrame,
    mask: Optional[gpd.GeoDataFrame] = None,
) -> dict:
    """Calculate IoU, precision, and recall for two geometries."""
    results = {}

    geom_A = gpd_A.geometry.union_all()
    geom_B = gpd_B.geometry.union_all()

    if mask is not None:
        buffer = mask.difference(geom_A)
        geom_B_buffer = geom_B.intersection(buffer)
        buffer_covered = geom_B_buffer.area / buffer.area
        results["buffer_covered"] = buffer_covered

        geom_B = geom_B.intersection(mask)

    intersection = geom_A.intersection(geom_B)
    union = geom_A.union(geom_B)

    intersection_area = intersection.area
    union_area = union.area

    recall = intersection_area / geom_A.area
    precision = intersection_area / geom_B.area
    iou = intersection_area / union_area if union_area else 0

    results["iou"] = iou
    results["recall"] = recall
    results["precision"] = precision

    return results


def load_geodataframes(folder: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load lane and polygon GeoDataFrames from a specified folder."""
    lanes = gpd.read_file(os.path.join(folder, "lanes.gpkg"))
    polygons = gpd.read_file(os.path.join(folder, "polygons.gpkg"))
    return lanes, polygons


def preprocess_ground_truth(
    lanes: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Filter ground truth data by polygon type and exclude cycle lanes."""
    polygons = polygons[polygons["type"] == RELEVANT_POLYGON_TYPE]
    lanes = lanes[lanes["road_type"] != CYCLE_LANE_ID]
    return merge_geometries(lanes, polygons)


if __name__ == "__main__":
    main()
