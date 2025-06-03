import os
from osgeo import gdal, osr
from typing import Optional
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, LineString
from typing import Optional
import argparse


from sam_maps.models import build_model
from sam_maps.utils.utils import set_seed

def get_main_config_path():
    for source in HydraConfig.get().runtime.config_sources:
        if source.get("provider") == "main":
            return source.get("path")
    return None  # fallback if not found

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    set_seed(config.seed)
    OmegaConf.set_struct(config, False)  # Open the struct

    config = OmegaConf.merge(config, config.rge_module)
    config = OmegaConf.merge(config, config.rs_module)
    config = OmegaConf.merge(config, config.rc_module)
    model = build_model(config)
    model(None)

if __name__ == "__main__":
    # Initialize GlobalHydra
    GlobalHydra.instance().clear()
    main()  # Call the main function to run the script
