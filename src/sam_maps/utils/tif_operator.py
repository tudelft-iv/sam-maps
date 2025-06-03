import os
import numpy as np
import tifffile as tifi
from pyproj import CRS, Transformer, transform

from samgeo import SamGeo, tms_to_geotiff, get_basemaps
import cv2 as cv
import geopandas as gpd
import pandas as pd
import fiona
from typing import Optional
from shapely.geometry import Polygon, mapping, MultiPolygon, LineString
import osmnx as ox
from json import JSONDecodeError
from osmnx._errors import InsufficientResponseError
import rasterio
from fiona.crs import from_epsg
from shapely.ops import unary_union


class TIFOperator:
    def __init__(self):
        """
        This class gives methods to transform data between pixel-level and different CRS
        It also incorporates transferring data to a geopackage
        
        """
        pass
    

    @classmethod
    def tif2img(cls, tif_path: str) -> np.ndarray:
        """
        Light-weight method to obtain image from large tif-files
        """
        return tifi.imread(tif_path)
    
    
    @classmethod
    def geo_to_pixel(cls, point: np.ndarray, transform: Transformer) -> tuple[int, int]:
        """
        Go to pixels from a point
        """
        lon = point[0]
        lat = point[1]
        x, y = ~transform * (lon, lat)
        return int(x), int(y)
    
    @classmethod
    def pixel_to_geo(cls, point: np.ndarray, transform: Transformer) -> tuple[float, float]:
        """
        Go to pixels from a point
        """
        x = point[0]
        y = point[1]
        lon, lat = transform * (x, y)
        return lon, lat
    
    @classmethod
    def transform_polygon(cls, geometry: np.ndarray, proj_orig, proj_goal) -> np.ndarray:
        """
        Input the Projection of the original CRS and the goal CRS to get the geometry in the 
        coordinate frame of the goal CRS. The geometry should be passed as an array
        """
        # geometry_lon_lat = np.zeros_like(geometry)
        geometry_lon_lat = []
        for i, (x, y) in enumerate(geometry):
            lon, lat = transform(proj_orig, proj_goal, x, y)
            geometry_lon_lat.append((lon, lat))
        return geometry_lon_lat
    
    @classmethod
    def transform_edge_to_tif(cls, edge: np.ndarray, transformer: Transformer) -> np.ndarray:
        """
        Transform an edge from OSM to the TIF
        """
        longs = edge[:, 0]
        lats = edge[:, 1]
        transformed_edge = np.array([transformer.transform(lon, lat) for lon, lat in zip(longs, lats)])
        return transformed_edge
    

    @classmethod
    def transform_edge_to_pixels(cls, edge: np.ndarray, transformer_tif: Transformer, transformer_pix: Transformer) -> np.ndarray:
        """
        Transform the edge from OSM to the pixels of the corresponding image of the tif
        """
        longs = edge[:, 0]
        lats = edge[:, 1]
        transformed_edge = np.array([cls.geo_to_pixel(transformer_tif.transform(lon, lat), transformer_pix) for lon, lat in zip(longs, lats)])
        return transformed_edge
    
    @classmethod
    def transform_pixels_to_edge(cls, edge: np.ndarray, transformer_pix: Transformer, transformer_tif: Transformer) -> np.ndarray:
        """
        Transform the edge from pixels to the OSM coordinates (lon, lat).

        :param edge: An array of points in pixel coordinates, shaped as (N, 2).
        :param transformer_pix: Affine transformer to convert pixels to geographic coordinates.
        :param transformer_tif: Transformer to convert geographic coordinates to OSM lon/lat.
        :return: Transformed edge as an array of OSM coordinates, shaped as (N, 2).
        """
        # Convert pixels to geographic (CRS of the TIFF)
        geographic_coords = np.array([transformer_pix * (x, y) for x, y in edge])
        # Convert geographic to OSM (WGS84 lon/lat)
        transformed_edge = np.array([
            transformer_tif.transform(*coord, direction="INVERSE")
            for coord in geographic_coords
        ])
        
        return transformed_edge

    @classmethod
    def get_transforms(cls, tif_path: str) -> tuple[Transformer, Transformer, str]:
        """
        Get the transforms for within the tif and to go to pixels
        """
        with rasterio.open(tif_path) as src:
                transformer_pix = src.transform
                crs = src.crs
        wgs84_crs = CRS.from_epsg(4326)
        transformer_tif = Transformer.from_crs(wgs84_crs, crs, always_xy=True)
        return transformer_tif, transformer_pix, crs
    
    @classmethod
    def transform_coords(cls, line: LineString, from_crs: str = "EPSG:4326", to_crs: str = "EPSG:32631") -> LineString:
        gdf_line = gpd.GeoDataFrame(geometry=[line], crs=from_crs)
        gdf_line = gdf_line.to_crs(to_crs)
        line_transformed = gdf_line.geometry.iloc[0]
        return line_transformed


    @classmethod
    def get_bounds_tif(cls, tif_path: str) -> np.ndarray:
        """
        the bounds can be extracted as followed:
        - bounds.left
        - bounds.right
        - bounds.bottom
        - bounds.top
        """
        # TODO: this is hardcoded
        with rasterio.open(tif_path) as src:
                bounds = src.bounds
                crs = src.crs
        wgs84_crs = CRS.from_epsg(4326)
        transformer_tif = Transformer.from_crs(crs, wgs84_crs, always_xy=True)
        bottom_left = transformer_tif.transform(bounds.left, bounds.bottom)
        top_right = transformer_tif.transform(bounds.right, bounds.top)
        bounds_arr = np.array([bottom_left[0], bottom_left[1], top_right[0], top_right[1]])
        return bounds_arr
    

    @classmethod
    def get_tif_from_edge(cls, edge: np.ndarray, id: str, output_dir: str, zoom: int = 21, offset: float = 0.2, overwrite=True) -> str:
        """
        Create a tif file for the edge, the offset gives some extra space, to not miss the outer parts of the road
        """

        dst_name = os.path.join(output_dir, f"satellite_road{id}.tif")
        bbox = cls.get_bounds_from_edge(edge, offset=offset)

        tms_to_geotiff(output=dst_name, bbox=bbox, zoom=zoom, source="Satellite", overwrite=overwrite, quiet=True)
        return dst_name, bbox
    

    @classmethod
    def get_bounds_from_edge(cls, edge: np.ndarray, offset: float = 0.2):
        """
        Obtain the bounds that should be used to create an image of a single edge that can then be warped
        """
        min_long = np.min(edge[:,0])
        max_long = np.max(edge[:,0])
        min_lat = np.min(edge[:,1])
        max_lat = np.max(edge[:,1])
        diff = np.max([(max_long - min_long), (max_lat - min_lat)])
        bounds = [min_long - diff * offset, min_lat - diff * offset,
                max_long + diff * offset, max_lat + diff * offset]
        return bounds

    
    @classmethod
    def combine_gpkg(cls, input_files: str, output_file: str, output_layer: str = 'combined_layer'):
        """
        Combine the geopackages to one geopackage and save it
        """
        combined_gdf = gpd.GeoDataFrame()

        # Iterate over each input file and layer, concatenating them into one GeoDataFrame
        for file in input_files:
            layers = fiona.listlayers(file)
            for layer in layers:
                gdf = gpd.read_file(file, layer=layer)
                combined_gdf = gpd.GeoDataFrame(pd.concat([combined_gdf, gdf], ignore_index=True), crs=gdf.crs)

        # Drop any rows with empty or invalid geometries
        combined_gdf = combined_gdf.dropna(subset=['geometry'])
        combined_gdf = combined_gdf[combined_gdf.is_valid]
        combined_gdf.to_file(output_file, driver='GPKG', layer=output_layer)


    @classmethod
    def transform_gpkg(cls, input_geopackage_path: str, output_geopackage_path: str, target_crs: str = 'EPSG:32631'): #4326'):
        """
        Transform the geopackage to your target crs
        """
        # Load the input GeoPackage
        gdf = gpd.read_file(input_geopackage_path)

        # Transform the GeoDataFrame to the target CRS
        gdf_transformed = gdf.to_crs(target_crs)

        # Save the transformed GeoDataFrame to a new GeoPackage
        gdf_transformed.to_file(output_geopackage_path, driver='GPKG')

    
    @classmethod
    def polygons2gpkg(cls, polygons: list, id: Optional[int] = "", city: str = "Bratislava", name: Optional[str]=None):
        """
        convert a list of polygons into a geopackage and save it
        """
        schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int'},
        }
        folder = os.path.join("mask_road_tiffs", city, "osm")
        if not os.path.isdir(folder):
            os.makedirs(folder)
        if name is None:
            geopackage_name = os.path.join(folder, f"mask_road{id}.gpkg")
        else:
            geopackage_name = os.path.join(folder, f"{name}.gpkg")
        labs = np.arange(1, len(polygons) + 1)
        with fiona.open(geopackage_name, 'w', driver='GPKG', crs=from_epsg(4326), schema=schema) as layer:
            for polygon, i in zip(polygons, labs):
                if polygon.geom_type == "Polygon":
                    layer.write({
                        'geometry': mapping(polygon),
                        'properties': {'id': 1},
                    })
    
    
    @classmethod
    def poly2list(cls, poly: list) -> list:
        """
        Create a list of coordinates from polygons/multipolygons
        This is used for water and buildings mask creation
        """
        polylist = []
        for geom in poly:
            if isinstance(geom, Polygon):
                # For single Polygon objects
                polylist.append([list(coord) for coord in geom.exterior.coords])
            elif isinstance(geom, MultiPolygon):
                # For MultiPolygon objects, extract each sub-polygon
                for poly in geom:
                    polylist.append([list(coord) for coord in poly.exterior.coords])
        return polylist


    @classmethod
    def get_water_poly(cls, bounds: np.ndarray) -> list:
        """
        Get all the water objects from OSM
        """
        bbox = bounds #np.array([bounds[3], bounds[1], bounds[2], bounds[0]])
        try:
            water = ox.features_from_bbox(bbox=bbox, tags={"natural": "water"})
            if "geometry" in water.keys():
                water_poly = water["geometry"]
            else:
                water_poly = []
        except InsufficientResponseError:
            water_poly = []
        except JSONDecodeError:
            water_poly = []
        return water_poly


    @classmethod
    def get_water_gpkg(cls, bounds: np.ndarray, city: str):
        """
        Get all the water objects from OSM and obtain a geopackage holding all this information
        """
        water_poly = cls.get_water_poly(bounds)
        cls.polygons2gpkg(water_poly, city=city, name="water")


    @classmethod
    def get_water_mask(cls, bounds: np.ndarray, tf_tif: Transformer, tf_pix: Transformer, img_shape: tuple) -> np.ndarray:
        """
        Get all the water from OSM and create a mask for a corresponding image
        """
        water_poly = cls.get_water_poly(bounds)
        for i, poly in enumerate(water_poly):
            if isinstance(poly, MultiPolygon):
                return np.zeros(img_shape[:2], dtype=np.uint8)
        water_list = cls.poly2list(water_poly)
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        for water in water_list:

            water_edge = cls.transform_edge_to_pixels(np.array(water), tf_tif, tf_pix)
            mask = cv.fillPoly(mask, pts=[water_edge], color=255)
        mask = mask.astype(bool)
        return mask


    @classmethod
    def get_buildings_poly(cls, bounds: np.ndarray) -> list:
        """
        get all the buildings from OSM
        """
        bbox = bounds #np.array([bounds[3], bounds[1], bounds[2], bounds[0]])
        try:
            buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})
            if "geometry" in buildings.keys():
                buildings_poly = buildings["geometry"]
            else:
                    buildings_poly = []
        except InsufficientResponseError:
            buildings_poly = []
        except JSONDecodeError:
            buildings_poly = []

        return buildings_poly
    

    @classmethod
    def get_buildings_gpkg(cls, bounds: np.ndarray, city: str):
        """
        Get all the buildings from OSM and obtain a geopackage holding all this information
        """
        buildings_poly = cls.get_buildings_poly(bounds)
        cls.polygons2gpkg(buildings_poly, city=city, name="water")


    @classmethod
    def get_buildings_mask(cls, bounds: np.ndarray, tf_tif: Transformer, tf_pix: Transformer, img_shape: list) -> np.ndarray:
        """
        Get all the buildings from OSM and create a mask for a corresponding image
        """
        buildings_poly = cls.get_buildings_poly(bounds)
        for i, poly in enumerate(buildings_poly):
            if isinstance(poly, MultiPolygon):
                return np.zeros(img_shape[:2], dtype=np.uint8)

        buildings_list = cls.poly2list(buildings_poly)
        bulidings = []
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        for building in buildings_list:
            building_edge = cls.transform_edge_to_pixels(np.array(building), tf_tif, tf_pix)
            mask = cv.fillPoly(mask, pts=[building_edge], color=255)
        mask = mask.astype(bool)
        return mask

