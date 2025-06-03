import numpy as np
from typing import Optional
from tqdm import tqdm
import shelve
import os
from addict import Dict
from samgeo import SamGeo, tms_to_geotiff, get_basemaps
from samgeo.text_sam import LangSAM
from segment_anything import sam_model_registry, SamPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import torch
import yaml
from pyproj import Proj
import geopandas as gpd
import cv2 as cv
import pickle
from shapely.geometry import Polygon, LineString,  MultiPoint
from pyproj import CRS
from copy import deepcopy
from scipy.spatial import ConvexHull
from addict import Dict
from affine import Affine
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from sam_maps.utils.tif_operator import TIFOperator
from .warp_operator import WarpOperator
from .mask_operator import MaskOperator, MaskPostProcessor, MaskPicker, MaskCombiner
from sam_maps.utils.utils import split_linestring, multi_line_string_2_line_string
from .centerline_extractor import CenterlineExtractor


class RoadGenerator:
    def __init__(self, config: Dict):        
        """
        Base Class for generating road masks
        """
        self.config = config
        self.w_init = config.w_init


    @staticmethod
    def _generate_road_boundary(centerline: LineString, offset: float, side: str, tolerance=0.1) -> LineString:
        linestring = centerline.parallel_offset(offset, side=side)
        linestring = multi_line_string_2_line_string(linestring)
        linestring = linestring.simplify(tolerance=tolerance, preserve_topology=True)
        return linestring
    

        


class GraphRoadGenerator(RoadGenerator):
    def __init__(self, config):
        super().__init__(config=config)
    
    def generate_lanes(self, network_data: dict):
        segmentation_results = {}
        for key, val in network_data["original_edges"].items():
            lane = network_data["original_edges"][key]
            lb = self._generate_road_boundary(lane["geometry"], self.w_init/2, "left")
            rb = self._generate_road_boundary(lane["geometry"], self.w_init/2, "right")
            segmentation_results[key] = {
                                        "combined_lanes":
                                        {
                                            "geometry": lane["geometry"],
                                            "boundary_left": np.array(lb.coords),
                                            "boundary_right": np.array(rb.coords),
                                            "length": lane["geometry"].length,
                                            "width": self.w_init,
                                        }
                                        }
        return segmentation_results
    


class AutomaticRoadMaskGenerator(RoadGenerator):
    def __init__(self, config: Dict, gpu_id: int, process_id: int):
    # def __init__(self, model_type: str = "vit_h", checkpoint: str = "/scratch/hjhboekema/mpvanandel/data/sam_weights/sam_vit_h_4b8939.pth", 
    #              grounding_dino_model_id="IDEA-Research/grounding-dino-base", w_trees: float = 0.0, w_dev: float = 0.0, 
    #              w_spreadx: float = 0.0, w_spready: float = 0.0, w_rfit: float = 1.0, w_overalltrees: float = 0.0, w_houses: float = 0.0):
        """
        This is the proposed method. The pipeline is as follows:
        1. Obtain all the relevant edges (roads), with the amount of lanes and width of the lanes
        2. For each road, create an individual tif
        3. Transform the tif and the edges to pixels
        4. Warping the image to a straight road
            4.1 For each section of the edge (part between two points) calculate a transform to warp the image
            4.2 All these warped sections are cut and placed behind each other, so the goal of the warp is to create a straight road
            4.3 Aplly the transform to the edges
            4.4 We crop the image based on the given width and amount of lanes to only capture the area where the road is, with a bit extra
            (since the lines of OSM are not placed perfectly)
        5. Find the Bounding Box
            5.1 Grounding Dino
                5.1.1 The First approach is to obtain the bounding box with Grounding Dino, by using the prompt: 'individual straight horizontal road'
                5.1.2 The given bounding box is checked whether it makes sense (is centered and covers the whole width)
                5.1.3 The found bounding boxes are slightly adapted to cover the entire width if they did not already
            5.2 Hough Lines
                5.2.1 If Grounding Dino did not yield to a logical bounding box, we can try to make a bounding box by looking for straight lines,
                this is done by first performing an edge detection
                5.2.2 Then we take away the boundary edges (caused by the warps)
                5.2.3 Then we compute Hough Lines
                5.2.4 These Hough Lines are conditioned and clustered to create lines covering the whole width
                5.2.5 Finally Several options of bounding boxes are created from these lines
        6. For each of the found bounding boxes, we use SAM to obtain a mask of the road, which should be the main segment within the bounding box
        7. Select the best bounding box, this method has room for improvement, now it is possible to select a minimal amount of "weight" of the mask 
        and it selects the one that is best centered around the OSM edge.
        8. Perform an inverse transformation (of the warp) on the masks, to retrieve the mask on the original shape of the road
        9. Save the masks to a geopackage

        Current issues and proposed solutions:
        - It sometimes includes houses, water or trees in the masks
        -----> OSM includes a geolocation of houses and water, this can be used to cut that from the masks, the geolocation is not perfect, so keep that into account 
               (don't make it a hard reset).
        -----> Trees we can obtain with LanguageSAM, which is a combination of Grounding Dino and SAM and then we can use this information to not include it in the mask, 
               but also to extrapolate road underneath these trees.
        - Too zoomed in or zoomed out roads can be difficult
        -----> Combine or split up edges, based on the node-id's given in OSM
        - The final masks can include the sidewalk in some cases
        -----> Use the information of an existing sidewalk to cut a part off, this is not very elegant but if it appears to be needed, it could be applied
        - The final mask is sometimes somewhat sparse
        -----> We need to apply some form of postprocessing where we interpolate and extrapolate the mask to form a smooth road
        """

        super().__init__(config=config)
        # config = load_config(config)
        self.process_id = process_id
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        # Load flags
        self.save_images = config.flags.save_images
        self.verbose = config.flags.verbose
        self.overwrite = config.flags.overwrite_tif_files
        # Initialize SAM
        if self.verbose:
            print("INITIALIZING SAM")
        model_type = config.rs.sam.model_type
        checkpoint = config.rs.sam.checkpoint
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(self.device)
        self.sam_predictor = SamPredictor(sam)

        if self.verbose:
            print("FINISHED INITIALIZING SAM")
            
            
        # Initialize Grounding Dino
        if self.verbose:
            print("INITIALIZING GROUNDING DINO")
        grounding_dino_model_id = config.rs.grounding_dino.model_id
        # self.gd_processor = AutoProcessor.from_pretrained(grounding_dino_model_id)
        # self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_dino_model_id).to(self.device)

        self.gd_text_prompt_road = config.rs.grounding_dino.roads.text_prompt
        self.gd_text_prompt_trees = config.rs.grounding_dino.trees.text_prompt

        self.gd_box_threshold_road = config.rs.grounding_dino.roads.box_threshold
        self.gd_text_threshold_road = config.rs.grounding_dino.roads.text_threshold
        self.gd_box_threshold_trees = config.rs.grounding_dino.trees.box_threshold
        self.gd_text_threshold_trees = config.rs.grounding_dino.trees.text_threshold
        self.lang_sam = LangSAM(gpu_id=gpu_id)
        if self.verbose:
            print("FINISHED INITIALIZING GROUNDING DINO")
        
        # Load other parameters
        self.zoom_satellite = config.rs.lane_generation.zoom
        self.safety_margin_width_warped_road = config.rs.warp_operator.safety_margin_width
        self.warp_crop_margin = config.rs.warp_operator.crop_margin
        self.tree_enlargement_factor = config.rs.lane_generation.tree_mask_size_factor
        self.max_len_road_section = config.rs.lane_generation.max_road_len
        constraints = config.rs.mask_operator.constraints
        self.mask_constraints = {"tree": constraints.tree, "house": constraints.house, 
                                 "water": constraints.water, "width": constraints.min_width}
        self.from_crs = config.from_crs
        self.target_crs = config.target_crs
        
        # Centerline Extraction
        centerline_params = config.rs.centerline_extractor
        self.centerline_extractor= CenterlineExtractor(centerline_params)
        self.main_data_folder = config.pred_datapath
        self.save_location_name = config.experiment_name


    def generate_mask_lane(self, lane: dict, lane_key: str, save_location_name: str, overwrite: bool = True) -> dict:
        # Extract the Lane Dictionary
        lane_count = lane["lanes"]
        width_lane = lane["width"]
        #split up the road in equal parts avoiding too long roads which are difficult to segment properly
        n_roads = int(lane["geometry"].length // self.max_len_road_section + 1)
        lane_sections, length = split_linestring(lane["geometry"], n_roads)
        segmentation_results = []
        for i, lane_section in enumerate(lane_sections):
            lane_section_id = f"{lane_key}_{i}"
            gdf = gpd.GeoDataFrame(geometry=[lane_section], crs=lane["crs"])
            gdf_world_coords = gdf.to_crs(epsg=4326)
            linestring_world_coords = gdf_world_coords.geometry.iloc[0]
            lane_section_coords = np.array(linestring_world_coords.coords)
            out_dir_satellite = os.path.join(self.main_data_folder, "satellite_roads", save_location_name)
            if not os.path.isdir(out_dir_satellite):
                os.makedirs(out_dir_satellite)
            tif_name, bounds_tif = TIFOperator.get_tif_from_edge(lane_section_coords, id=lane_section_id, output_dir=out_dir_satellite, 
                                                                 zoom=self.zoom_satellite, overwrite=overwrite)
            
            img = TIFOperator.tif2img(tif_name)
            # WARP ABLATION RESULTS
            # bboxes_no_warp = np.array([[0,0,img.shape[0], img.shape[1]]])
            # self.sam_predictor.set_image(img)
            # masks_per_box_no_warp = np.array(self.find_all_masks(bboxes_no_warp))
            # mask_no_warp = masks_per_box_no_warp[0]
            # with rasterio.open(tif_name) as src:
            #     transform = src.transform
            #     crs = src.crs
            #     height, width = src.height, src.width

               
            # # Step 3: Convert mask to polygons
            # shapes_generator = shapes(mask_no_warp.astype('uint8'), transform=transform)

            # # Create a list of geometries and values
            # warp_ablation_geometries = []
            # for geom, value in shapes_generator:
            #     if value:  # Only include shapes for the mask (value > 0)
            #         warp_ablation_geometries.append(shape(geom))
            
            tf_tif, tf_pix, crs = TIFOperator.get_transforms(tif_name)
            road = TIFOperator.transform_edge_to_pixels(lane_section_coords, tf_tif, tf_pix)
            warped_img, M_arr, M_inv_arr, warp_shifts, intervals, width_pixels_warp, resolution = WarpOperator.warp_image_efficient(img, road, tif_name, width_lane=width_lane, lanes=lane_count, 
                                                                                                                        safety_margin_width=self.safety_margin_width_warped_road, crop_margin=self.warp_crop_margin)
            if self.save_images:
                warped_img_dir = os.path.join(self.main_data_folder,"warped_images", save_location_name)
                if not os.path.isdir(warped_img_dir):
                    os.makedirs(warped_img_dir)
                cv.imwrite(os.path.join(dir, f"warped_img{lane_section_id}.png"), warped_img)

            warped_img_path_temp = f"warped_img_temp_{self.process_id}.png"
            cv.imwrite(warped_img_path_temp, warped_img)

            house_mask = TIFOperator.get_buildings_mask(bounds_tif, tf_tif, tf_pix, img.shape)

            warped_house_mask, _, _, _, _, _, _ = WarpOperator.warp_image_efficient(house_mask.astype(np.uint8), road, width_lane=width_lane, lanes=lane_count, 
                                                                                 safety_margin_width=self.safety_margin_width_warped_road, 
                                                                                 crop_margin=self.warp_crop_margin, width_pixels=width_pixels_warp)
                        
           
            water_mask = TIFOperator.get_water_mask(bounds_tif, tf_tif, tf_pix, img.shape)
            warped_water_mask, _, _, _, _, _, _ = WarpOperator.warp_image_efficient(water_mask.astype(np.uint8), road, width_lane=width_lane, lanes=lane_count,
                                                                                 safety_margin_width=self.safety_margin_width_warped_road, 
                                                                                 crop_margin=self.warp_crop_margin, width_pixels=width_pixels_warp)

            
            # Get Tree Mask
            warped_tree_img, _, _, _, _, _, _ = WarpOperator.warp_image_efficient(img, road, width_lane=width_lane, lanes=lane_count,
                                                                               safety_margin_width=self.safety_margin_width_warped_road*self.tree_enlargement_factor, 
                                                                               crop_margin=self.warp_crop_margin, width_pixels=width_pixels_warp)

            if  False: #warped_tree_img.shape[0] != self.tree_enlargement_factor * warped_img.shape[0]: #TODO: look here for debugging
                # warped_tree_mask = np.zeros(warped_img.shape[:2], dtype=bool)
                # print("WTF HAPPENED TO THE TREEES")
                pass
            else:
                temp_tree_img_file = f"tree_mask_temp_{self.process_id}.png"
                cv.imwrite(temp_tree_img_file, warped_tree_img)
                try:
                    tree_mask = MaskOperator.get_tree_mask(temp_tree_img_file, self.lang_sam)
                    warped_tree_mask = tree_mask[int(tree_mask.shape[0]//(self.tree_enlargement_factor*2)):int(tree_mask.shape[0]//(self.tree_enlargement_factor*2)+warped_img.shape[0])]
                except Exception:
                    print("failed to find tree mask")
                    warped_tree_mask = np.zeros(warped_img.shape[:2], dtype=bool)
                # os.unlink(temp_tree_img_file)
               
            if self.save_images:            
                non_road_masks_dir = os.path.join(self.main_data_folder, "masks", save_location_name, "non_road_masks")
                if not os.path.isdir(non_road_masks_dir):
                    os.makedirs(non_road_masks_dir)
                np.save(os.path.join(non_road_masks_dir, f"buildings_mask{lane_section_id}.npy"), warped_house_mask.astype(bool))
                np.save(os.path.join(non_road_masks_dir, f"water_mask{lane_section_id}.npy"), warped_water_mask.astype(bool))
                np.save(os.path.join(non_road_masks_dir, f"tree_mask{lane_section_id}.npy"), warped_tree_mask.astype(bool))
            
            if self.verbose:
                print("Start setting image sam")
            self.sam_predictor.set_image(warped_img)
            
            if self.verbose:
                print("SAM IMAGE SET")
            
            bboxes = MaskOperator.find_bboxes(warped_img, self.gd_text_prompt_road, warped_img_path_temp, self.lang_sam)
            # bboxes = [np.array([0,0,warped_img.shape[0], warped_img.shape[1]])] # Bboxes ablation
            os.unlink(warped_img_path_temp)
            if self.verbose:
                print("FIND BEST MASK")

            road_mask_dir = os.path.join(self.main_data_folder, "masks", save_location_name, "road_masks", lane_section_id)
            if self.save_images:            
                if not os.path.isdir(road_mask_dir):
                    os.makedirs(road_mask_dir)
            
            masks_per_box = np.array(self.find_all_masks(bboxes, save_masks=self.save_images, output_path=road_mask_dir))

            result = MaskPicker.select_best_mask(masks_per_box, house_mask=warped_house_mask, tree_mask=warped_tree_mask, 
                                                  water_mask=warped_water_mask, constraints=self.mask_constraints, resolution=resolution, verbose=self.verbose)

            original_tree_mask = WarpOperator.unwarp_mask_efficient(img, warped_tree_mask, intervals, M_inv_arr, warp_shifts, road) 

            if result["mask"] is None:
                original_mask = None
            else:
                original_mask = WarpOperator.unwarp_mask_efficient(img, result["mask"], intervals, M_inv_arr, warp_shifts, road)

            segmentation_result = {
                "geometry": lane_section,
                "geometry_coords": lane_section_coords,
                "road": road,
                "warped_img": warped_img,
                "transform": M_arr,
                "transform_inv": M_inv_arr,
                "intervals": intervals,
                "original_img": img,
                "tf_tif": tf_tif,
                "tf_pix": tf_pix,
                "crs": crs,
                "tree_mask": warped_tree_mask,
                "water_mask": warped_water_mask,
                "building_mask": warped_house_mask,
                "warp_shifts": warp_shifts,
                "bboxes": bboxes,
                "final_warped_mask": result["mask"],
                "all_masks": masks_per_box,
                "mask_score": result["score"],
                "best_width_fit": result["width"],
                "best_center": result["center"],
                "original_tree_mask": original_tree_mask,
                "original_mask": original_mask,
                # "warp_ablation": warp_ablation_geometries
            }
            segmentation_results.append(segmentation_result)

        return segmentation_results

    def combine_segmented_lane_sections(self, segmentation_results: list, lane_key: str, lane_original: dict):
        scores = [res["mask_score"] for res in segmentation_results]
        
        combined_lane = self._combine_lanes(segmentation_results)
        combined_img = combined_lane["img"] # None

        if not (None in scores):
            lane_sections = segmentation_results
            width_lane_warp = np.max([l["best_width_fit"] for l in lane_sections])
            centerline_results, all_final_masks, all_lines, heatmap_original, heatmap_smooth, all_warped_masks, all_found_masks, all_scores, all_warped_imgs = self.centerline_extractor.find_best_fitting_centerline(combined_lane, width_lane_warp)
            width_lane_warp = centerline_results["width"]
            centerline_results["line"] = multi_line_string_2_line_string(centerline_results["line"])
            centerline_results["line"] = centerline_results["line"].simplify(tolerance=0.1, preserve_topology=True)
            
            centerline = LineString(centerline_results["line"])

            # Transform the boundaries and centerline to the target_crs
            geom = TIFOperator.transform_pixels_to_edge(np.array(centerline_results["line"].coords), combined_lane["tf_pix"], combined_lane["tf_tif"])
            geom = TIFOperator.transform_coords(LineString(geom), from_crs=self.from_crs, to_crs=self.target_crs)
            
            
        else:
            centerline = LineString(combined_lane["road"])
            geom = TIFOperator.transform_pixels_to_edge(np.array(centerline.coords), combined_lane["tf_pix"], combined_lane["tf_tif"])
            geom = TIFOperator.transform_coords(LineString(geom), from_crs=self.from_crs, to_crs=self.target_crs)
            width_lane_warp = np.abs(self.w_init / combined_lane["tf_pix"].a)
            
            # Debugging data not available
            heatmap_original, heatmap_smooth, all_final_masks, all_warped_masks, all_found_masks, all_lines, all_scores, all_warped_imgs = None, None, None, None, None, None, None, None

        length = geom.length

        lb_pix = self._generate_road_boundary(centerline, width_lane_warp/2, side="left")
        rb_pix = self._generate_road_boundary(centerline, width_lane_warp/2, side="right")
        lb = TIFOperator.transform_pixels_to_edge(np.array(lb_pix.coords), combined_lane["tf_pix"], combined_lane["tf_tif"])
        lb = TIFOperator.transform_coords(LineString(lb), from_crs=self.from_crs, to_crs=self.target_crs)
        lb = np.array(lb.coords)

        rb = TIFOperator.transform_pixels_to_edge(np.array(rb_pix.coords), combined_lane["tf_pix"], combined_lane["tf_tif"])
        rb = TIFOperator.transform_coords(LineString(rb), from_crs=self.from_crs, to_crs=self.target_crs)
        rb = np.array(rb.coords)
            # if self.verbose:
            #     print("USED ORIGINAL ROAD FOR KEY: ", lane_key)
            # geom = lane_original["geometry"]
            # lb = lane_original["boundary_left"][::-1]
            # rb = lane_original["boundary_right"]
            # lb_pix, rb_pix, centerline, width_lane_warp, combined_img, length  = None, None, None, None, None, None
        combined_results = {
            "img": combined_img,
            "geometry": geom,
            "boundary_left": lb,
            "boundary_right": rb,
            "length": length,
            "boundary_left_pixel": lb_pix,
            "boundary_right_pixel": rb_pix,
            "geometry_pixel": centerline,
            "width": width_lane_warp,
            "all_final_masks": all_final_masks,
            "all_lines": all_lines,
            "heatmap_original": heatmap_original,
            "heatmap_smooth": heatmap_smooth,
            "all_warped_masks": all_warped_masks,
            "all_final_warped_masks": all_found_masks,
            "all_scores": all_scores,
            "all_warped_imgs": all_warped_imgs

        }
        return combined_results
        
    def generate_lanes(self, network_data: dict, lane_keys: list,
                       cache_file: str = "cached_results.shlv", use_cache: bool = False, save_cache: bool = True):
        results = {}
        for lane_key in tqdm(lane_keys):
            lane = network_data["original_edges"][lane_key]
            segmentation_results = self.generate_mask_lane(lane, lane_key, save_location_name=self.save_location_name, overwrite=self.overwrite)
            combined_results = self.combine_segmented_lane_sections(segmentation_results, lane_key, lane)
            results[lane_key] = {
                "lanes": segmentation_results,
                "combined_lanes": combined_results
                }
            if save_cache:
                cache = shelve.open(cache_file)
                cache[lane_key] = results[lane_key]
                cache.close()
        print(f"------------ Completed process _{self.process_id} -------------")

        return results
    

    @classmethod
    def finalize_map_generation(cls, network_data: dict, results: dict, save_dir: str):
        # TODO: move to RC
        adapted_network_data = deepcopy(network_data)
        adapted_network_data["original_edges"] = cls._format_lanes(results, network_data)
        adapted_network_data = cls.build_connectors(adapted_network_data)
        adapted_network_data = cls.build_polygons(adapted_network_data)

        cls._create_lanes_gpkg(adapted_network_data["original_edges"], save_dir)
        cls._create_connectors_gpkg(adapted_network_data["connectors"], save_dir)
        cls._create_polygons_gpkg(adapted_network_data["polygons"], save_dir)


    @staticmethod
    def _format_lanes(warped_results: dict, network_data: dict):
        formatted_lanes = deepcopy(network_data["original_edges"])
        for key in formatted_lanes.keys():
            formatted_lanes[key]["geometry"] = multi_line_string_2_line_string(warped_results[key]["combined_lanes"]["geometry"])
            formatted_lanes[key]["boundary_left"] = warped_results[key]["combined_lanes"]["boundary_left"][::-1]
            formatted_lanes[key]["boundary_right"] = warped_results[key]["combined_lanes"]["boundary_right"]
            formatted_lanes[key]["length"] = warped_results[key]["combined_lanes"]["length"]
            # formatted_lanes[key]["width"] = warped_results[key]["combined_lanes"]["width"]
        return formatted_lanes
            
    

    def find_all_masks(self, bboxes: list, save_masks: bool = False, output_path: Optional[str] = None) -> list:
        """
        Find all masks and optionaly visualize the masks for the given bounding boxes
        """
        masks_per_box = []

        for i, bbox in enumerate(bboxes):
            masks, _, _ = MaskOperator.predict_mask_sam_bbox(self.sam_predictor, bbox)
            masks_per_box.append(masks[1]) # This mask always yields the best result
            if save_masks:
                output_file = os.path.join(output_path, f"mask_{i}.npy")
                np.save(output_file, masks[1])

        return masks_per_box



    def add_connections_lanes(warped_results: dict, network_data: dict) -> dict:
        """
        This adds the connections between lanes to the warped_results, such that it is prepared for adding it 
        to the lanes.gpkg
        """
        intersect_counter = 0
        all_intersections = {}
        warped_results["grouped"]["invalid_idcs"] = []
        for key, val in network_data["original_edges"].items():
            road_id = int(key)

            if str(road_id) not in warped_results["grouped"].keys():
                continue
            # print("INTERSECTIONS: ", val)
            # for p in val["intersections"]:
            #     if type(p) == MultiPoint:
            #         for i_t_, p_ in enumerate(p.geoms):
            #             p_prev = np.array(p_.coords)
            connected_idcs = []
            inters_arr = np.empty((0,2))
            for inter_t, idx_inter in zip(val["intersections"], val["idcs"]):
                if type(inter_t) == MultiPoint:
                    inter_t = np.array([geom.coords for geom in inter_t.geoms]).squeeze()
                    connected_idcs += [idx_inter] * len(inter_t)
                else:
                    inter_t = np.array(inter_t.coords)
                    connected_idcs.append(idx_inter)
                inters_arr = np.concatenate((inters_arr, inter_t))
                
            intersections, idx_correspondence = np.unique(inters_arr, axis=0, return_inverse=True)

            intersections = intersections.reshape(intersections.shape[0], 2)
            intersection_keys = []
            all_intersects = np.array(list(all_intersections.values()))
            all_intersect_keys = list(all_intersections.keys())
            for i, inter in enumerate(intersections):
                if len(all_intersects)!=0:
                    intersect_idx = np.argwhere((all_intersects[:,0]==inter[0]) * (all_intersects[:,1]==inter[1]))
                else:
                    intersect_idx = []
                if len(intersect_idx)==0:
                    key_inter = f"i{intersect_counter}"
                    all_intersections[key_inter] = inter
                    intersection_keys.append(key_inter)
                    intersect_counter+=1
                else:
                    key_inter = all_intersect_keys[intersect_idx.flatten()[0]]
                    intersection_keys.append(key_inter)
                

            start_point = warped_results["grouped"][str(road_id)]["lines"]["middle"][0]
            end_point = warped_results["grouped"][str(road_id)]["lines"]["middle"][-1]
            if intersections.shape[0]==1:
                key_inter = f"i{intersect_counter}"
                intersection_keys.append(key_inter)
                intersect_counter+=1
                s_e_check = np.argmin([np.linalg.norm(intersections-start_point, axis=1), np.linalg.norm(intersections-start_point, axis=1)])
                if s_e_check==0:
                    all_intersections[key_inter] = start_point
                    start = 0
                    end = None
                    warped_results["grouped"][str(road_id)]["from_object"] = intersection_keys[1]
                    warped_results["grouped"][str(road_id)]["to_object"] = intersection_keys[0]
                else:
                    all_intersections[key_inter] = end_point
                    start = None
                    end = 0
                    warped_results["grouped"][str(road_id)]["from_object"] = intersection_keys[1]
                    warped_results["grouped"][str(road_id)]["to_object"] = intersection_keys[0]
            elif len(intersections)==0:
                for iter in range(2):
                    key_inter = f"i{intersect_counter}"
                    intersection_keys.append(key_inter)
                    intersect_counter+=1
                warped_results["grouped"][str(road_id)]["from_object"] = intersection_keys[0]
                warped_results["grouped"][str(road_id)]["to_object"] = intersection_keys[1]
            elif intersections.shape[0]==2:
                start = np.argmin(np.linalg.norm(intersections - start_point, axis=1))
                end = (start + 1) % 2
            
                warped_results["grouped"][str(road_id)]["from_object"] = intersection_keys[start]
                warped_results["grouped"][str(road_id)]["to_object"] = intersection_keys[end]
            else:
                warped_results["grouped"]["invalid_idcs"].append(str(road_id))
                continue
                raise ValueError("invalid amount of intersections was found")
            
            warped_results["grouped"][str(road_id)]["predecessors"] = []
            warped_results["grouped"][str(road_id)]["successors"] = []
            for idx, connected_id in zip(idx_correspondence, connected_idcs):
                if str(connected_id) not in network_data["original_edges"].keys():
                    continue

                if idx == start:
                    warped_results["grouped"][str(road_id)]["predecessors"].append(connected_id)
                else:
                    warped_results["grouped"][str(road_id)]["successors"].append(connected_id)
        return warped_results


    @staticmethod
    def generate_geopackage_lanes(warped_group_results: dict, edges: dict) -> dict:
        """
        Generate the dictionary for lanes.gpkg that is formatted like the view-of-delft lanes.gpkg
        """
        geo_dict = {"element_id": [], "road_type": [], "lane_id": [], "from_object": [], "to_object": [], 
                    "predecessors": [], "successors": [], "boundary_left": [], "boundary_right": [], "geometry": []}
        for (k, warped_group_result), edge in zip(warped_group_results.items(), edges["original_edges"].values()):
            if k in warped_group_results["invalid_idcs"] or (k == "invalid_idcs"):
                continue

            edge = np.array(edge["geometry"].coords)
            geo_dict["element_id"].append(int(k))
            geo_dict["lane_id"].append(int(k))
            geo_dict["road_type"].append(1)
            geo_dict["from_object"].append(warped_group_result["from_object"])
            geo_dict["to_object"].append(warped_group_result["to_object"])
            geo_dict["predecessors"].append(repr(warped_group_result["predecessors"])[1:-1])
            geo_dict["successors"].append(repr(warped_group_result["successors"])[1:-1])

            top = "boundary_left"
            bottom = "boundary_right"
            if np.argmin([np.linalg.norm(warped_group_result["lines"]["middle"][0] - edge[0]), np.linalg.norm(warped_group_result["lines"]["middle"][-1] - edge[0])]) == 1:
                top, bottom = bottom, top

            geo_dict[top].append(repr(warped_group_result["lines"]["top"])[1:-1])
            geo_dict[bottom].append(repr(warped_group_result["lines"]["bottom"])[1:-1])
            geo_dict["geometry"].append(Polygon(np.concatenate((warped_group_result["lines"]["top"],warped_group_result["lines"]["bottom"][::-1], [warped_group_result["lines"]["top"][0]]))))

        return geo_dict
    
    @staticmethod
    def generate_geopackage_connectors(warped_results_connectors: dict) -> dict:
        """
        Generate the dictionary for connectors.gpkg that is formatted like the view-of-delft connectors.gpkg
        """
        geo_dict = {"connector_id": [], "source": [], "dest": [], "intersection_id": [], "lane_type": [], "legal": [], "geometry": []}
        for connector_id, connector_info in warped_results_connectors.items():
            for key in geo_dict.keys():
                if key == "connector_id":
                    geo_dict[key].append(connector_id)
                elif key == "geometry":
                    geo_dict[key].append(LineString(np.array([[1,1],[2,2]])))
                else:
                    geo_dict[key].append(connector_info[key])
                
        return geo_dict

        
    @staticmethod
    def _combine_lanes(lanes: list) -> dict:
        """
        Long lanes are split up into multiple lane segments.
        This is better for the segmentation process. This method
        will conbine the lane segments back to one large segment 
        from which the centerline can be extracted
        """
        shifts = []
        shapes = []
        tfs_a, tfs_c, tfs_e, tfs_f = [], [], [], []
        transforms = [lane["tf_pix"] for lane in lanes]
        imgs = [lane["original_img"] for lane in lanes]
        masks = [lane["original_mask"] for lane in lanes]
        tree_masks = [lane["original_tree_mask"] for lane in lanes]
        roads = [lane["road"] for lane in lanes]
        geometries = [lane["geometry"] for lane in lanes]
        tf_tif_base = lanes[0]["tf_tif"]
        combined_geom = np.empty((0,2))
        for geom in geometries:
            combined_geom = np.concatenate((combined_geom, np.array(geom.coords)))
        lane_length = LineString(combined_geom).length
        
        for i, (transform, img) in enumerate(zip(transforms, imgs)):         
            tfs_c.append(transform.c)
            tfs_f.append(transform.f)
            tfs_a.append(transform.a)
            tfs_e.append(transform.e)
            shapes.append([img.shape[0], img.shape[1]])
            if i==0:
                base_transform = transform
                shifts.append([0,0])
                continue
            shift_y = int((transform.c - base_transform.c)/base_transform.a)
            shift_x = int((transform.f - base_transform.f)/base_transform.e)
            shifts.append([shift_x, shift_y])
        
        tf_a = np.mean(tfs_a)
        tf_e = np.mean(tfs_e)
        tf_c = np.min(tfs_c) if tf_a > 0 else np.max(tfs_c)
        tf_f = np.min(tfs_f) if tf_e > 0 else np.max(tfs_f)
        tf_pix_combined = Affine(tf_a, 0, tf_c, 0, tf_e, tf_f)
        
        shapes, shifts = np.array(shapes), np.array(shifts)
        shifts = shifts - np.array([np.min(shifts[:,0]), np.min(shifts[:,1])])
        end_pts = shapes + shifts
        x_shape = np.max(end_pts[:,0])
        y_shape = np.max(end_pts[:,1])

        combined_road = np.empty((0,2))
        for road, shift in zip(roads, shifts):
            new_road = road + np.array([shift[1], shift[0]])
            combined_road = np.concatenate((combined_road, new_road))
        
        combined_img = np.zeros((x_shape, y_shape, 3), dtype=np.uint8)
        combined_mask = np.zeros((x_shape, y_shape))
        combined_tree_mask = np.zeros((x_shape, y_shape))
        for shift, shape, img, mask, tree_mask in zip(shifts, shapes, imgs, masks, tree_masks):
            combined_img[shift[0]:shift[0]+shape[0], shift[1]:shift[1]+shape[1], :] = img
            if mask is not None:
                combined_mask[shift[0]:shift[0]+shape[0], shift[1]:shift[1]+shape[1]] += mask
            if tree_mask is not None:
                combined_tree_mask[shift[0]:shift[0]+shape[0], shift[1]:shift[1]+shape[1]] += tree_mask
            
        combined_mask = combined_mask.astype(bool)
        combined_tree_mask = combined_tree_mask.astype(bool)
        combined_lane = {
            "road": combined_road,
            "img": combined_img,
            "mask": combined_mask,
            "tree_mask": combined_tree_mask,
            "geometry": combined_geom,
            "length": lane_length,
            "tf_pix": tf_pix_combined,
            "tf_tif": tf_tif_base
        }
        return combined_lane

        
