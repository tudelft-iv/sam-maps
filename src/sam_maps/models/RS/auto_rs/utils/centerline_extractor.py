import os 
import numpy as np
import cv2 as cv
import pickle
from typing import Optional
from shapely.geometry import LineString
from addict import Dict

from sklearn.linear_model import LinearRegression
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import UnivariateSpline

from .warp_operator import WarpOperator
from .mask_operator import MaskPostProcessor

class CenterlineExtractor:
    def __init__(self, config: Dict):
        # Load parameters
        self.interpolation_step_size = config.interpolation_step_size
        self.min_smoothness_spline = config.min_smoothness_spline
        self.smoothness_factors_spline = config.smoothness_factors_spline
        self.min_angle_spline = config.min_angle_spline
        self.best_road_score = config.best_road_score
        self.inpaint_radius_trees = config.inpaint_radius_trees
        self.gap_filter_size = config.gap_filter_size
        self.centerline_selection_proximity = config.centerline_selection_proximity
        self.centerline_heatmap_pool_threshold = config.centerline_heatmap_pool_threshold
        self.centerline_pool_size = config.centerline_pool_size
        self.heatmap_pool_size = config.heatmap_pool_size
        self.simplify_tolerance_spline = config.simplify_tolerance_spline
        self.min_road_length_fit = config.min_road_length_fit
        self.min_length_ratio = config.min_length_ratio
    

    def find_best_fitting_centerline(self, combined_lane: dict, width_lane_warp: int) -> dict:

        centerlines, original_mask, heatmap_original, heatmap_smooth = self._get_proposed_centerlines(combined_lane["road"], combined_lane["length"], combined_lane["img"], combined_lane["mask"], combined_lane["tree_mask"])
                                                                                
        results = {}
        for key, val in centerlines.items():
            if key=="road":
                warped_img, _, _, _, _, _, _ = WarpOperator.warp_image_efficient(combined_lane["img"], val, width_pixels=width_lane_warp)
                warped_mask, M_arr, M_inv_arr, shifts, intervals, _, _ = WarpOperator.warp_image_efficient(original_mask.astype(np.uint8), val, width_pixels=width_lane_warp)
                m, center, w, fit_val = MaskPostProcessor.find_best_iou_match_road(warped_mask)
                final_m = WarpOperator.unwarp_mask_efficient(combined_lane["img"], m.astype(np.uint8), intervals, M_inv_arr, shifts, val)
                edge = val
                k_res = f"{key}"
                fit_val = np.max((self.best_road_score,fit_val))
                shift_center = center - m.shape[0]/2
                final_centerline = LineString(edge).parallel_offset(shift_center, side='left')
                final_centerline = final_centerline.simplify(tolerance=self.simplify_tolerance_spline, preserve_topology=False)
                results[k_res] = {"mask": final_m, "line": final_centerline, "score": fit_val, "width": w, "warped_mask": warped_mask, "found_mask": m, "warped_img": warped_img}
            else:
                for key2, val2 in val.items():
                    if key2 == "splines":
                        for key3, val3 in val2.items():
                            try:
                                warped_img, _, _, _, _, _, _ = WarpOperator.warp_image_efficient(combined_lane["img"], val3, width_pixels=width_lane_warp)
                                warped_mask, M_arr, M_inv_arr, shifts, intervals, _, _ = WarpOperator.warp_image_efficient(original_mask.astype(np.uint8), val3, width_pixels=width_lane_warp)
                                m, center, w, fit_val = MaskPostProcessor.find_best_iou_match_road(warped_mask)
                                final_m = WarpOperator.unwarp_mask_efficient(combined_lane["img"], m.astype(np.uint8), intervals, M_inv_arr, shifts, val3)
                                edge = val3
                                k_res = f"{key}_{key2}_{key3}"
                                shift_center = center - m.shape[0]/2
                                final_centerline = LineString(edge).parallel_offset(shift_center, side='left')
                                final_centerline = final_centerline.simplify(tolerance=self.simplify_tolerance_spline, preserve_topology=False)
                                results[k_res] = {"mask": final_m, "line": final_centerline, "score": fit_val, "width": w, "warped_mask": warped_mask, "found_mask": m, "warped_img": warped_img}
                            except Exception as e:
                                print("Warp Failed")
                                continue
                    elif key2 == "linear":
                        if val2 is None or len(val2)<=1:
                            continue
                        warped_img, _, _, _, _, _, _ = WarpOperator.warp_image_efficient(combined_lane["img"], val2, width_pixels=width_lane_warp)

                        warped_mask, M_arr, M_inv_arr, shifts, intervals, _, _ = WarpOperator.warp_image_efficient(original_mask.astype(np.uint8), val2, width_pixels=width_lane_warp)
                        try:
                            m, center, w, fit_val = MaskPostProcessor.find_best_iou_match_road(warped_mask)
                            final_m = WarpOperator.unwarp_mask_efficient(combined_lane["img"], m.astype(np.uint8), intervals, M_inv_arr, shifts, val2)
                        except Exception:
                            print("WARP Failed")
                            continue
                        edge = val2
                        k_res = f"{key}_{key2}"
                        shift_center = center - m.shape[0]/2
                        final_centerline = LineString(edge).parallel_offset(shift_center, side='left')
                        final_centerline = final_centerline.simplify(tolerance=self.simplify_tolerance_spline, preserve_topology=False)
                        results[k_res] = {"mask": final_m, "line": final_centerline, "score": fit_val, "width": w, "warped_mask": warped_mask, "found_mask": m, "warped_img": warped_img}
                    else:
                        raise ValueError("Invalid key name found")
            if ((LineString(edge).length) / (LineString(centerlines["road"]).length)) < self.min_length_ratio:
                continue
            shift_center = center - m.shape[0]/2
            final_centerline = LineString(edge).parallel_offset(shift_center, side='left')
            final_centerline = final_centerline.simplify(tolerance=self.simplify_tolerance_spline, preserve_topology=False)
           
        best = min(results, key=lambda k: results[k]["score"])
        # print("Selected Centerline: ", best, " with score: ", results[best]["score"])
        score = np.round(results[best]["score"],3)
        # print(results.keys())
        return results[best], [res["mask"] for res in results.values()], [res["line"] for res in results.values()], heatmap_original, heatmap_smooth, [res["warped_mask"] for res in results.values()], [res["found_mask"] for res in results.values()], [res["score"] for res in results.values()], [res["warped_img"] for res in results.values()]


    def _get_proposed_centerlines(self, road: np.ndarray, road_length: float, original_img: np.ndarray, original_mask: np.ndarray,
                                  original_tree_mask: np.ndarray) -> dict:
        """
        Generate multiple proposals for the centerline:
        1. The initial proposed road from SAMRoad or OSM
        2. Connected points sampled from the original heatmap
        3. Connected points sampled from the smooth mask
        4. Linear line from the original mask
        5. Linear line from the smooth mask
        6. Spline from the original mask
        7. Spline from the smooth mask
        """   
        if road_length < self.min_road_length_fit:
            centerlines = {"road": road}
            return centerlines, original_mask, None, None     
        # Generate the heatmaps    
        heatmap_original = self._create_heatmap(original_mask)
        smooth_mask = self._generate_smooth_mask(original_mask, original_tree_mask, inpaint_radius=self.inpaint_radius_trees, gap_filter_size=self.gap_filter_size)
        heatmap_smooth = self._create_heatmap(smooth_mask)

        # Interpolate the centerline
        interpolated_road = self._equally_spaced_interpolation(road, step_size=self.interpolation_step_size) # step size in pixels

        # Select points and fit linear line for original heatmap
        selected_points_original = self._select_centerline_points(interpolated_road, heatmap_original, self.centerline_pool_size, 
                                                                 threshold_factor=self.centerline_heatmap_pool_threshold, prox=self.centerline_selection_proximity)
        best_fit_line_original = self._fit_line(selected_points_original, road)
        if best_fit_line_original is not None:
            best_fit_line_original = self._equally_spaced_interpolation(best_fit_line_original, step_size=self.interpolation_step_size)
        # Select points and fit linear line for smooth heatmap
        selected_points_smooth = self._select_centerline_points(interpolated_road, heatmap_smooth, pool_size=self.centerline_pool_size, 
                                                       threshold_factor=self.centerline_heatmap_pool_threshold, prox=self.centerline_selection_proximity)
        best_fit_line_smooth = self._fit_line(selected_points_smooth, road)
        if best_fit_line_smooth is not None:
            best_fit_line_smooth = self._equally_spaced_interpolation(best_fit_line_smooth, step_size=self.interpolation_step_size)
        
        # Generate a spline for the original heatmap
        res_univariate_splines_original = self._get_univariate_spline(heatmap_original, road, pool_size=self.heatmap_pool_size, 
                                                                     min_smoothness=self.min_smoothness_spline, smoothness_factors=self.smoothness_factors_spline, 
                                                                     simplify_tolerance=self.simplify_tolerance_spline, min_angle=self.min_angle_spline)
        
        # Generate a spline for the smooth heatmap
        res_univariate_splines_smooth = self._get_univariate_spline(heatmap_smooth, road, pool_size=self.heatmap_pool_size, 
                                                                   min_smoothness=self.min_smoothness_spline, smoothness_factors=self.smoothness_factors_spline, 
                                                                   simplify_tolerance=self.simplify_tolerance_spline, min_angle=self.min_angle_spline)

        centerlines = {
            "road": road,
            "original": {
                # "sampled_points": selected_points_original,
                "linear": best_fit_line_original,
                "splines": res_univariate_splines_original
            },
            "smooth": {
                # "sampled_points": selected_points_smooth,
                "linear": best_fit_line_smooth,
                "splines": res_univariate_splines_smooth
            }
        }
        return centerlines, original_mask, heatmap_original, heatmap_smooth

    @classmethod
    def _get_univariate_spline(cls, heatmap: np.ndarray, road: np.ndarray, pool_size: int = 25, 
                               min_smoothness: int = 5000, smoothness_factors: list = [1], 
                               simplify_tolerance: int = 5, min_angle: float = np.pi/2) -> np.ndarray:
        """
        Otain a univariate spline from the heatmap.
        To achieve this a couple of steps are required:
        1. perform max pooling on the heatmap and regularize
        2. perform a transformation on the mask to obtain always increasing values of x
        3. calculate the univariate spline
        4. invert the transformation
        """
        ls = LineString(road)
        length_road = ls.length
        splines = {}
        for smoothness_factor in smoothness_factors:
            smoothness = int(np.max((min_smoothness, smoothness_factor * length_road)))
            T, _ = cls._get_transformation_matrix(road[0], road[-1])
            probs, pts = cls._pooling_heatmap(heatmap, pool_size=pool_size)
            probs_reshape = probs.flatten()

            pts_reshape_orig = pts.reshape(pts.shape[0]*pts.shape[1], pts.shape[2])[probs_reshape>0]
            pts_reshape_orig = np.vstack((pts_reshape_orig[:,1], pts_reshape_orig[:,0])).T

            pts_reshape = np.vstack([pts_reshape_orig[:,0], pts_reshape_orig[:,1], np.ones((pts_reshape_orig.shape[0]))]).T

            pts_reshape = ((T@pts_reshape.T).T)[:,:2]
            probs_reshape = probs_reshape[probs_reshape>0]
            if len(probs_reshape)==0:
                continue
            x = pts_reshape[:,0]
            y = pts_reshape[:,1]
            probabilities = probs_reshape**2/(np.max(probs_reshape**2)) # Squared to force more direct following of the line

            # Sort data by x (important for spline fitting)
            sorted_indices = np.argsort(x)
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]
            weights = probabilities[sorted_indices]

            # Fit a spline (cubic) with a smooth factor
            try:
                spline = UnivariateSpline(x_sorted, y_sorted, w=weights, s=smoothness)

                # Evaluate spline at the x values
                y_fit_spline = spline(x_sorted)    
                pts_spline = np.vstack((x_sorted, y_fit_spline, np.ones((x_sorted.shape[0])))).T
                final_points = (np.linalg.inv(T)@pts_spline.T).T[:,:2]
                ls = LineString(final_points)
                simplified_line = ls.simplify(tolerance=simplify_tolerance, preserve_topology=False)
                
                final_points = np.array(simplified_line.coords)
            except Exception:
                continue
            if cls._validate_spline(final_points, mask_shape=heatmap.shape, min_angle=min_angle):
                splines[f"spline_{smoothness_factor}"] = final_points
        return splines
    

    
    @staticmethod
    def _validate_spline(spline: np.ndarray, mask_shape: tuple, min_angle: float = np.pi/2) -> bool:
        """
        Some splines overfit too much leading too extreme oscilations
        These splines will have many points and cause a long warping procedure
        This step will look at step direction changes in the spline and return whether
        A spline has too large jumps.
        """
        if (np.isnan(spline).any()) or ((spline < 0).any()) or ((spline[:,0] > mask_shape[0]).any()) or ((spline[:,1] > mask_shape[1]).any()):
            return False
        vectors = np.diff(spline, axis=0)
        vectors = vectors[np.linalg.norm(vectors, axis=1)!=0]
        for v1, v2 in zip(vectors[:-1], vectors[1:]):
            u1 = v1 / np.linalg.norm(v1)
            u2 = v2 / np.linalg.norm(v2)
            dot_product = np.clip(np.dot(u1, -u2), -1.0, 1.0)
            angle = np.arccos(dot_product)
            if angle < min_angle:
                return False
        return True

    @staticmethod
    def _create_heatmap(mask: np.ndarray) -> np.ndarray:
        """
        Generate a heatmap of how far from the edge a point in the mask is
        """
        mask = mask.astype(bool)
        mask_extended = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2))
        mask_extended[1:-1, 1:-1] = mask
        distance_inside = distance_transform_edt(mask_extended)
        return distance_inside[1:-1]
    

    @staticmethod
    def _pooling_heatmap(heatmap: np.ndarray, pool_size: int = 20, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The Pooling of the Heatmap reduces the dimensionality of the points in the mask
        """
        tot = np.sum(heatmap)
        xsteps, ysteps = heatmap.shape[0] // pool_size, heatmap.shape[1] // pool_size
        probs = np.zeros((xsteps, ysteps))
        pts = np.zeros((xsteps, ysteps, 2))
        for i in range(xsteps):
            for j in range(ysteps):
                if not np.any(heatmap[i * pool_size:(i + 1) * pool_size, j * pool_size:(j + 1) * pool_size]):
                    probs[i, j] = 0
                    pts[i, j] = np.array([i * pool_size, j * pool_size])
                    continue
                pool = heatmap[i * pool_size:(i + 1) * pool_size, j * pool_size:(j + 1) * pool_size]
                max_2d_index = np.array(np.unravel_index(np.argmax(pool), pool.shape))
                pts[i, j] = max_2d_index + np.array([i * pool_size, j * pool_size])
                probs[i, j] = np.sum(pool) / tot
        return probs, pts
    

    @staticmethod
    def _equally_spaced_interpolation(line: np.ndarray, step_size: float) -> np.ndarray:
        """ 
        Generate an equally spaced interpolated line of the proposed road
        """
        distances = np.cumsum(np.sqrt(np.sum(np.diff(line, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)
        interpolated_distances = np.arange(0, distances[-1], step_size)
        interpolated_x = np.interp(interpolated_distances, distances, line[:, 0])
        interpolated_y = np.interp(interpolated_distances, distances, line[:, 1])
        return np.stack((interpolated_x, interpolated_y), axis=1)


    @staticmethod
    def _select_centerline_points(line: np.ndarray, heatmap: np.ndarray, pool_size: int = 150, threshold_factor: float = 0.5, prox: int = 15) -> np.ndarray:
        """ 
        Loop through an interpolated line of the road, which gives equally spaced points
        And look for the highest value on the heatmap within some radius so that we get points 
        that are on the supposed centerline of the mask.
        """
        selected_points = []
        for x, y in line:
            x_min, x_max = max(0, int(x - pool_size // 2)), min(heatmap.shape[1], int(x + pool_size // 2))
            y_min, y_max = max(0, int(y - pool_size // 2)), min(heatmap.shape[0], int(y + pool_size // 2))
            pool = heatmap[y_min:y_max, x_min:x_max]
            max_val = np.max(pool)
            if max_val > threshold_factor * np.max(heatmap):
                max_loc = np.unravel_index(np.argmax(pool), pool.shape)
                adjusted_x = x_min + max_loc[1]
                adjusted_y = y_min + max_loc[0]
                selected_points.append([adjusted_x, adjusted_y])
        return np.array(selected_points)


    @classmethod
    def _fit_line(cls, points: np.ndarray, road: np.ndarray) -> np.ndarray:
        """
        Fit a straight line from the sampled points from the mask
        """
        if len(points) < 2:
            return None

        T, dist = cls._get_transformation_matrix(road[0], road[-1])
        points_padded = np.vstack((points[:,0], points[:,1], np.ones(points.shape[0])))
        points_transformed = (T@points_padded).T[:,:2]
        linear_model = LinearRegression()
        linear_model.fit(points_transformed[:, 0].reshape(-1, 1), points_transformed[:, 1])
        
        x_values = np.array([0,dist])
        y_values = linear_model.predict(x_values.reshape(-1, 1))
        padded_results = np.vstack((x_values, y_values, np.ones(x_values.shape[0])))
        results = (np.linalg.inv(T)@padded_results).T[:,:2]

        return results


    @staticmethod
    def _generate_smooth_mask(mask: np.ndarray, tree_mask: np.ndarray, inpaint_radius: int = 5, gap_filter_size: int = 100) -> np.ndarray:
        """
        Generate a smooth Mask to obtain better results from the splines,
        when there are many occlusions by trees and cars in the mask.
        """
        smooth_mask = cv.inpaint(mask.astype(np.uint8), tree_mask.astype(np.uint8), inpaintRadius=inpaint_radius, flags=cv.INPAINT_TELEA)
        smooth_mask = cv.dilate(smooth_mask, np.ones((inpaint_radius, inpaint_radius), dtype=np.uint8), iterations=1)
        smooth_mask = cv.erode(smooth_mask, np.ones((inpaint_radius, inpaint_radius), dtype=np.uint8), iterations=1)

        kernel = np.ones((inpaint_radius, inpaint_radius), np.uint8)
        smooth_mask = cv.morphologyEx(smooth_mask, cv.MORPH_CLOSE, kernel)

        big_mask = np.zeros((smooth_mask.shape[0] + gap_filter_size * 2, smooth_mask.shape[1] + gap_filter_size * 2), dtype=np.uint8)
        big_mask[gap_filter_size:-gap_filter_size, gap_filter_size:-gap_filter_size] = smooth_mask.astype(np.uint8)
        smooth_mask = cv.erode(cv.dilate(big_mask.astype(np.uint8), np.ones((gap_filter_size, gap_filter_size), dtype=np.int32)), np.ones((gap_filter_size, gap_filter_size), dtype=np.int32))[gap_filter_size:-gap_filter_size, gap_filter_size:-gap_filter_size]
        return smooth_mask


    @staticmethod
    def _get_transformation_matrix(point1: np.ndarray, point2: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Calculate a Transformation matrix from two points which transforms the two endpoints of the road
        to be straight horizontal line between them. This will allow for an increasing x, allowing standard spline applications.
        """
        dist = np.linalg.norm(point2 - point1)  # Compute the Euclidean distance
        theta = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])  # Angle of rotation
        
        cos_theta = np.cos(-theta)
        sin_theta = np.sin(-theta)
    
        # Transformation matrix
        transformation_matrix = np.array([
            [cos_theta, -sin_theta, -point1[0] * cos_theta + point1[1] * sin_theta],
            [sin_theta, cos_theta, -point1[0] * sin_theta - point1[1] * cos_theta],
            [0, 0, 1]
        ])

        return transformation_matrix, dist
