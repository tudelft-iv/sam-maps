import numpy as np
import cv2 as cv
from typing import Optional
from rasterio.features import rasterize
from affine import Affine
from shapely.geometry import LineString, Polygon, Point
import rasterio 

from .utils import find_intersection, get_distance_coordinates

class WarpOperator:
    def __init__(self):
        pass
    
    @classmethod
    def warp_image(cls, image: np.ndarray, edge: np.ndarray, bounds_tif: np.ndarray, safety_factor_length: float = 2.0, 
                   safety_margin_width: float = 1.0, width_lane: float = 3.5, lanes: int = 1, max_img_size: int = 10000, eps=1e-9) -> tuple[np.ndarray, np.ndarray, list, list, list, float, list, int]:
        """
        Warp the image by section. We look at the x coordinates (in pixels) of the edge and by that create edges. 
        We make an initial guess for the width of the road (based on the zoom and lane size, this should be tweaked a bit more).
        And then we grab a point at the expected boundaries of the road and warp that to a road with the same width but then a straight horizontal road.
        We paste all these individual sections into one image with the corresponding length of that particular section 
        and then we crop it to slightly more than just the expected road.
        """
        scale_factor = np.min([1, max_img_size/np.max([image.shape[0], image.shape[1]])]) # Scale in case the image is too large (maximum set to 10000, for reducing computation time)
        img_scaled = cv.resize(image, (0,0), fx=scale_factor, fy=scale_factor)
        img_scaled_shape = np.array(img_scaled.shape[:2])
        width = cls._calculate_width_factor(img_scaled_shape, bounds_tif, width_lane, lanes=lanes) #int(width_factor * scale_factor)
        mask_road, edge_scaled = cls._mask_image_road(image, edge, width=width, scale_factor=scale_factor, safety_factor=safety_factor_length)

        warp_img = np.zeros_like(mask_road)
        warp_img[:img_scaled.shape[0], :img_scaled.shape[1]] = img_scaled
        cum_dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(edge_scaled, axis=0), axis=1))])
    
        offset = width * safety_margin_width / 2

        pts_src = []
        pts_dst = []
        for i, pt in enumerate(edge_scaled[:-1]):
            # Select the corner points as the points that we want to transform in order to create a straight section from it
            pt1 = pt
            pt2 = edge_scaled[i+1]
            rc = (pt2 - pt1)/(np.linalg.norm(pt2-pt1)+eps)
            rc_inv = np.array([-rc[1], rc[0]]) 
            c1_src = pt1 -  (width / 2) * rc_inv
            c2_src = pt1 +  (width / 2) * rc_inv
            c3_src = pt2 -  (width / 2) * rc_inv
            c4_src = pt2 +  (width / 2) * rc_inv
            
            c_src = np.array([c1_src, c2_src, c3_src, c4_src])
            c_dst = np.array([[cum_dist[i], offset - width/2], [cum_dist[i], offset + width/2], 
                            [cum_dist[i+1], offset - width/2], [cum_dist[i+1], offset + width/2]])
            pts_dst.append(c_dst)
            pts_src.append(c_src)

        pts_src = np.array(pts_src)
        pts_dst = np.array(pts_dst)

        imgs = []
        M_arr = []
        M_inv_arr = []
        t = 0
        for src, dst in zip(pts_src, pts_dst):
            # Compute the transformation matrices and their inverse for getting the final mask from the warped mask
            M = cv.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
            M_inv = cv.getPerspectiveTransform(dst.astype(np.float32), src.astype(np.float32))
            M_arr.append(M)
            M_inv_arr.append(M_inv)
            imgs.append(cv.warpPerspective(warp_img, M, (warp_img.shape[1], warp_img.shape[0])))


        final_img = np.zeros_like(warp_img)
        start_idx = 0
        intervals = []
        for i, dist in enumerate(cum_dist[1:]):
            # Determine the intervals with their corresponding images, that if overlayed with their respective intervals, it will form
            # the final warped image
            final_idx = int(dist)
            final_img[:, start_idx:final_idx] = imgs[i][:, start_idx:final_idx]
            intervals.append([start_idx, final_idx])
            start_idx = final_idx
        final_img_cut = final_img[0:int(width*safety_margin_width), :int(cum_dist[-1])]
        return final_img_cut, img_scaled, M_arr, M_inv_arr, intervals, scale_factor, imgs
    

    @classmethod
    def warp_image_efficient(cls, image: np.ndarray, edge: np.ndarray, tif_file: Optional[str]=None, 
                             safety_margin_width: float = 2.0, 
                             width_lane: float = 3.5, lanes: int = 1, eps: float = 1e-9, crop_margin: float = 1.5, 
                             width_pixels: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        More efficient warp technique which does not warp the entire image, but rather crops the relevant sections,
        Making the warping step scale with N rather than N^2
        """
        if width_pixels is None:
            resolution = cls._calculate_resolution(tif_file)
            width = int(width_lane/resolution)
        else:
            resolution = width_lane/width_pixels
            width = width_pixels

        cum_dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(edge, axis=0), axis=1))])
        
        offset = int(width * safety_margin_width / 2)

        if image.ndim == 2:  # Mask: single channel
            final_image = np.zeros((offset*2, int(cum_dist[-1])), dtype=image.dtype)
        elif image.ndim == 3:  # Image: color channels (e.g., RGB)
            final_image = np.zeros((offset*2, int(cum_dist[-1]), image.shape[2]), dtype=image.dtype)
        else:
            raise ValueError(f"Unexpected number of dimensions in input: {image.ndim}")

        t = 0
        start_points = []
        shifts = []
        intervals = []
        M_inv_arr = []
        M_arr = []

        for i, (pt1, pt2, cd) in enumerate(zip(edge[:-1], edge[1:], cum_dist[:-1])):
            prev_dist, next_dist = int(cum_dist[i]), int(cum_dist[i+1])
            if prev_dist==next_dist:
                continue
            direction = (pt2 - pt1) / (np.linalg.norm(pt2 - pt1) + eps)
            direction_inv = (np.array([-direction[1], direction[0]]))
            
            # points to warp
            src_1 = pt1 -  (width / 2) * direction_inv
            src_2 = pt1 +  (width / 2) * direction_inv
            src_3 = pt2 -  (width / 2) * direction_inv
            src_4 = pt2 +  (width / 2) * direction_inv
            original_src_pts = np.array([src_1, src_2, src_3, src_4])
            
            # Corner points of interest:
            b1 = (pt2+pt1)/2 - (pt2 - pt1) * (crop_margin / 2)
            b2 = (pt2+pt1)/2 + (pt2 - pt1) * (crop_margin / 2)
            c_1 = b1 -  (width / 2) * direction_inv * safety_margin_width * crop_margin
            c_2 = b1 +  (width / 2) * direction_inv * safety_margin_width * crop_margin
            c_3 = b2 -  (width / 2) * direction_inv * safety_margin_width * crop_margin
            c_4 = b2 +  (width / 2) * direction_inv * safety_margin_width * crop_margin
            corner_points = np.array([c_1, c_2, c_3, c_4])
            x_min, x_max = int(np.max((0, np.min(corner_points[:,0])))), int(np.min((np.max(corner_points[:,0]), image.shape[1])))
            y_min, y_max =  int(np.max((0, np.min(corner_points[:,1])))), int(np.min((np.max(corner_points[:,1]), image.shape[0])))         
            origin = np.array([x_min, y_min])
            cropped_img = image[y_min:y_max, x_min:x_max]
            shifted_src_pts = original_src_pts - origin
   
            dist = np.linalg.norm(pt2 - pt1)
            dst_pts = np.array([[0, offset - width/2], [0, offset + width/2], 
                            [dist, offset - width/2], [dist, offset + width/2]])
            
            M = cv.getPerspectiveTransform(shifted_src_pts.astype(np.float32), dst_pts.astype(np.float32))
            M_inv = cv.getPerspectiveTransform(dst_pts.astype(np.float32), shifted_src_pts.astype(np.float32))
            
            M_arr.append(M)
            M_inv_arr.append(M_inv)
            shifts.append(origin)
            intervals.append([prev_dist, next_dist])

            # Perform the warp
            if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                continue
            result_warp = cv.warpPerspective(cropped_img, M, (int(next_dist - prev_dist), offset*2)) 
            final_image[:offset*2,prev_dist:next_dist] = result_warp

        return final_image, np.array(M_arr), np.array(M_inv_arr), np.array(shifts), np.array(intervals), width, resolution


    @classmethod
    def unwarp_mask(cls, original_img: np.ndarray, mask: np.ndarray, intervals: np.ndarray, T_invs: np.ndarray) -> np.ndarray:
        """
        Unwarp the Mask to obtain the original Mask
        """
        original_mask = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=bool)
        for interval, T in zip(intervals, T_invs):

            mask_temp = np.zeros_like(mask)
            mask_temp[:, interval[0]:interval[1]] = mask[:, interval[0]:interval[1]]
            mask_temp = mask_temp.astype(np.uint8)
            transformed_mask_section = cv.warpPerspective(
                mask_temp, T, (original_mask.shape[1], original_mask.shape[0])
            )
            original_mask = original_mask | transformed_mask_section.astype(bool)
        
        return original_mask


    @classmethod
    def unwarp_mask_efficient(cls, original_img: np.ndarray, mask: np.ndarray, intervals: np.ndarray, T_invs: np.ndarray, shifts: np.ndarray, edge: np.ndarray) -> np.ndarray:
        """
        This unwarping procedure matches the efficient warping procedure and also accounts for gaps
        in the mask caused by the pointwise warping (in this case "unwarping") which leads to overlapping pieces in the mask
        as well as some gaps on the outside of corners.
        """

        original_mask = np.zeros((2*original_img.shape[0], 2*original_img.shape[1]), dtype=bool)
        for interval, T, shift in zip(intervals, T_invs, shifts):
            mask_temp = mask[:, interval[0]:interval[1]].astype(np.uint8)
            transformed_mask_section = cv.warpPerspective(
                mask_temp, T, (original_img.shape[1], original_img.shape[0])
            )
            original_mask[shift[1]:original_img.shape[0]+shift[1], shift[0]:original_img.shape[1]+shift[0]] +=  transformed_mask_section.astype(bool)
        
        for i, (interval, T, T_next, shift, shift_next) in enumerate(zip(intervals[:-1], T_invs[:-1], T_invs[1:], shifts[:-1], shifts[1:])):
            mask_relevant = mask[:, interval[1]].astype(int)
            diff_mask = np.concatenate(([0], np.diff(mask_relevant)))
            starts = list(np.argwhere(diff_mask==1).flatten())
            ends = list(np.argwhere(diff_mask==-1).flatten())
            if len(ends) < len(starts):
                ends += [mask.shape[0]]
            for start, end in zip(starts, ends):
                start_left =  (T@(np.array([interval[1]-interval[0],start, 1]).reshape(-1, 1))).flatten()[:2]+ shift
                start_right = (T_next@(np.array([0, start, 1]).reshape(-1, 1))).flatten()[:2] + shift_next
                end_left =  (T@(np.array([interval[1]-interval[0], end, 1]).reshape(-1,1))).flatten()[:2] + shift
                end_right = (T_next@(np.array([0,end, 1]).reshape(-1,1))).flatten()[:2] + shift_next
                pts1 = np.array([start_left, end_left])
                pts2 = np.array([start_right, end_right])

                road_pts = np.array([edge[i], edge[i+1], edge[i+2]])
                corner_mask = cls._correct_turns_in_mask(pts1, pts2, road_pts)
                
                if corner_mask is not None:
                    original_mask[:corner_mask.shape[0], :corner_mask.shape[1]] += corner_mask[:min(corner_mask.shape[0], original_mask.shape[0]), :min(corner_mask.shape[1], original_mask.shape[1])].astype(bool)
                
        original_mask = original_mask[:original_img.shape[0], :original_img.shape[1]]
        return original_mask

    @classmethod
    def unwarp_edge(cls, center: int, intervals: np.ndarray, T_invs: np.ndarray, shifts: np.ndarray) -> np.ndarray:
        transformed_pts = []
        x = np.array(intervals).flatten()
        x_diff = np.array([[interval[0]]*2 for interval in intervals]).flatten()
        x = x - x_diff
        pts = np.vstack([x, center * np.ones_like(x), np.ones_like(x)]).T
        for j, pt in enumerate(pts):
            transformed_pts.append((T_invs[j//2]@pt)[:2]+shifts[j//2])

        return LineString(np.array(transformed_pts))

    @classmethod
    def get_transformed_line(cls, y: int, Mat_invs: np.ndarray, intervals: list) -> np.ndarray:
        """
        The Warping process requires to perform an inverse transformation after the transformation. 
        The fact that the warp is piecewise, means that the transformation has to be performed multiple times.
        These transformations result in some discontinuities, some parts of the mask will overlap and others will have a gap.
        To combine this, this function wil look at intersecting bounds and remove overshooting parts and connect the individual parts.
        This will remove the discontinuities and make it a smoother mask.
        """
        lines = np.empty((0, 2))
        lines_sep = []
        for Mat_inv, interval in zip(Mat_invs, intervals):
            pts_x = interval
            line = np.vstack((pts_x, y * np.ones_like(pts_x), np.ones_like(pts_x))) 
            line_trans = (Mat_inv @ line)   
            line_trans = line_trans[:2] / line_trans[2]
            line_trans = line_trans.T
            lines = np.concatenate((lines, line_trans))
            lines_sep.append(line_trans)
        lines_sep = np.array(lines_sep)
        lines_final = np.empty((0,2))
        prev_intersect = False
        if len(lines_sep)==1:
            lines_final = lines_sep[0]
        else:
            for line1, line2 in zip(lines_sep[:-1], lines_sep[1:]):
                intersect_point, intersect_bool = find_intersection(line1, line2)
                if prev_intersect:
                    line1 = line1[1:]
                    prev_intersect = False
                if intersect_point is None:
                    lines_final = np.concatenate((lines_final, line1))
                    continue
                if not intersect_bool:
                    lines_final = np.concatenate((lines_final, line1, intersect_point))
                else:
                    lines_final = np.concatenate((lines_final, line1[:-1], intersect_point))
                    prev_intersect = True

        return lines_final


    @staticmethod
    def _mask_image_road(image: np.ndarray, edge: np.ndarray, width: int = 150, scale_factor: float = 0.5, safety_factor: float = 2) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine the final image size and scale everyting accordingly.
        """
        final_img_size = int(scale_factor * max(image.shape[0], image.shape[1]) * safety_factor)
        edge_scaled = edge * scale_factor
        
        if len(image.shape)==3:
            final_img_initial = np.zeros((final_img_size, final_img_size, 3), dtype=np.uint8)
        else:
            final_img_initial = np.zeros((final_img_size, final_img_size), dtype=np.uint8)

        return final_img_initial, edge_scaled


    @staticmethod
    def _calculate_resolution(tif_path: str) -> float:
        with rasterio.open(tif_path) as f:
            transform = f.transform
            resolution = np.abs(np.mean([transform[0], -transform[4]]))
        return resolution
   
    @staticmethod
    def _calculate_width_factor(shape_img: tuple, bounds: np.ndarray, width: float, lanes: int = 1) -> int:
        """
        Calculate the required amount of pixels to cover the width of the road
        """
        pixels_per_meter = []
        for i in range(2):  
            dist = get_distance_coordinates(np.array([bounds[1], bounds[0]]), np.array([bounds[int(3 - 2 * i)], bounds[int(2*i)]]))
            pixels_per_meter.append(shape_img[i]/dist)
        width_pixels = int(np.max(pixels_per_meter) * lanes * width)
        return width_pixels


    @classmethod
    def _correct_turns_in_mask(cls, pts1: np.ndarray, pts2: np.ndarray, road_pts: np.ndarray) -> np.ndarray:
        """
        This function serves the purpose to smoothly fill in the corners after unwarping the mask
        The unwarping process, will cause discontinuities at the ends of the intervals where we warped
        This method will fill these gaps with a circle section to make a smooth mask
        """
        vec_road1 = road_pts[1] - road_pts[0]
        vec_road2 = road_pts[2] - road_pts[1]
        direction = "right" if vec_road1[0] * vec_road2[1] - vec_road1[1] * vec_road2[0] > 0 else "left"

        pts1_sides = np.array([cls._point_side_of_line(np.array([road_pts[0], road_pts[1]]), pts1[0]), cls._point_side_of_line(np.array([road_pts[0], road_pts[1]]), pts1[1])])
        pts2_sides = np.array([cls._point_side_of_line(np.array([road_pts[1], road_pts[2]]), pts2[0]), cls._point_side_of_line(np.array([road_pts[1], road_pts[2]]), pts2[1])])
        
        l1 = LineString(pts1)
        l2 = LineString(pts2)
        origin = np.array(l1.intersection(l2).coords)

        relevant_points_1 = pts1[pts1_sides==direction]
        relevant_points_2 = pts2[pts2_sides==direction]
        if len(relevant_points_1)==0 or (len(relevant_points_2)==0):
            return None
        if len(origin) == 0:
            l1_extended = cls._extend_linestring(l1, distance=50)
            l2_extended = cls._extend_linestring(l2, distance=50)
            v_1 = pts1[1] - pts1[0]
            v_2 = pts2[1] - pts2[0]
            start_angle, end_angle = np.arctan2(v_1[1], v_1[0]), np.arctan2(v_2[1], v_2[0])

            if np.abs(start_angle - end_angle) > np.pi:
                if start_angle < 0:
                    start_angle+=2*np.pi
                else:
                    end_angle += 2*np.pi
            if start_angle > end_angle:
                start_angle, end_angle = end_angle, start_angle
            origin = np.array(l1_extended.intersection(l2_extended).coords)
            if len(origin)==0:
                return None
            origin = origin.reshape(2)

            dists_2 = [l1.distance(Point(origin)), l2.distance(Point(origin))]
            r2 = np.max(dists_2)
            dists_1 = [np.max(np.linalg.norm(pts1 - origin, axis=1)), np.max(np.linalg.norm(pts2 - origin, axis=1))]
            r1 = np.min(dists_1)
            arc1 = cls._create_arc(origin, r1, start_angle, end_angle, num_points = 100)
            arc2 = cls._create_arc(origin, r2, start_angle, end_angle, num_points = 100)

            poly, shape = cls._create_polygon(arc1, arc2=arc2)
            if shape[0] <= 0 or shape[1] <= 0:
                return None
            mask = cls._polygon_to_mask_pixel_frame(poly, shape).astype(bool)
        else:
            origin = origin.flatten()
            if len(origin) != 2:
                return None
            
            dists = np.array([np.linalg.norm(origin-relevant_points_1), np.linalg.norm(origin-relevant_points_2)])
            r1 = np.min(dists)
            v_1 = (relevant_points_1 - origin).flatten()
            v_2 = (relevant_points_2 - origin).flatten()

            start_angle, end_angle = np.arctan2(v_1[1], v_1[0]), np.arctan2(v_2[1], v_2[0]) 

            if np.abs(start_angle - end_angle) > np.pi:
                if start_angle < 0:
                    start_angle+=2*np.pi
                else:
                    end_angle += 2*np.pi
            if start_angle > end_angle:
                start_angle, end_angle = end_angle, start_angle
            arc1 = cls._create_arc(origin, r1, start_angle, end_angle, num_points = 100)
            poly, shape = cls._create_polygon(arc1, origin=origin)
            if shape[0] <= 0 or shape[1] <= 0:
                return None
            mask = cls._polygon_to_mask_pixel_frame(poly, shape).astype(bool)
        return mask


    @staticmethod
    def _extend_linestring(linestring: LineString, distance: float) -> LineString:
        """
        Extend the linestring. This serves the purpose to find intersections between linestrings
        if they are not within the domain of the linestrings itself
        """
        if len(linestring.coords) != 2:
            raise ValueError("The LineString must have exactly two points")

        # Extract the start and end points
        start, end = linestring.coords
        direction = np.array([end[0] - start[0], end[1] - start[1]])
        unit_direction =  direction / np.linalg.norm(direction) 
        
        # Extend both ends
        new_start = start - distance * unit_direction
        new_end = end + distance * unit_direction

        return LineString([new_start, new_end])


    @staticmethod
    def _create_arc(origin: np.ndarray, radius: float, start_angle: float, end_angle: float, 
                    num_points: int = 100, extra: float = 0.05) -> np.ndarray:
        """
        Generate an arc connecting the corner points of the individual warped sections  
        """
        angles = np.linspace(start_angle-extra, end_angle+extra, num_points)

        arc_points = np.array([(origin[0] + radius * np.cos(angle),
                                origin[1] + radius * np.sin(angle)) for angle in angles])
        return arc_points
    

    @staticmethod
    def _polygon_to_mask_pixel_frame(polygon: Polygon, shape: tuple) -> np.ndarray:
        """
        Generate a mask from polygon
        """
        if not isinstance(polygon, Polygon):
            raise ValueError("Input must be a shapely Polygon")
        transform = Affine.identity()
        # Rasterize the polygon
        mask = rasterize(
            [(polygon, 1)],  # The polygon and its value
            out_shape=shape,  # The shape of the output array
            fill=0,  # Value for areas outside the polygon
            transform=transform, 
            dtype='uint8'
        )
        return mask


    @staticmethod
    def _create_polygon(arc1: np.ndarray, origin: Optional[np.ndarray] = None, arc2: Optional[np.ndarray] = None) -> tuple[np.ndarray, tuple]:
        """
        Create a polygon from two arcs or from an origin and and arc
        """
        if (origin is None) and (arc2 is None):
            raise ValueError("Incomplete polygon")
        elif origin is None:
            pts = np.concatenate((arc1, arc2[::-1]))
        elif arc2 is None:
            pts = np.concatenate((arc1, [origin]))
        else:
            raise ValueError("Can not generate Polygon from two arcs and an origin.")
        
        shape = (int(np.max(pts[:,1]))+1, int(np.max(pts[:,0]))+1)
        return Polygon(pts), shape
    
    @staticmethod
    def _point_side_of_line(line: np.ndarray, point: np.ndarray) -> str:
        """
        Determine on which side of the line a point is
        """
        sgn = np.sign(np.cross(line[1] - line[0], point - line[0]))
        if sgn == 1:
            return "left"
        else:
            return "right"
