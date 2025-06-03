import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from samgeo.text_sam import LangSAM
from typing import Optional
from segment_anything import SamPredictor

from scipy.optimize import differential_evolution
from .utils import get_iou
from PIL import Image
    
class HoughLinesDetector:
    def __init__(self):
        pass
    
    @staticmethod
    def _include_bound_edge_image(edge_img, extension=20):
        """
        In case the boundaries are important and could include relevant edges, 
        we want to include the edges found on the boundary of the image.
        """
        extended_edge_img = np.zeros((edge_img.shape[0]+2*extension, edge_img.shape[1] + 2 * extension), dtype=np.uint8)
        extended_edge_img[extension:-extension, extension:-extension] = edge_img
        return extended_edge_img
    
    
    @classmethod
    def edge_detection(cls, img: np.ndarray, threshold1: int = 200, threshold2: int = 400, apertureSize: int = 3, extension: int = 20) -> np.ndarray:
            """
            Perform Canny Edge detection.
            """
            if len(img.shape)==3:
                gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            else:
                gray_img = img
            if extension > 0:
                extended_gray_img = cls._include_bound_edge_image(gray_img, extension=extension)
            else:
                extended_gray_img = gray_img
            edge_img = cv.Canny(extended_gray_img, threshold1, threshold2, None, apertureSize)

            return edge_img
        

    @classmethod
    def find_boundary_outline(cls, img: np.ndarray, max_percentage_black: float = 0.075) -> list:
        """
        Find the boundaries (often) created by the warp, which leads to a black area.
        This is only required in vertical direction.
        """
        mask = cv.cvtColor(img, cv.COLOR_RGB2GRAY).astype(bool).astype(np.uint8)
        if np.sum(mask)>(1-max_percentage_black)*(img.shape[0]*img.shape[1]):
            y_range = [0, img.shape[0]]
        else:
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours!=():
                contours = contours[0].reshape(contours[0].shape[0],contours[0].shape[2])
                y_c = contours[:,1]
                y_under = y_c[y_c < int(mask.shape[0]/2)]
                y_over = y_c[y_c > int(mask.shape[0]/2)]
                if len(y_under)==0:
                    min_y = 0
                else:
                    min_y = np.max(y_under) 
                if len(y_over)==0:
                    max_y = img.shape[0]
                else:
                    max_y = np.min(y_over)
                y_range = [min_y, max_y]
            else:
                y_range = [0, img.shape[0]]
        return y_range


    @classmethod
    def filter_contour_edges(cls, edge_img: np.ndarray, original_img: np.ndarray, filter_size: int = 100) -> np.ndarray:
        """
        Filter out edges that Include the contours, since those will not correspond to the actual road boundaries
        """
        mask_original_img = cv.cvtColor(original_img, cv.COLOR_RGB2GRAY).astype(bool).astype(np.uint8)
        mask = cv.erode(mask_original_img, np.ones((filter_size, filter_size), dtype=np.uint8))
        edge_img = edge_img * mask
        return edge_img


    @classmethod
    def find_hough_linesp(cls, edge_img: np.ndarray, original_img: np.ndarray, rho: float = 1, theta: float = np.pi/180, threshold: int = 50, 
                          minLineLength: int = 50, maxLineGap: int = 10, width_line: int =10, extension: int = 20) -> tuple[list, np.ndarray]:
        """
        Find Hough Lines that could correspond to the boundaries of the road
        """
        linesP = cv.HoughLinesP(edge_img, rho, theta, threshold, None, minLineLength, maxLineGap)
        mask_lines = np.zeros_like(edge_img, dtype=np.uint8)
        
        if linesP is not None:
            linesP[:,:,0] = np.clip(linesP[:,:,0] - extension, 0, original_img.shape[1])
            linesP[:,:,1] = np.clip(linesP[:,:,1] - extension, 0, original_img.shape[0])
            linesP[:,:,2] = np.clip(linesP[:,:,2] - extension, 0, original_img.shape[1])
            linesP[:,:,3] = np.clip(linesP[:,:,3] - extension, 0, original_img.shape[0])
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(mask_lines, (l[0], l[1]), (l[2], l[3]), 255, width_line, cv.LINE_AA)
        return linesP, mask_lines


    @classmethod
    def get_y_and_rc(cls, linesP: list, epsilon: float = 1e-5, max_rc: float = 0.08):
        """
        For each of the found lines, we want to find the y coordinate at x=0 and their directional coefficient (rc), so that we have a formula for each line.
        If the rcs are to high, they will not be considered, since the road should be horizontal, so then that line is not likely to correspond to the road boundary.
        """
        rcs = np.divide(linesP[:,:,3]-linesP[:,:,1], (linesP[:,:,2] - linesP[:,:,0] + epsilon))
        start_ys = linesP[:,:,1] - np.multiply(rcs, linesP[:,:,0])
        start_ys = start_ys[np.abs(rcs)<=max_rc]
        start_ys = start_ys.reshape((start_ys.shape[0],1))
        linesP_rel = linesP[np.abs(rcs)<=max_rc]
        linesP_rel = linesP_rel.reshape(linesP_rel.shape[0], 1, linesP_rel.shape[1])
        rcs = rcs[np.abs(rcs)<=max_rc]
        rcs = rcs.reshape((rcs.shape[0],1))
        return linesP_rel, start_ys, rcs


    @classmethod
    def cluster_houghlines(cls, img: np.ndarray, hough_lines: list, start_ys: int, colors: list = [(255,0,0), (0,255,0), (0,0,255)], 
                           eps: float = 10, min_samples: int = 5) -> tuple[DBSCAN, np.ndarray]:
        """
        Based on y(x=0) of each line, we will cluster the hough lines
        """
        if len(hough_lines)==0:
            return None, None
        cluster = DBSCAN(eps=10, min_samples=5)
        cluster.fit(start_ys)
        copy_img = img.copy()
        for i, line in enumerate(hough_lines):
            l = line[0]
            lab = cluster.labels_[i]
            if lab==-1:
                continue
            cv.line(copy_img, (l[0], l[1]), (l[2], l[3]), colors[lab%len(colors)], 3, cv.LINE_AA)
        return cluster, copy_img


    @classmethod
    def get_cluster_means(cls, img: np.ndarray, cluster: DBSCAN, start_ys: int, rcs: list, 
                          colors: list = [(255,0,0), (0,255,0), (0,0,255)]) -> tuple[np.ndarray, list]:
        """
        Based on the computed clusters, we obtain the mean of each cluster and these mean lines will represent a line that will be used to compute the bounding boxes.
        """
        img_mean_lines = img.copy()
        lines = []
        if cluster is not None:
            for lab in np.unique(cluster.labels_):
                if lab ==-1:
                    continue
                rcs_reshaped = rcs.reshape(1,-1).flatten()

                y_s = np.mean(start_ys[cluster.labels_==lab]).astype(int)
                x_s = 0
                x_e = img_mean_lines.shape[1]
                y_e = (y_s + np.mean(rcs_reshaped[cluster.labels_==lab]) *x_e).astype(int)
                lines.append([[x_s, y_s],[x_e, y_e]])
                cv.line(img_mean_lines, (x_s, y_s), (x_e, y_e), colors[lab%len(colors)], 3, cv.LINE_AA)
        return img_mean_lines, lines

class MaskOperator:
    def __init__(self):
        pass
    
    @classmethod
    def find_bboxes(cls, warped_img: np.ndarray, text_prompt: str, img_path: str, lang_sam: LangSAM, colors: list = [(255,0,0), (0,255,0), (0,0,255)]) -> list:
        """
        Find the bounding boxes. First we try to find it with grounding dino, if this does not yield to succes we use the houghlines approach.
        In the houghlines approach we also incorporate the outer most possible edges, to ensure that we will always get a bounding box.
        """
        boxes = []

        img_shape = warped_img.shape
        bboxes, masks = cls.apply_grounding_dino(img_path, text_prompt, lang_sam)
        if len(bboxes) != 0:
            bboxes = bboxes.cpu().numpy().tolist()
            masks = masks.cpu().data.numpy()
        else:
            bboxes = [[0,0,warped_img.shape[1], warped_img.shape[0]]]
        # # bboxes = [[0,0,warped_img.shape[1], warped_img.shape[0]]]
        # edge_img = HoughLinesDetector.edge_detection(warped_img, extension=0)
        # edge_img = HoughLinesDetector.filter_contour_edges(edge_img, warped_img)
        # linesP, mask_lines = HoughLinesDetector.find_hough_linesp(edge_img, warped_img)
        # if linesP is None:
        #     cluster, warped_img_hough, start_ys, rcs = None, None, None, None
        # else:
        #     linesP, start_ys, rcs = HoughLinesDetector.get_y_and_rc(linesP, max_rc=0.03)
        #     if len(linesP)!=0:
        #         cluster, warped_img_hough = HoughLinesDetector.cluster_houghlines(warped_img, linesP, start_ys, colors=colors)
        #     else:
        #         cluster, warped_img_hough = None, None
        # warped_img_hough_clustered, lines = HoughLinesDetector.get_cluster_means(warped_img, cluster, start_ys, rcs, colors=colors)
    
        # bboxes += cls.build_bbox_lines(warped_img, lines)
        bboxes = np.array(bboxes)
        return bboxes

    
    @classmethod
    def apply_grounding_dino(cls, image: str, text_prompt: str, lang_sam: LangSAM, box_threshold: float = 0.22, text_threshold: float = 0.22) -> tuple[list, list]:
        """
        Apply Grounding Dino with a given text_prompt to find the bounding boxes corresponding to the text_prompt
        """
        try:
            results = lang_sam.predict(image, text_prompt, box_threshold=box_threshold, text_threshold=text_threshold, return_results=True)
        except RuntimeError:
            return [], []
        if results is None:
            return [], []
        return lang_sam.boxes, lang_sam.masks
    
    
    @classmethod
    def build_bbox_lines(cls, image: np.ndarray, lines: list) -> list:
        """
        Create all possible bounding boxes from all the computed horizontal lines in the image
        """
        y_range = HoughLinesDetector.find_boundary_outline(image)
        extra_lines = [[[0, y_range[0]], 
                        [image.shape[1], y_range[0]]],
                    [[0,y_range[1]], 
                        [image.shape[1], y_range[1]]]]
        
        all_lines = extra_lines + lines
        all_lines = np.array(all_lines)
        bboxes = []
        for i, line1 in enumerate(all_lines[:-1]):
            for j, line2 in enumerate(all_lines[i+1:]):
                bbox = np.array([np.min(np.concatenate((line1[:,0], line2[:,0]))),
                                np.min(np.concatenate((line1[:,1], line2[:,1]))),
                                np.max(np.concatenate((line1[:,0], line2[:,0]))),
                                np.max(np.concatenate((line1[:,1], line2[:,1])))])
                bboxes.append(bbox)
        return bboxes
    

    @classmethod
    def predict_mask_sam_bbox(cls, sam_predictor: SamPredictor, bbox: np.ndarray, multimask_output: bool = True) -> tuple[list, list, list]:
        """
        Predict a mask based on a bounding box as the input to the SAM network
        """
        masks, scores, logits = sam_predictor.predict(
            box=bbox,
            multimask_output=multimask_output
        )
        return masks, scores, logits


    @classmethod
    def get_tree_mask(cls, image_path: str, lang_sam: LangSAM, text_prompt: str = "trees", 
                      box_threshold: float = 0.28, text_threshold: float = 0.28, viz: bool = False) -> np.ndarray:
        """
        Obtain the mask of trees using language Sam.
        """
        res = lang_sam.predict(image_path, text_prompt, box_threshold=box_threshold, text_threshold=text_threshold, return_results=True)
        img_shape = cv.imread(image_path).shape
        mask_tree = np.zeros(img_shape[:-1], dtype=bool)
        if res is not None:
            masks = lang_sam.masks.cpu().numpy()
            for m in masks:
                mask_tree = np.logical_or(mask_tree, m)
            if viz:
                plt.figure()
                plt.title("trees")
                plt.imshow(mask_tree)
                plt.show()
        return mask_tree
    

    @classmethod
    def visualize_mask_bbox(cls, img: np.ndarray, mask: np.ndarray, bbox: Optional[np.ndarray] = None, road_line: Optional[np.ndarray] = None, 
                            color_bbox: tuple = (255, 0, 0), thickness: int = 10, color_mask: tuple = (255 , 0, 0), alpha_mask: int = 100, 
                            viz: bool = False, save: bool = False, id: str = "", city: str = "Bratislava", name=None) -> None:
        """
        Visualize the results and possible save these images.
        """
        alpha_mask = 0
        img_copy = img.copy()
        if bbox is not None:
            img_copy = cv.rectangle(img_copy, bbox[:2], bbox[2:], color=color_bbox, thickness=thickness) 
            img_copy = cv.line(img_copy, (0, int(0.5*img_copy.shape[0])), (img_copy.shape[1], int(0.5*img_copy.shape[0])), (0,255,0), 3, cv.LINE_AA)
        if road_line is not None:
            img_copy = cv.polylines(img_copy, [road_line.astype(np.int32)], isClosed=False, color=(0,255,0), thickness=3)
        rgba = [color_mask[0], color_mask[1], color_mask[2], alpha_mask]
        rgba_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba_mask[mask.astype(bool)] = rgba
        mask_pil = Image.fromarray(rgba_mask)
        img_pil = Image.fromarray(img_copy).convert("RGBA")
        # Red color with 50% opacity (128 out of 255)
        combined_image = Image.alpha_composite(img_pil, mask_pil)
        if viz:
            combined_image.show()
        if save:
            if not os.path.isdir(os.path.join("results_mask_road", city)):
                os.mkdir(os.path.join("results_mask_road", city))
            print("SAVE")
            combined_image.save(os.path.join("results_mask_road", city, f"annotated_road{id}.png"))
            if not os.path.isdir(os.path.join("post_processing_data", city)):
                os.makedirs(os.path.join("post_processing_data", city, "imgs"))
                os.makedirs(os.path.join("post_processing_data", city, "masks"))
            cv.imwrite(name if name is not None else os.path.join("post_processing_data", city, "imgs", f"img{id}.png"), img)
            np.save(os.path.join("post_processing_data", city, "masks", f"mask{id}.npy"), mask)

    

class MaskPostProcessor:
    def __init__(self):
        pass


    @classmethod
    def find_highest_iou_rect_score(cls, params: list, mask: np.ndarray, eps: float = 1e-8) -> float:
        a, b = params
        a, b = int(a), int(b)
        mask_end = np.zeros_like(mask)
        mask_end[a:b, :] = True
        intersection = np.sum(np.logical_and(mask_end, mask))
        union = np.sum(np.logical_or(mask_end, mask))
        
        return 1 / ((intersection / (((union-intersection) + eps)) + eps))


    @classmethod
    def find_best_iou_match_road(cls, mask: np.ndarray) -> np.ndarray:
        bounds = [(0, mask.shape[0]), (0, mask.shape[0])]
        road_fit = differential_evolution(cls.find_highest_iou_rect_score, bounds, strategy='best1bin', seed=42, args=[mask]) 
        r_mask = np.zeros_like(mask)
        r_mask[int(road_fit.x[0]):int(road_fit.x[1]), :] = True
        y1, y2 = int(road_fit.x[0]), int(road_fit.x[1])
        if y1 > y2:
            y1, y2 = y2, y1
        c = (y1 + y2) / 2
        w = (y2 - y1)
        return r_mask, c, w, road_fit.fun


    # Car Removal
    @classmethod
    def eliminate_cars(cls, mask: np.ndarray, image_path: str, lang_sam: LangSAM, text_prompt: str = "vehicles",
                       box_threshold: float = 0.25, text_threshold: float = 0.25, max_format: int = 200, 
                       box_extension_factor: float = 0.2, viz: bool = True) -> np.ndarray:
        """
        If a car is on the road, it should be included in the mask of the road
        This method will inpaint the mask, by detecting cars and inpainting the corresponding bounding boxes.
        """
        res = lang_sam.predict(image_path, text_prompt, box_threshold=box_threshold, 
                               text_threshold=text_threshold, return_results=True)
        mask_final = mask.copy()
        if res is not None:
            if viz:
                lang_sam.show_anns(
                cmap="Greens",
                box_color="red",
                title="Automatic Segmentation of Cars",
                blend=True,
                )
                plt.show()
            boxes = cls.validate_car_bboxes(lang_sam.boxes.cpu().numpy(), max_format=max_format)
            mask_final = cls.inpaint_cars(mask, boxes, box_extension_factor=box_extension_factor)
        return mask_final
    

    @classmethod
    def validate_car_bboxes(cls, bboxes: np.ndarray, max_format: int = 200) -> np.ndarray:
        """
        To avoid adding bounding boxes that do not correspond to a car, we take a way too big bounding boxes, 
        which could not correspond to a car. Still it is possible that we get false positives, 
        but since we only take small boxes, they are not as catestrophic to the final results
        This returns the bounding boxes that are valid
        """
        return bboxes[[((np.abs(box[2]-box[0]) < max_format) and (np.abs(box[3]-box[1]) < max_format)) for box in bboxes]]
    
    
    @classmethod
    def inpaint_cars(cls, mask: np.ndarray, car_boxes: np.ndarray, box_extension_factor: float = 0.2) -> np.ndarray:
        """
        Take the current mask of the road and inpaint the car boxes with some extension factor 
        to make sure that the entire car is captured by it. Inpainting is not simply giving it a True for all the pixels in those boxes,
        but it looks at neighboring pixels. It returns a mask where the gaps caused by cars on the road are removed.
        """
        mask_copy = mask.copy().astype(np.uint8)
        for box in car_boxes:
            p = box_extension_factor
            w, h = np.abs(box[2]-box[0]) , np.abs(box[3]-box[1])
            box = np.array([box[0] - w*p, box[1] - h*p, box[2] + w*p, box[3] + h*p]).astype(int)
            box = box.astype(int)
            car_mask = np.zeros_like(mask_copy)
            car_mask[box[1]:box[3], box[0]:box[2]] = 1
            mask_copy = cv.inpaint(mask_copy, car_mask, inpaintRadius=0, flags=cv.INPAINT_TELEA)
            mask_copy = cv.dilate(mask_copy, np.ones((2,2), dtype=np.uint8), iterations=1)
            mask_copy = cv.erode(mask_copy, np.ones((2,2), dtype=np.uint8), iterations=1)
        mask_copy = mask_copy.astype(bool)
        return mask_copy
    
    @classmethod
    def find_post_processed_mask(cls, mask: np.ndarray, image: np.ndarray, viz: bool = True, min_width_part: float = 0.2) -> np.ndarray:
        image_copy = image.copy()
        edge_img = HoughLinesDetector.edge_detection(mask.astype(np.uint8)*255)
        linesP, mask_lines = HoughLinesDetector.find_hough_linesp(edge_img, image_copy, rho=1, theta= np.pi/180, threshold=50, 
                            minLineLength = 100, maxLineGap= 80)
        if linesP is not None:
            linesP, _, rcs = HoughLinesDetector.get_y_and_rc(linesP, max_rc=0.025)

            ys = (np.array([[(line[0][3] + line[0][1])/2 for line in linesP]])).T
            if len(ys) >=2:
                clusterer = KMeans(n_clusters=2)
                clusterer.fit(ys)
                if viz:
                    for line, lab in zip(linesP, clusterer.labels_):
                        image_copy = cv.line(image_copy, np.array(line[0])[:2], np.array(line[0])[2:], color=colors[lab], thickness=10)
                    plt.imshow(image_copy)
                    plt.show()
                    image_copy = image.copy()
                labs = clusterer.labels_.astype(bool)
                if np.any(labs): # In case the vectors are the same, so there is only one cluster
                    y1 = int(np.mean(ys[labs]))
                    y2 = int(np.mean(ys[~labs]))
                    if y1 > y2:
                        y1, y2 = y2, y1
                    if viz: 
                        line_1 = np.array([[0, y1], [image.shape[1], y1]]).astype(int)
                        line_2 = np.array([[0, y2], [image.shape[1], y2]]).astype(int)

                        image_copy = cv.line(image_copy, line_1[0], line_1[1], color=colors[0], thickness=10)
                        image_copy = cv.line(image_copy, line_2[0], line_2[1], color=colors[1], thickness=10)
                        plt.imshow(image_copy)
                        plt.show()
                        image_copy = image.copy()

                    mask_final = np.zeros(image.shape[:2], dtype=bool)
                    mask_final[y1:y2, :] = True
                    if viz:
                        plt.imshow(mask_final)
                        plt.show()
                    c = (y1 + y2)/ 2
                    w = (y2 - y1)
                    if w > (min_width_part * image.shape[0]):
                        return mask_final, c, w
        mask_final, c, w, _ = cls.find_best_iou_match_road(mask)
        return mask_final, c, w
    

    @classmethod
    def fill_cavities(cls, mask: np.ndarray, max_cav_size: int = 50) -> np.ndarray:
        """
        This method removes small holes and small outliers from the mask, 
        so the main objective is to make the mask a bit smoother
        """
        kernel = np.ones((max_cav_size, max_cav_size))
        mask_copy = mask.copy()
        mask_fill = cv.dilate(mask_copy.astype(np.uint8), kernel, iterations=1)
        mask_fill = cv.erode(mask_fill, kernel, iterations=1).astype(bool)
        mask_copy2 = mask.copy()
        mask_erode = cv.erode(mask_copy.astype(np.uint8), kernel, iterations=1)
        mask_erode = cv.dilate(mask_erode, kernel, iterations=1).astype(bool)
        mask_final = np.logical_and(mask_fill, mask_erode)
        return mask_final
    

class MaskPicker:
    def __init__(self):
        pass

    
    @classmethod
    def select_best_mask(cls, masks: np.ndarray, house_mask: np.ndarray, tree_mask: np.ndarray, 
                         water_mask: np.ndarray, constraints: dict, resolution: float, verbose: bool = False) -> dict:
        """
        From all the created masks, this method selects the best mask based on some defined weights and some limits
        """
        results = {}
        for i, mask in enumerate(masks):
            if mask.shape != masks[0].shape or (np.sum(mask)==0):
                if verbose:
                    print("Invalid Mask shape found")
                continue

            # Constraints
            if get_iou(tree_mask, mask) > constraints["tree"]:
                if verbose:
                    print("Invalid Mask found (trees)")

                continue
        
            if get_iou(house_mask, mask) > constraints["house"]:
                if verbose:
                    print("Invalid Mask found (houses)")

                continue
        
            if get_iou(water_mask, mask) > constraints["water"]:
                if verbose:
                    print("Invalid Mask found (houses)")
                continue
            
            r_mask, c, w, score = MaskPostProcessor.find_best_iou_match_road(mask)
            if w < (constraints["width"]/resolution): #convert meters to pixels
                if verbose:
                    print("Invalid Mask found (too narrow road mask)")
                continue
            
            results[str(i)] = {"mask": mask, "width": w, "score": score, "center": c}
            
            
        if results == {}:
            if verbose:
                print("no valid mask")
            return {"mask": None, "width": None, "score": None, "center": None}
        
        best_mask_idx = min(results, key=lambda k: results[k]["score"])
        if verbose:
            print("Best Mask idx: ", best_mask_idx)
        return results[best_mask_idx]
    

class MaskCombiner:
    def __init__(self):
        pass

    @classmethod
    def combine_edge_masks(cls, centers: list, widths: list, road_scores: list) -> tuple[int, int, int, int]:
        """
        Combine the edges based on the scores and output a common width and center
        """
        sum_scores = np.sum(road_scores)
        center = 0
        width = 0
        tot_s = 0
        for c, w, s in zip(centers, widths, road_scores): 
            center += (sum_scores / s)**2 * c
            width += (sum_scores / s)**2 * w
            tot_s += (sum_scores / s)**2 
        center /= tot_s
        width /= tot_s
        y_1 = center - width / 2
        y_2 = y_1 + width
        return int(y_1), int(y_2), center, width
