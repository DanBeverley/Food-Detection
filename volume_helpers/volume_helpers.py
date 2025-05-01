import numpy as np
import logging
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError 

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def depth_map_to_masked_points(depth_map:np.ndarray, segmentation_mask:np.ndarray,
                               fx:float, fy:float, cx:float, cy:float) -> np.ndarray | None:
    """
    Converts a depth map region (defined by a mask) to a 3D point cloud.

    Args:
        depth_map (np.ndarray): 2D numpy array of depth values (e.g., in mm).
                                Assumes invalid/no-return depth is <= 0.
        segmentation_mask (np.ndarray): 2D boolean numpy array (same shape as depth_map)
                                        where True indicates the object pixels.
        fx (float): Camera focal length in x (pixels).
        fy (float): Camera focal length in y (pixels).
        cx (float): Camera principal point x (pixels).
        cy (float): Camera principal point y (pixels).

    Returns:
        np.ndarray | None: An (N, 3) numpy array of 3D points (x, y, z) in mm,
                          or None if no valid points are found or inputs are invalid.
    """
    if depth_map.ndim != 2 or segmentation_mask.ndim != 2:
        logger.error("Depth map and mask must be 2D arrays")
        return None
    if depth_map.shape != segmentation_mask.shape:
        logger.error(f"Depth map shape {depth_map.shape} and mask shape {segmentation_mask.shape} must have the same shape")
        return None
    if not np.issubdtype(segmentation_mask, np.bool_):
        logger.warning("Segmentation mask is not boolean, attempting conversion")
        segmentation_mask = segmentation_mask.astype(bool)
    height, width = depth_map.shape
    # Create pixel indices
    jj, ii = np.meshgrid(np.arange(width), np.arange(height)) #jj stands for x and ii for y
    # Select indices and depth values within the mask
    mask_indices = np.where(segmentation_mask) # Returns tuple of arrays (row_indices, col_indices)
    ii_masked = ii[mask_indices]
    jj_masked = jj[mask_indices]
    depth_value_masked = depth_map[mask_indices]

    # Filter out invalid depth values (e.g., <= 0)
    valid_depth_filter = depth_value_masked > 0
    ii_final = ii_masked[valid_depth_filter]
    jj_final = jj_masked[valid_depth_filter]
    depth_final = depth_value_masked[valid_depth_filter]

    if depth_final.size == 0:
        logger.warning(f"No vald depth pixels found within the segmentation mask")
        return None
    # Convert pixel coordinates and depth to 3D points (mm)
    # Follows standard pinhole camera model projection equations inverted
    x = (jj_final - cx)*depth_final / fx
    y = (ii_final - cy)*depth_final / fy
    z = depth_final

    points = np.vstack((x, y, z)).T
    logging.info(f"Generated {points.shape[0]} valid 3D points from masked depth map.")
    return points