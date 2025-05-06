import numpy as np
import logging
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError 
from typing import Optional

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def depth_map_to_masked_points(depth_map:np.ndarray, segmentation_mask:np.ndarray,
                               fx:float, fy:float, cx:float, cy:float,
                               min_depth_m: Optional[float] = None,
                               max_depth_m: Optional[float] = None) -> Optional[np.ndarray]:
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
        min_depth_m (Optional[float]): Minimum valid depth in METERS. Points below this are discarded.
        max_depth_m (Optional[float]): Maximum valid depth in METERS. Points above this are discarded.

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
    # Ensure mask is boolean type for indexing
    if not np.issubdtype(segmentation_mask.dtype, np.bool_):
        logging.debug(f"Converting segmentation mask from {segmentation_mask.dtype} to bool.")
        segmentation_mask = segmentation_mask.astype(bool)
    height, width = depth_map.shape
    # Create pixel indices
    jj, ii = np.meshgrid(np.arange(width), np.arange(height)) #jj stands for x and ii for y
    # Select indices and depth values within the mask
    mask_indices = np.where(segmentation_mask) # Returns tuple of arrays (row_indices, col_indices)
    ii_masked = ii[mask_indices]
    jj_masked = jj[mask_indices]
    depth_value_masked = depth_map[mask_indices]

    # Convert min/max depth from meters (config) to millimeters (depth map unit)
    min_depth_mm = min_depth_m * 1000 if min_depth_m is not None else 0 # Default min to 0 if None
    max_depth_mm = max_depth_m * 1000 if max_depth_m is not None else np.inf # Default max to infinity if None

    # Filter out invalid depth values (e.g., <= 0) and values outside min/max range
    valid_depth_filter = (
        (depth_value_masked > min_depth_mm) & 
        (depth_value_masked < max_depth_mm)
    )

    ii_final = ii_masked[valid_depth_filter]
    jj_final = jj_masked[valid_depth_filter]
    depth_final = depth_value_masked[valid_depth_filter]

    if depth_final.size == 0:
        logger.warning(f"No valid depth pixels found within the segmentation mask")
        return None
    # Convert pixel coordinates and depth to 3D points (mm)
    # Follows standard pinhole camera model projection equations inverted
    x = (jj_final - cx)*depth_final / fx
    y = (ii_final - cy)*depth_final / fy
    z = depth_final

    points = np.vstack((x, y, z)).T
    logging.info(f"Generated {points.shape[0]} valid 3D points from masked depth map.")
    return points

def estimate_volume_convex_hull(points:np.ndarray) -> float:
    """
    Estimates the volume from a set of 3D points using Convex Hull.

    Args:
        points (np.ndarray): An (N, 3) numpy array of 3D points. Units should be
                             consistent (e.g., mm) for volume to be in mm³.

    Returns:
        float: Estimated volume in cubic units (consistent with point coordinates),
               or 0.0 if estimation fails (e.g., fewer than 4 points, co-planar points).
    """
    if points is None or points.ndim != 2 or points.shape[1] != 3:
        logger.error("Invalid input points array for volume estimation")
        return 0.0
    num_points = points.shape[0]
    if num_points < 4:
        logger.warning(f"Need at least 4 points for ConvexHull volume estimation but got {num_points}")
        return 0.0
    try:
        hull = ConvexHull(points)
        volume = hull.volume 
        logger.info(f"Estimated volume using ConvexHull: {volume:.4f}")
        # Volume unit is cubic unit of input points (e.g., mm^3 if points are in mm)
        return volume
    except QhullError as e:
        logger.warning(f"ConvexHull volume estimation failed: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"An unexpected error occured during ConvexHull volume estimation: {e}")
        return 0.0

# Example usage:
if __name__ == '__main__':
    print("Testing volume_helpers (Convex Hull method)...")

    # --- Create Dummy Data ---
    H, W = 100, 120
    # Dummy Depth Map (mm)
    dummy_depth = np.random.rand(H, W).astype(np.float32) * 500 + 100 # Depths between 100mm and 600mm
    dummy_depth[50:70, 50:70] += 200 # Make a region slightly closer
    dummy_depth[0:10, 0:10] = 0 # Add some invalid depth areas

    # Dummy Segmentation Mask (boolean)
    dummy_mask = np.zeros((H, W), dtype=bool)
    dummy_mask[40:80, 40:80] = True # A square object in the middle

    # Dummy Camera Intrinsics (Example for a 120x100 camera)
    fx, fy = 150.0, 150.0 # Focal lengths
    cx, cy = W / 2 - 0.5, H / 2 - 0.5 # Principal point (adjust for 0-based indexing)

    # --- Convert depth map region to points ---
    masked_points = depth_map_to_masked_points(dummy_depth, dummy_mask, fx, fy, cx, cy)

    if masked_points is not None:
        print(f"Successfully extracted {masked_points.shape[0]} points using the mask.")

        # --- Estimate Volume ---
        volume_ch = estimate_volume_convex_hull(masked_points)
        # Units will be mm³ if depth was in mm
        print(f"Estimated Volume (Convex Hull): {volume_ch:.2f} mm³")

        # --- Test degenerate cases ---
        print("\nTesting degenerate cases:")
        # Fewer than 4 points
        few_points = masked_points[:3] if masked_points.shape[0] >= 3 else None
        vol_few = estimate_volume_convex_hull(few_points)
        print(f"Volume estimate with {few_points.shape[0] if few_points is not None else 0} points: {vol_few}")

        # Co-planar points (create a flat plane)
        if masked_points is not None and masked_points.shape[0] >= 4:
            planar_points = masked_points.copy()
            planar_points[:, 2] = np.mean(planar_points[:, 2]) # Set all Z to the average Z
            vol_planar = estimate_volume_convex_hull(planar_points)
            print(f"Volume estimate with co-planar points: {vol_planar}") # Expect 0.0 or error
    else:
        print("Failed to extract masked points from dummy data.")