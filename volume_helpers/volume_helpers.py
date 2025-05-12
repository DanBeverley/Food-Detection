import numpy as np
import logging
from scipy.spatial import ConvexHull, QhullError
from typing import Optional
import trimesh
from trimesh.repair import fill_holes

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def depth_map_to_masked_points(depth_map:np.ndarray, segmentation_mask:np.ndarray,
                               fx:float, fy:float, cx:float, cy:float,
                               min_depth_m: Optional[float] = None,
                               max_depth_m: Optional[float] = None,
                               depth_scale_factor: float = 1.0) -> Optional[np.ndarray]:
    """
    Converts a depth map region (defined by a mask) to a 3D point cloud.

    Args:
        depth_map (np.ndarray): 2D numpy array of depth values.
                                Assumes invalid/no-return depth is <= 0.
                                Values are scaled by depth_scale_factor to get mm.
        segmentation_mask (np.ndarray): 2D boolean numpy array (same shape as depth_map)
                                        where True indicates the object pixels.
        fx (float): Camera focal length in x (pixels).
        fy (float): Camera focal length in y (pixels).
        cx (float): Camera principal point x (pixels).
        cy (float): Camera principal point y (pixels).
        min_depth_m (Optional[float]): Minimum valid depth in METERS. Points below this are discarded.
        max_depth_m (Optional[float]): Maximum valid depth in METERS. Points above this are discarded.
        depth_scale_factor (float): Factor to multiply raw depth values by to convert them to millimeters.
                                    Defaults to 1.0 (assumes raw depth is already in mm).

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

    # Apply depth scale factor to convert raw depth values to millimeters
    depth_value_masked_mm = depth_value_masked.astype(np.float32) * depth_scale_factor

    # Convert min/max depth from meters (config) to millimeters (depth map unit)
    min_depth_mm = min_depth_m * 1000 if min_depth_m is not None else 0 # Default min to 0 if None
    max_depth_mm = max_depth_m * 1000 if max_depth_m is not None else np.inf # Default max to infinity if None

    # Filter out invalid depth values (e.g., <= 0) and values outside min/max range
    valid_depth_filter = (
        (depth_value_masked_mm > min_depth_mm) & 
        (depth_value_masked_mm < max_depth_mm)
    )

    ii_final = ii_masked[valid_depth_filter]
    jj_final = jj_masked[valid_depth_filter]
    depth_final = depth_value_masked_mm[valid_depth_filter]

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
        hull = ConvexHull(points, qhull_options="QJ") # Added qhull_options
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

def estimate_volume_from_mesh(file_path: str) -> float | None:
    """
    Estimates the volume of a 3D model from its mesh or point cloud file.
    Supports .obj, .ply, and other formats loadable by trimesh.
    For point clouds (e.g., .ply with only points), volume is calculated from their convex hull.

    Args:
        file_path (str): The absolute path to the 3D file.

    Returns:
        float | None: The calculated volume in cubic centimeters (cm^3).
                      Returns None if the file can't be loaded or volume calculation fails.
    """
    try:
        # Use load_path to handle scenes or single geometry files
        loaded_data = trimesh.load_path(file_path)

        geometry = None
        if isinstance(loaded_data, trimesh.Scene):
            if not loaded_data.geometry:
                logger.warning(f"No geometry found in scene file: {file_path}")
                return None
            # Heuristic: try to pick the 'main' geometry. 
            # This might need refinement based on dataset structure.
            # For now, pick the first one, prioritizing Trimesh objects with volume.
            for geom_obj in loaded_data.geometry.values():
                if isinstance(geom_obj, trimesh.Trimesh):
                    geometry = geom_obj
                    break
            if geometry is None: # If no Trimesh found, take the first geometry whatever it is
                 geometry = list(loaded_data.geometry.values())[0]
            logger.info(f"Extracted {type(geometry).__name__} from scene in {file_path}")
        elif isinstance(loaded_data, (trimesh.Trimesh, trimesh.points.PointCloud)):
            geometry = loaded_data # It's a single geometry object
        else:
            logger.warning(f"Loaded data from {file_path} is of an unhandled type: {type(loaded_data)}")
            return None

        if geometry is None:
            logger.error(f"Could not extract a valid geometry from {file_path}")
            return None

        logger.info(f"Processing {file_path} as type {type(geometry).__name__}")
        raw_volume_units = 0.0

        if isinstance(geometry, trimesh.Trimesh):
            logger.info(f"Processing as Trimesh object.")
            if not geometry.is_watertight:
                logger.info(f"Mesh from {file_path} is not watertight. Attempting to fill holes...")
                fill_holes(geometry) # Modifies the geometry in-place
                if geometry.is_watertight:
                    logger.info(f"Mesh from {file_path} is now watertight after filling holes.")
                else:
                    logger.warning(f"Mesh from {file_path} is still not watertight. Volume may be approximate.")
            else:
                logger.info(f"Mesh from {file_path} is already watertight.")
            raw_volume_units = geometry.volume
        
        elif isinstance(geometry, trimesh.points.PointCloud):
            logger.info(f"Processing as PointCloud object. Calculating volume from its convex hull.")
            if len(geometry.vertices) < 4:
                logger.warning(f"PointCloud from {file_path} has {len(geometry.vertices)} vertices. Need at least 4 for convex hull volume.")
                return None
            try:
                # The convex_hull of a PointCloud is a Trimesh object
                convex_hull_mesh = geometry.convex_hull
                raw_volume_units = convex_hull_mesh.volume
            except QhullError as e:
                logger.warning(f"QhullError computing convex hull for PointCloud {file_path}: {e}. This may occur for co-planar or degenerate points.")
                return None
            except Exception as e_cvx:
                logger.error(f"Unexpected error computing convex hull for PointCloud {file_path}: {e_cvx}", exc_info=True)
                return None
        else:
            logger.warning(f"Geometry from {file_path} is of an unsupported type for volume calculation: {type(geometry)}")
            return None

        # CRITICAL ASSUMPTION: Input 3D file units are in METERS.
        # If units are different (e.g., mm), this conversion factor will be incorrect.
        # 1 m^3 = 1,000,000 cm^3
        volume_cm3 = raw_volume_units * 1_000_000
        
        logger.info(f"Raw volume (source units, assumed m^3): {raw_volume_units}")
        logger.info(f"Calculated volume: {volume_cm3:.4f} cm^3 for {file_path}")
        return float(volume_cm3)

    except FileNotFoundError:
        logger.error(f"3D file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading or processing 3D file {file_path}: {e}", exc_info=True)
        return None