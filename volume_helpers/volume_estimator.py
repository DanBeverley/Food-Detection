import numpy as np
import open3d as o3d
import logging

logger = logging.getLogger(__name__)

# --- Placeholder Camera Intrinsics ---
# These are VERY ROUGH ESTIMATES. Actual values depend heavily on the specific
# iPhone model, RevoPoint POP model, resolution, and API used.
# For MetaFood3D, if the resolution of depth maps is known (e.g., 640x480, 1280x720),
# these would need to be adjusted. Without official intrinsics from the dataset,
# volume estimation will have inherent inaccuracies.

# Camera intrinsics are now loaded from config files for better modularity
# This eliminates hardcoded camera parameters and improves robustness

def create_point_cloud_from_depth_mask(
    depth_map: np.ndarray,
    segmentation_mask: np.ndarray,
    camera_intrinsics: dict,
    depth_scale: float = 1000.0, # Converts depth map units to meters (e.g., if depth_map is in mm, scale is 1000.0)
    depth_trunc_m: float = 3.0,    # Max depth in meters to consider
    project_valid_depth_only: bool = True
) -> o3d.geometry.PointCloud:
    """
    Creates an Open3D PointCloud object from a depth map and a segmentation mask.

    Args:
        depth_map: Depth map (H, W), units are defined by depth_scale (e.g. millimeters).
        segmentation_mask: Binary segmentation mask (H, W), 1 for food, 0 for background.
        camera_intrinsics: Dict with 'fx', 'fy', 'cx', 'cy', 'width', 'height'.
        depth_scale: Factor to convert depth_map values to meters.
        depth_trunc_m: Maximum depth value (in meters) to consider.
        project_valid_depth_only: If True, only uses depth values > 0.

    Returns:
        o3d.geometry.PointCloud: The resulting point cloud of the segmented object.
    """
    if depth_map.shape != segmentation_mask.shape:
        raise ValueError("Depth map and segmentation mask must have the same dimensions.")
    
    img_height, img_width = depth_map.shape
    if img_width != camera_intrinsics['width'] or img_height != camera_intrinsics['height']:
        logger.warning(
            f"Depth map dimensions ({img_height}x{img_width}) do not match "
            f"camera_intrinsics dimensions ({camera_intrinsics['height']}x{camera_intrinsics['width']}). "
            f"Ensure intrinsics (fx, fy, cx, cy) are correctly scaled or chosen for the depth map resolution."
        )

    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']

    # --- Debugging: Mask and Depth Stats ---
    num_mask_pixels = np.count_nonzero(segmentation_mask)
    logger.info(f"PCD_CREATE_DEBUG: Number of non-zero pixels in segmentation_mask: {num_mask_pixels}")

    if num_mask_pixels > 0:
        depth_map_m = depth_map.astype(np.float32) / depth_scale # Convert to meters
        masked_depth_values_m = depth_map_m[segmentation_mask > 0]
        
        # Filter out non-positive depth values before stats, as they are invalid for point cloud generation
        valid_masked_depth_values_m = masked_depth_values_m[masked_depth_values_m > 0]
        
        if valid_masked_depth_values_m.size > 0:
            logger.info(f"PCD_CREATE_DEBUG: Scaled depth map (meters) stats (where mask > 0 and depth > 0): "
                        f"Min={np.min(valid_masked_depth_values_m):.4f}m, Max={np.max(valid_masked_depth_values_m):.4f}m, Mean={np.mean(valid_masked_depth_values_m):.4f}m")
        else:
            logger.info("PCD_CREATE_DEBUG: No valid positive depth values found where mask > 0.")
            # Also log stats for all masked depth values, even non-positive, to see raw values
            if masked_depth_values_m.size > 0:
                 logger.info(f"PCD_CREATE_DEBUG: Raw scaled depth map (meters) stats (where mask > 0, including non-positive): "
                             f"Min={np.min(masked_depth_values_m):.4f}m, Max={np.max(masked_depth_values_m):.4f}m, Mean={np.mean(masked_depth_values_m):.4f}m") 
            else:
                logger.info("PCD_CREATE_DEBUG: Mask is non-empty, but no depth values extracted under the mask (possibly all NaN or similar issue).")

    else: # num_mask_pixels is 0
        logger.info("PCD_CREATE_DEBUG: Segmentation mask is empty. No depth statistics to calculate.")

    logger.info(f"PCD_CREATE_DEBUG: Using depth_trunc_m (max depth): {depth_trunc_m:.4f}m")
    logger.info(f"PCD_CREATE_DEBUG: Using depth_scale: {depth_scale}")
    logger.info(f"PCD_CREATE_DEBUG: project_valid_depth_only: {project_valid_depth_only}")
    # --- End Debugging ---

    points = []
    for v_idx in range(img_height):  # y-coordinate
        for u_idx in range(img_width): # x-coordinate
            if segmentation_mask[v_idx, u_idx] > 0:  # If the pixel is part of the segmented food
                Z_raw = depth_map[v_idx, u_idx]
                if Z_raw <= 0 and project_valid_depth_only:
                    continue # Skip invalid depth pixels
                
                Z_m = Z_raw / depth_scale # Convert to meters
                if Z_m <= 0 or Z_m > depth_trunc_m: # Skip pixels too close, too far, or invalid after scaling
                    continue

                X_m = (u_idx - cx) * Z_m / fx
                Y_m = (v_idx - cy) * Z_m / fy
                points.append([X_m, Y_m, Z_m])

    if not points:
        logger.warning("No points generated for the point cloud. Check mask, depth values, or intrinsics.")
        return o3d.geometry.PointCloud()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    return pcd

def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    config: dict = None
) -> o3d.geometry.PointCloud:
    """
    Preprocesses a point cloud: downsampling and outlier removal.

    Args:
        pcd: Input Open3D PointCloud.
        config: Dictionary with preprocessing parameters:
            "downsample_voxel_size_m": Voxel size for downsampling (in meters). Skips if None/0.
            "outlier_removal_nb_neighbors": For statistical outlier removal.
            "outlier_removal_std_ratio": For statistical outlier removal.

    Returns:
        o3d.geometry.PointCloud: The preprocessed point cloud.
    """
    cfg = {
        "downsample_voxel_size_m": 0.005, # 0.5 cm
        "outlier_removal_nb_neighbors": 20,
        "outlier_removal_std_ratio": 2.0
    }
    if config: cfg.update(config)

    if not pcd.has_points():
        logger.warning("Input point cloud for preprocessing has no points.")
        return pcd

    pcd_processed = pcd
    if cfg["downsample_voxel_size_m"] and cfg["downsample_voxel_size_m"] > 0:
        pcd_processed = pcd.voxel_down_sample(voxel_size=cfg["downsample_voxel_size_m"])
        logger.info(f"Point cloud downsampled from {len(pcd.points)} to {len(pcd_processed.points)} points.")
    else:
        logger.info("Skipping point cloud downsampling.")

    if not pcd_processed.has_points():
        logger.warning("Point cloud has no points after downsampling.")
        return pcd_processed

    cl, ind = pcd_processed.remove_statistical_outlier(
        nb_neighbors=cfg["outlier_removal_nb_neighbors"],
        std_ratio=cfg["outlier_removal_std_ratio"]
    )
    pcd_final = pcd_processed.select_by_index(ind)
    removed_count = len(pcd_processed.points) - len(pcd_final.points)
    if removed_count > 0:
        logger.info(f"Statistical outlier removal: {removed_count} points removed.")
    
    if not pcd_final.has_points():
        logger.warning("Point cloud has no points after outlier removal.")
    
    return pcd_final

def estimate_volume_voxel_grid(
    pcd: o3d.geometry.PointCloud,
    config: dict = None
) -> float:
    """
    Estimates volume of a point cloud using voxelization.

    Args:
        pcd: Input Open3D PointCloud.
        config: Dictionary with volume estimation parameters:
            "volume_voxel_size_m": The size of each voxel in meters.

    Returns:
        float: Estimated volume in cubic meters. Returns 0.0 if estimation fails.
    """
    cfg = {
        "volume_voxel_size_m": 0.005 # 0.5 cm
    }
    if config: cfg.update(config)

    if not pcd.has_points():
        logger.warning("Cannot estimate volume from an empty point cloud (voxel method).")
        return 0.0
        
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=cfg["volume_voxel_size_m"])
    num_voxels = len(voxel_grid.get_voxels())
    volume_m3 = num_voxels * (cfg["volume_voxel_size_m"] ** 3)
    logger.info(f"Voxel volume estimation: {num_voxels} voxels, voxel_size={cfg['volume_voxel_size_m']}m, volume={volume_m3:.6f} m^3")
    return volume_m3

def estimate_volume_from_depth(
    depth_map: np.ndarray, # Expected in millimeters or units defined by depth_scale
    segmentation_mask: np.ndarray,
    camera_intrinsics_key: str = 'default', # Key to select intrinsics
    custom_intrinsics: dict = None, # Allow passing a full intrinsics dict directly
    all_camera_intrinsics: dict = None, # Dictionary containing all known intrinsics from config
    config: dict = None
) -> float:
    """
    Main wrapper function to estimate volume from depth map and segmentation mask.

    Args:
        depth_map: (H, W) np.ndarray, depth values (e.g., in mm).
        segmentation_mask: (H, W) np.ndarray, binary mask (0 or 1).
        camera_intrinsics_key: String key (e.g., '640x480', '1920x1440', 'default') to select
                                 intrinsics. Also used with 'custom' if custom_intrinsics is provided.
        custom_intrinsics: A dict with {'width', 'height', 'fx', 'fy', 'cx', 'cy'}.
                           Takes precedence if provided.
        all_camera_intrinsics: A dictionary mapping keys (like '1920x1440') to their
                                 full intrinsics dictionaries. Used if custom_intrinsics is not set.
        config: Dictionary for overriding default processing parameters:
            "depth_scale": float (e.g., 1000.0 for mm to m)
            "depth_trunc_m": float (max depth in meters)
            "project_valid_depth_only": bool
            "downsample_voxel_size_m": float (preprocessing)
            "outlier_removal_nb_neighbors": int
            "outlier_removal_std_ratio": float
            "volume_voxel_size_m": float (for volume calculation)

    Returns:
        float: Estimated volume in cubic centimeters (cm^3). Returns 0.0 if fails.
    """
    # Use volume processing params from pipeline config for better modularity
    cfg = config if config else {
        "depth_scale": 1000.0,
        "depth_trunc_m": 3.0,
        "project_valid_depth_only": True,
        "downsample_voxel_size_m": 0.005,
        "outlier_removal_nb_neighbors": 20,
        "outlier_removal_std_ratio": 2.0,
        "volume_voxel_size_m": 0.005
    }

    # --- Intrinsics Debug Logging --- 
    logger.info(f"VOLUME_ESTIMATOR_DEBUG: Received camera_intrinsics_key: '{camera_intrinsics_key}'")
    if custom_intrinsics is not None:
        logger.info(f"VOLUME_ESTIMATOR_DEBUG: custom_intrinsics type: {type(custom_intrinsics)}, keys: {list(custom_intrinsics.keys()) if isinstance(custom_intrinsics, dict) else 'N/A'}")
    else:
        logger.info("VOLUME_ESTIMATOR_DEBUG: custom_intrinsics is None")
    if all_camera_intrinsics is not None:
        logger.info(f"VOLUME_ESTIMATOR_DEBUG: all_camera_intrinsics type: {type(all_camera_intrinsics)}, keys: {list(all_camera_intrinsics.keys()) if isinstance(all_camera_intrinsics, dict) else 'N/A'}")
    else:
        logger.info("VOLUME_ESTIMATOR_DEBUG: all_camera_intrinsics is None")
    # --- End Intrinsics Debug Logging ---

    # Select camera intrinsics with improved logic
    selected_intrinsics = None
    source_of_intrinsics = "None"

    if custom_intrinsics and isinstance(custom_intrinsics, dict) and all(k in custom_intrinsics for k in ['width', 'height', 'fx', 'fy', 'cx', 'cy']):
        selected_intrinsics = custom_intrinsics
        source_of_intrinsics = f"custom_intrinsics provided directly for key '{camera_intrinsics_key}'"
    elif all_camera_intrinsics and isinstance(all_camera_intrinsics, dict) and camera_intrinsics_key in all_camera_intrinsics:
        selected_intrinsics = all_camera_intrinsics[camera_intrinsics_key]
        source_of_intrinsics = f"all_camera_intrinsics using key '{camera_intrinsics_key}'"
    elif camera_intrinsics_key == '640x480' and 'CAMERA_INTRINSICS_640_480' in globals():
        selected_intrinsics = CAMERA_INTRINSICS_640_480
        source_of_intrinsics = "local CAMERA_INTRINSICS_640_480"
    elif camera_intrinsics_key == '1280x720' and 'CAMERA_INTRINSICS_1280_720' in globals():
        selected_intrinsics = CAMERA_INTRINSICS_1280_720
        source_of_intrinsics = "local CAMERA_INTRINSICS_1280_720"
    
    if selected_intrinsics:
        logger.info(f"Using camera intrinsics from: {source_of_intrinsics}")
        logger.info(f"Selected intrinsics details: w={selected_intrinsics['width']}, h={selected_intrinsics['height']}, fx={selected_intrinsics['fx']}")
    else:
        selected_intrinsics = DEFAULT_CAMERA_INTRINSICS # Fallback to default
        logger.warning(
            f"Camera intrinsics key '{camera_intrinsics_key}' not found in custom_intrinsics, all_camera_intrinsics, or local predefined. "
            f"Falling back to DEFAULT_CAMERA_INTRINSICS ({DEFAULT_CAMERA_INTRINSICS.get('width', 'N/A')}x{DEFAULT_CAMERA_INTRINSICS.get('height', 'N/A')}). "
            f"This may lead to inaccurate volume estimation."
        )

    # 1. Create Point Cloud
    pcd = create_point_cloud_from_depth_mask(
        depth_map,
        segmentation_mask,
        selected_intrinsics,
        depth_scale=cfg["depth_scale"],
        depth_trunc_m=cfg["depth_trunc_m"],
        project_valid_depth_only=cfg["project_valid_depth_only"]
    )
    if not pcd.has_points():
        logger.error("Point cloud creation failed or resulted in no points.")
        return 0.0
    logger.info(f"Initial point cloud created with {len(pcd.points)} points.")

    # 2. Preprocess Point Cloud
    pcd_processed = preprocess_point_cloud(pcd, config=cfg)
    if not pcd_processed.has_points():
        logger.error("Point cloud preprocessing resulted in no points.")
        return 0.0
    logger.info(f"Processed point cloud has {len(pcd_processed.points)} points.")

    # 3. Estimate Volume
    volume_m3 = estimate_volume_voxel_grid(pcd_processed, config=cfg)

    # Convert volume from cubic meters to cubic centimeters (1 m^3 = 1,000,000 cm^3)
    volume_cm3 = volume_m3 * 1_000_000
    logger.info(f"Final Estimated volume: {volume_cm3:.2f} cm^3 ({volume_m3:.6f} m^3)")

    return volume_cm3

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing volume_estimator.py...")

    # Create dummy data based on a common resolution
    test_intrinsics_key = '1440x1920'
    intr = CAMERA_INTRINSICS_640_480
    img_h, img_w = intr['height'], intr['width']
    
    # Dummy depth map (e.g., a slanted plane from 500mm to 1000mm)
    # Depth values are in mm, matching typical sensor output
    dummy_depth_mm = np.fromfunction(
        lambda y, x: 500 + (x + y) * (500 / (img_h + img_w)), 
        (img_h, img_w), dtype=np.float32
    )
    
    # Dummy segmentation mask (e.g., a central square)
    dummy_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    dummy_mask[img_h//4 : 3*img_h//4, img_w//4 : 3*img_w//4] = 1

    test_config = {
        "depth_scale": 1000.0, # Input depth is mm
        "volume_voxel_size_m": 0.01, # 1cm voxel size for volume calculation
        "downsample_voxel_size_m": 0.005 # 0.5cm for preprocessing downsample
    }

    estimated_volume_cm3 = estimate_volume_from_depth(
        dummy_depth_mm, 
        dummy_mask, 
        camera_intrinsics_key=test_intrinsics_key,
        config=test_config
    )
    print(f"TEST - Estimated Volume: {estimated_volume_cm3:.2f} cm^3")

    # --- Visualize the point cloud (optional, requires Open3D visualization) ---
    # To run this part, you might need to be in an environment where Open3D can open a window.
    # print("Attempting to visualize point cloud from dummy data...")
    # selected_intrinsics_viz = CAMERA_INTRINSICS_640_480
    # pcd_test = create_point_cloud_from_depth_mask(
    #     dummy_depth_mm, dummy_mask, selected_intrinsics_viz,
    #     depth_scale=test_config["depth_scale"]
    # )
    # if pcd_test.has_points():
    #     pcd_processed_test = preprocess_point_cloud(pcd_test, config=test_config)
    #     if pcd_processed_test.has_points():
    #         print("Visualizing processed point cloud...")
    #         o3d.visualization.draw_geometries([pcd_processed_test])
    #
    #         print("Visualizing voxel grid...")
    #         voxel_grid_viz = o3d.geometry.VoxelGrid.create_from_point_cloud(
    #             pcd_processed_test, voxel_size=test_config["volume_voxel_size_m"]
    #         )
    #         o3d.visualization.draw_geometries([voxel_grid_viz])
    #     else:
    #         print("Processed point cloud has no points, skipping visualization.")
    # else:
    #     print("Initial point cloud has no points, skipping visualization.")