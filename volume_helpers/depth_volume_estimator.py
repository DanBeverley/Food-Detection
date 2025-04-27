import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError # For catching errors

def estimate_volume_from_depth(depth_map: np.ndarray, segmentation_mask: np.ndarray, pixel_area_mm2: float = 1.0) -> float:
    """
    Estimate volume from a depth map using a segmentation mask (Simple Depth Summation Method).
    Depth map should be a 2D array where each value represents depth in mm.
    Segmentation mask should be a 2D boolean array of the same shape,
    where True indicates the pixel belongs to the object of interest.

    Args:
        depth_map: 2D numpy array of depth values (in mm).
        segmentation_mask: 2D boolean numpy array identifying object pixels.
        pixel_area_mm2: The area covered by each pixel in square millimeters (e.g., pixel_width_mm * pixel_height_mm).

    Returns:
        Estimated volume in cubic millimeters (mm³).
    """
    if depth_map.ndim != 2:
        raise ValueError("Depth map must be 2D.")
    if segmentation_mask.ndim != 2:
        raise ValueError("Segmentation mask must be 2D.")
    if depth_map.shape != segmentation_mask.shape:
        raise ValueError("Depth map and segmentation mask must have the same shape.")
    if segmentation_mask.dtype != bool:
        # Try converting if it's integer type (0s and 1s)
        if np.issubdtype(segmentation_mask.dtype, np.integer):
            print("Warning: Segmentation mask is integer type, converting to boolean (assuming 0=False, non-zero=True).")
            segmentation_mask = segmentation_mask.astype(bool)
        else:
            raise ValueError("Segmentation mask must be boolean (or integer convertible to boolean).")

    # Select depth values corresponding to the segmented object
    object_depths = depth_map[segmentation_mask]

    # Remove invalid depth values (e.g., zero or negative, depending on sensor)
    # Assuming valid depths are positive
    valid_object_depths = object_depths[object_depths > 0]

    if valid_object_depths.size == 0:
        print("Warning: No valid depth pixels found within the segmentation mask.")
        return 0.0

    # Estimate volume by summing the volume of each pixel's column within the mask
    # Volume = sum(pixel_depth * pixel_area)
    volume_mm3 = np.sum(valid_object_depths) * pixel_area_mm2

    return volume_mm3

def estimate_volume_point_cloud(depth_map: np.ndarray, segmentation_mask: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> float | None:
    """
    Estimate volume from a depth map using point cloud convex hull.
    Requires camera intrinsic parameters to convert depth map to point cloud.

    Args:
        depth_map: 2D numpy array of depth values (in mm).
        segmentation_mask: 2D boolean numpy array identifying object pixels.
        fx: Camera focal length in x (pixels).
        fy: Camera focal length in y (pixels).
        cx: Camera principal point x (pixels).
        cy: Camera principal point y (pixels).

    Returns:
        Estimated volume in cubic millimeters (mm³) or None if calculation fails.
    """
    if depth_map.ndim != 2 or segmentation_mask.ndim != 2 or depth_map.shape != segmentation_mask.shape:
        raise ValueError("Invalid input shapes or dimensions for depth map or mask.")
    if segmentation_mask.dtype != bool:
        if np.issubdtype(segmentation_mask.dtype, np.integer):
            segmentation_mask = segmentation_mask.astype(bool)
        else:
            raise ValueError("Segmentation mask must be boolean.")

    height, width = depth_map.shape

    # Create pixel coordinate grid
    jj, ii = np.meshgrid(np.arange(width), np.arange(height))

    # Select only pixels within the mask
    mask_indices = np.where(segmentation_mask)
    ii = ii[mask_indices]
    jj = jj[mask_indices]
    depth_values = depth_map[mask_indices]

    # Filter out invalid depth values (e.g., zero or negative)
    valid_depth_indices = depth_values > 0
    ii = ii[valid_depth_indices]
    jj = jj[valid_depth_indices]
    depth_values = depth_values[valid_depth_indices]

    if depth_values.size < 4: # Convex hull requires at least 4 points
        print("Warning: Not enough valid points in the mask to form a 3D shape for convex hull.")
        return None

    # Convert pixel coordinates and depth to 3D points (in mm)
    x = (jj - cx) * depth_values / fx
    y = (ii - cy) * depth_values / fy
    z = depth_values

    # Stack points into an N x 3 array
    points = np.stack((x, y, z), axis=-1)

    try:
        # Calculate convex hull
        hull = ConvexHull(points)
        volume_mm3 = hull.volume
        return volume_mm3
    except QhullError as e:
        print(f"Warning: Convex hull calculation failed: {e}. Points might be degenerate (e.g., co-planar).")
        # Fallback or return None
        return None
    except Exception as e:
        print(f"Warning: An unexpected error occurred during convex hull volume calculation: {e}")
        return None


# Example usage
if __name__ == '__main__':
    # Sample depth map (mm)
    sample_depth = np.array([
        [50, 60, 0],
        [70, 80, 70],
        [0, 75, 65]
    ])

    # Sample mask (True indicates the object)
    sample_mask = np.array([
        [False, True, False],
        [True, True, True],
        [False, True, True]
    ])

    # Assume 1mm x 1mm pixels
    pixel_area = 1.0 * 1.0 # mm²

    try:
        vol = estimate_volume_from_depth(sample_depth, sample_mask, pixel_area)
        print(f"Estimated volume: {vol:.2f} mm³") # Expected: (60+70+80+70+75+65)*1.0 = 420.00

        # Test with mask covering no valid pixels
        mask_no_valid = np.array([
            [True, False, True],
            [False, False, False],
            [True, False, False]
        ], dtype=bool) # Ensure boolean type
        vol_invalid = estimate_volume_from_depth(sample_depth, mask_no_valid, pixel_area)
        print(f"Estimated volume (invalid mask): {vol_invalid:.2f} mm³") # Expected: (50)*1.0 = 50.00

        # Test with all zero depth within mask
        zero_depth = np.zeros_like(sample_depth)
        vol_zero = estimate_volume_from_depth(zero_depth, sample_mask, pixel_area)
        print(f"Estimated volume (zero depth): {vol_zero:.2f} mm³") # Expected: 0.00

    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Point Cloud Volume Example ---")
    # Example camera intrinsics (replace with actual values)
    fx, fy, cx, cy = 500.0, 500.0, sample_depth.shape[1] / 2, sample_depth.shape[0] / 2

    # Use the same sample data as before
    try:
        vol_pc = estimate_volume_point_cloud(sample_depth, sample_mask.astype(bool), fx, fy, cx, cy)
        if vol_pc is not None:
            print(f"Estimated volume (Point Cloud Convex Hull): {vol_pc:.2f} mm³")
        else:
            print("Point Cloud volume calculation failed.")

        # Example with co-planar points (should fail)
        flat_depth = np.ones((5, 5)) * 100
        flat_mask = np.ones((5, 5), dtype=bool)
        fx_flat, fy_flat, cx_flat, cy_flat = 500.0, 500.0, 2.0, 2.0
        print("\nTesting with likely co-planar points:")
        vol_flat = estimate_volume_point_cloud(flat_depth, flat_mask, fx_flat, fy_flat, cx_flat, cy_flat)
        if vol_flat is None:
            print("Convex hull failed as expected for co-planar points.")
        else:
            print(f"Convex hull succeeded unexpectedly? Volume: {vol_flat:.2f} mm³")

    except ValueError as e:
        print(f"Error: {e}")
    except ImportError:
        print("Error: SciPy is required for point cloud volume estimation. Please install it (`pip install scipy`).")
