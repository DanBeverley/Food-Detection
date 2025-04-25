import numpy as np

def estimate_volume_from_depth(depth_map: np.ndarray, pixel_size_mm: float = 1.0) -> float:
    """
    Estimate volume from a depth map assuming a grid of pixels.
    Depth map should be a 2D array where each value represents depth in mm.
    
    Args:
        depth_map: 2D numpy array of depth values.
        pixel_size_mm: Size of each pixel in millimeters.
    
    Returns:
        Estimated volume in cubic millimeters.
    """
    if depth_map.ndim != 2:
        raise ValueError("Depth map must be 2D.")
    
    # Simple volume estimation by summing voxel volumes 
    height, width = depth_map.shape
    volume = np.sum(depth_map) * (pixel_size_mm ** 2) * pixel_size_mm  # Voxel volume calculation
    return volume

# Example usage
if __name__ == '__main__':
    sample_depth = np.array([[100, 100, 100], [100, 200, 100], [100, 100, 100]])
    vol = estimate_volume_from_depth(sample_depth)
    print(f"Estimated volume: {vol} mm³")
