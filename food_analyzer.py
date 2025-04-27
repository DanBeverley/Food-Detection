import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from volume_helpers.depth_volume_estimator import estimate_volume_from_depth, estimate_volume_point_cloud
from volume_helpers.density_lookup import lookup_density


def analyze_food_item(food_name: str, depth_map: np.ndarray, segmentation_mask: np.ndarray, 
                      fx: float, fy: float, cx: float, cy: float, 
                      pixel_area_mm2: float = 1.0) -> dict:
    """
    Analyzes a food item by estimating volume, looking up density, and calculating mass.
    Uses a segmentation mask and attempts point cloud estimation first, falling back to depth summation.

    Args:
        food_name: The name of the food item (output from classification).
        depth_map: 2D numpy array representing depth data (e.g., from LiDAR/ToF, in mm).
        segmentation_mask: 2D boolean numpy array identifying food pixels in the depth map.
        fx, fy, cx, cy: Camera intrinsic parameters.
        pixel_area_mm2: The area of each pixel (used for fallback volume estimation).

    Returns:
        A dictionary containing the results: volume, density, mass, volume_method, and status.
    """
    results = {
        'food_name': food_name,
        'volume_mm3': None,
        'volume_method': None, # 'PointCloud' or 'DepthSummation'
        'density_g_cm3': None,
        'mass_g': None,
        'status': 'Processing failed'
    }

    try:
        # --- 1. Estimate Volume --- 
        volume_mm3 = None
        volume_method = None

        # Try Point Cloud method first
        try:
            pc_volume_mm3 = estimate_volume_point_cloud(depth_map, segmentation_mask, fx, fy, cx, cy)
            if pc_volume_mm3 is not None and pc_volume_mm3 > 0:
                volume_mm3 = pc_volume_mm3
                volume_method = 'PointCloud'
                print(f"Volume estimated using Point Cloud Convex Hull.")
            else:
                print("Point cloud volume estimation did not yield a valid positive volume.")
        except ValueError as pc_ve:
            print(f"Point cloud volume estimation error: {pc_ve}")
        except ImportError:
             print("Error: SciPy is required for point cloud volume estimation.")
             # Decide if you want to stop or just fallback
             # For now, we'll let it fall back
        except Exception as pc_e: # Catch other potential errors from point cloud method
            print(f"Unexpected error during point cloud volume estimation: {pc_e}")

        # Fallback to simple depth summation if point cloud method failed
        if volume_mm3 is None:
            print("Falling back to simple depth summation volume estimation.")
            volume_mm3 = estimate_volume_from_depth(depth_map, segmentation_mask, pixel_area_mm2)
            if volume_mm3 > 0:
                volume_method = 'DepthSummation'
            else:
                 print("Warning: Depth summation method also resulted in zero or negative volume.")
                 volume_mm3 = None # Ensure it's None if calculation fails

        if volume_mm3 is None:
            results['status'] = 'Volume Estimation Failed'
            print("Error: Both volume estimation methods failed.")
            return results # Cannot proceed without volume

        results['volume_mm3'] = volume_mm3
        results['volume_method'] = volume_method
        
        # Convert volume to cm³ (1 cm³ = 1000 mm³)
        volume_cm3 = volume_mm3 / 1000.0
        print(f"Estimated Volume for {food_name} ({volume_method}): {volume_cm3:.2f} cm³ ({volume_mm3:.2f} mm³)")

        # --- 2. Lookup Density (g/cm³) ---
        api_key = os.environ.get("USDA_API_KEY")
        if not api_key:
            print("Warning: USDA_API_KEY environment variable not set. USDA API lookups will be skipped.")

        density_g_cm3 = lookup_density(food_name, api_key=api_key)
        results['density_g_cm3'] = density_g_cm3

        if density_g_cm3 is not None:
            print(f"Density for {food_name}: {density_g_cm3:.2f} g/cm³")

            # 3. Calculate Mass (Mass = Density * Volume)
            mass_g = density_g_cm3 * volume_cm3
            results['mass_g'] = mass_g
            results['status'] = 'Success'
            print(f"Calculated Mass for {food_name}: {mass_g:.2f} g")
        else:
            print(f"Density not found for {food_name}. Cannot calculate mass.")
            results['status'] = 'Success (Volume only)'

    except ValueError as ve:
        print(f"Error processing {food_name}: {ve}")
        results['status'] = f"Error: {ve}"
    except Exception as e:
        print(f"An unexpected error occurred for {food_name}: {e}")
        results['status'] = f"Unexpected Error: {e}"

    return results


if __name__ == '__main__':
    print("--- Food Analyzer Simulation ---")

    # Simulate classification output
    classified_food = "apple" # Try changing this to 'orange' or 'cooked white rice'

    # Simulate depth map data (e.g., from a sensor, in mm)
    simulated_depth_map = np.array([
        [0, 50, 0],
        [50, 100, 50],
        [0, 50, 0]
    ]) * 1.0 # Ensure float

    # Simulate segmentation mask (True where the food item is)
    # This would normally come from your segmentation model
    simulated_mask = np.array([
        [False, True, False],
        [True, True, True],
        [False, True, False]
    ])

    # Assume 1mm x 1mm pixel area
    simulated_pixel_area = 1.0 * 1.0 # mm²

    # Placeholder Camera Intrinsics (MUST be replaced with actual values from your sensor)
    # Using values typical for a phone camera perhaps
    H, W = simulated_depth_map.shape
    sim_fx, sim_fy = 600.0, 600.0 # Example focal lengths in pixels
    sim_cx, sim_cy = W / 2, H / 2 # Example principal point (image center)

    print(f"\nAnalyzing item: '{classified_food}'")
    # Run Analysis
    analysis_result = analyze_food_item(
        food_name=classified_food,
        depth_map=simulated_depth_map,
        segmentation_mask=simulated_mask,
        fx=sim_fx, fy=sim_fy, cx=sim_cx, cy=sim_cy, # Pass intrinsics
        pixel_area_mm2=simulated_pixel_area # Still needed for fallback
    )

    print(f"\n--- Analysis Complete for '{classified_food}' --- ")
    print(f"Status: {analysis_result['status']}")
    if analysis_result['volume_method']:
        print(f"Volume Method: {analysis_result['volume_method']}")
    if analysis_result['volume_mm3'] is not None:
        print(f"Volume: {analysis_result['volume_mm3'] / 1000.0:.2f} cm³ ({analysis_result['volume_mm3']:.2f} mm³)")
    if analysis_result['density_g_cm3'] is not None:
        print(f"Density: {analysis_result['density_g_cm3']:.2f} g/cm³")
    if analysis_result['mass_g'] is not None:
        print(f"Mass: {analysis_result['mass_g']:.2f} g")

    # Example with a food likely not in custom DB initially
    classified_food_2 = "cheddar cheese"
    simulated_depth_map_2 = np.array([
        [0, 0, 0, 0],
        [60, 70, 60, 0],
        [70, 80, 70, 0],
        [60, 70, 60, 0],
        [0, 0, 0, 0]
    ]) * 1.0

    # Simulate mask for the second item
    simulated_mask_2 = np.array([
        [False, False, False, False],
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, False],
        [False, False, False, False]
    ])

    # Use same intrinsics for simplicity in example
    H2, W2 = simulated_depth_map_2.shape
    sim_fx2, sim_fy2 = 600.0, 600.0
    sim_cx2, sim_cy2 = W2 / 2, H2 / 2

    print(f"\nAnalyzing item: '{classified_food_2}'")
    # Run analysis for the second food item
    analysis_result_2 = analyze_food_item(
        food_name=classified_food_2,
        depth_map=simulated_depth_map_2,
        segmentation_mask=simulated_mask_2,
        fx=sim_fx2, fy=sim_fy2, cx=sim_cx2, cy=sim_cy2,
        pixel_area_mm2=simulated_pixel_area # For fallback
    )

    print(f"\n--- Analysis Complete for '{classified_food_2}' --- ")
    print(f"Status: {analysis_result_2['status']}")
    if analysis_result_2['volume_method']:
        print(f"Volume Method: {analysis_result_2['volume_method']}")
    if analysis_result_2['volume_mm3'] is not None:
        print(f"Volume: {analysis_result_2['volume_mm3'] / 1000.0:.2f} cm³ ({analysis_result_2['volume_mm3']:.2f} mm³)")
    if analysis_result_2['density_g_cm3'] is not None:
        print(f"Density: {analysis_result_2['density_g_cm3']:.2f} g/cm³")
    if analysis_result_2['mass_g'] is not None:
        print(f"Mass: {analysis_result_2['mass_g']:.2f} g")

    print("\nCheck 'data/databases/custom_density_db.json' to see if new items were added.")
