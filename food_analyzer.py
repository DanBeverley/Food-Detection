import numpy as np
import os
import sys

# Ensure the helper modules can be imported
# Add the project root to the Python path if running this script directly
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from volume_helpers.depth_volume_estimator import estimate_volume_from_depth
from volume_helpers.density_lookup import lookup_density


def analyze_food_item(food_name: str, depth_map: np.ndarray, pixel_size_mm: float = 1.0) -> dict:
    """
    Analyzes a food item by estimating volume, looking up density, and calculating mass.

    Args:
        food_name: The name of the food item (output from classification).
        depth_map: 2D numpy array representing depth data (e.g., from LiDAR/ToF).
        pixel_size_mm: The size of each pixel in the depth map in millimeters.

    Returns:
        A dictionary containing the results: volume, density, mass, and status.
    """
    results = {
        'food_name': food_name,
        'volume_mm3': None,
        'density_g_cm3': None,
        'mass_g': None,
        'status': 'Processing failed'
    }

    try:
        # 1. Estimate Volume
        volume_mm3 = estimate_volume_from_depth(depth_map, pixel_size_mm)
        # Convert volume to cm³ (1 cm³ = 1000 mm³)
        volume_cm3 = volume_mm3 / 1000.0
        results['volume_mm3'] = volume_mm3
        print(f"Estimated Volume for {food_name}: {volume_cm3:.2f} cm³ ({volume_mm3:.2f} mm³)")

        # 2. Lookup Density (g/cm³)
        # This will check cache, custom DB, and then API (updating custom DB if new API result found)
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

    # --- Simulation Inputs ---

    # Simulate classification output
    classified_food = "apple" 

    # Simulate depth map data (e.g., from a sensor)
    # Example: A simple 3x3 map representing an object
    # (values are depth in millimeters)
    simulated_depth_map = np.array([
        [0, 50, 0],
        [50, 100, 50],
        [0, 50, 0]
    ]) * 1.0 # Ensure float

    # Assume each pixel corresponds to 1mm x 1mm area on the sensor plane
    simulated_pixel_size = 1.0 # mm

    print(f"\nAnalyzing item: '{classified_food}'")
    analysis_result = analyze_food_item(
        food_name=classified_food,
        depth_map=simulated_depth_map,
        pixel_size_mm=simulated_pixel_size
    )

    print(f"\n--- Analysis Complete for '{classified_food}' --- ")
    print(f"Status: {analysis_result['status']}")
    if analysis_result['volume_mm3'] is not None:
        print(f"Volume: {analysis_result['volume_mm3'] / 1000.0:.2f} cm³")
    if analysis_result['density_g_cm3'] is not None:
        print(f"Density: {analysis_result['density_g_cm3']:.2f} g/cm³")
    if analysis_result['mass_g'] is not None:
        print(f"Mass: {analysis_result['mass_g']:.2f} g")

    
    classified_food_2 = "cheddar cheese"
    simulated_depth_map_2 = np.array([
        [60, 70, 60],
        [70, 80, 70],
        [60, 70, 60]
    ]) * 1.0

    print(f"\nAnalyzing item: '{classified_food_2}'")
    
    analysis_result_2 = analyze_food_item(
        food_name=classified_food_2,
        depth_map=simulated_depth_map_2,
        pixel_size_mm=simulated_pixel_size
    )

    print(f"\n--- Analysis Complete for '{classified_food_2}' --- ")
    print(f"Status: {analysis_result_2['status']}")
    if analysis_result_2['volume_mm3'] is not None:
        print(f"Volume: {analysis_result_2['volume_mm3'] / 1000.0:.2f} cm³")
    if analysis_result_2['density_g_cm3'] is not None:
        print(f"Density: {analysis_result_2['density_g_cm3']:.2f} g/cm³")
    if analysis_result_2['mass_g'] is not None:
        print(f"Mass: {analysis_result_2['mass_g']:.2f} g")

    print("\nCheck 'data/databases/custom_density_db.json' to see if new items were added.")
