import numpy as np
import os
import logging
import yaml
from pathlib import Path
import time 

try:
    from models.segmentation.predict_segmentation import run_segmentation_inference, load_segmentation_model
except ImportError:
    logging.error("Failed to import segmentation functions. Ensure predict_segmentation.py is refactored.")
    def load_segmentation_model(path): raise NotImplementedError("Segmentation model loading not implemented/imported.")
    def run_segmentation_inference(*args): raise NotImplementedError("Segmentation inference not implemented/imported.")

try:
    from models.classification.predict_classification import run_classification_inference, load_classification_model
except ImportError:
    logging.error("Failed to import classification functions. Ensure predict_classification.py is refactored.")
    def load_classification_model(path): raise NotImplementedError("Classification model loading not implemented/imported.")
    def run_classification_inference(*args): raise NotImplementedError("Classification inference not implemented/imported.")

try:
    from volume_helpers.volume_helpers import depth_map_to_masked_points, estimate_volume_convex_hull
    from volume_helpers.density_lookup import lookup_density
except ImportError as e:
     logging.error(f"Failed to import local helpers: {e}. Ensure volume_helpers and density_lookup exist.")
     def depth_map_to_masked_points(*args): raise NotImplementedError("depth_map_to_masked_points not implemented/imported.")
     def estimate_volume_convex_hull(*args): raise NotImplementedError("estimate_volume_convex_hull not implemented/imported.")
     def lookup_density(*args): raise NotImplementedError("lookup_density not implemented/imported.")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _get_project_root() -> Path:
    """Gets the project root directory."""
    return Path(__file__).parent

def load_pipeline_config(config_path: str) -> dict:
    """Loads pipeline configuration."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(_get_project_root(), config_path)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded pipeline configuration from: {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Pipeline configuration file not found: {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading or parsing pipeline configuration {config_path}: {e}")
        raise

# Core Analysis Function

def analyze_food_item(
    image_path: str,
    depth_map_path: str,
    config: dict,
    usda_api_key: str = None
) -> dict | None:
    """
    Runs the full food analysis pipeline: segmentation, classification, volume, density, mass.

    Args:
        image_path (str): Path to the input RGB image.
        depth_map_path (str): Path to the corresponding depth map (.npy format assumed).
        config (dict): Loaded pipeline configuration dictionary.
        usda_api_key (str, optional): USDA API key for density lookup fallback. Defaults to None.

    Returns:
        dict | None: A dictionary containing analysis results ('food_label', 'confidence',
                     'volume_cm3', 'density_g_cm3', 'estimated_mass_g', 'segmentation_mask_shape'),
                     or None if a critical step fails.
    """
    results = {}
    timing = {} # For simple profiling
    project_root = _get_project_root()
    start_time = time.time()

    #1. Load Input Data 
    try:
        t0 = time.time()
        # Depth map loaded here, image loading might happen inside inference functions
        depth_map = np.load(depth_map_path)
        logging.info(f"Loaded depth map from {depth_map_path}, shape: {depth_map.shape}")
        # Basic validation - Assuming depth in mm
        if depth_map.ndim != 2:
             logging.error("Depth map must be a 2D NumPy array.")
             return None
        timing['load_depth'] = time.time() - t0
    except FileNotFoundError:
        logging.error(f"Depth map file not found: {depth_map_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading depth map {depth_map_path}: {e}")
        return None

    #2. Load Models (Consider loading once if analyzing multiple items)
    # This part might be better outside this function if running in a loop
    try:
        t0 = time.time()
        seg_model_path = os.path.join(project_root, config['models']['segmentation_tflite'])
        # Assume load_segmentation_model returns the interpreter & input/output details
        seg_interpreter, seg_input_details, seg_output_details = load_segmentation_model(seg_model_path)

        clf_model_path = os.path.join(project_root, config['models']['classification_tflite'])
        # Assume load_classification_model returns interpreter, details, and class labels
        clf_interpreter, clf_input_details, clf_output_details, class_labels = load_classification_model(clf_model_path)
        logging.info("Loaded segmentation and classification models.")
        timing['load_models'] = time.time() - t0
    except FileNotFoundError as e:
         logging.error(f"Model file not found: {e}. Check paths in config_pipeline.yaml.")
         return None
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        return None
    try:
        t0 = time.time()
        # Read segmentation params from config
        seg_input_size = tuple(config['model_params']['segmentation_input_size'])
        seg_num_classes = config['model_params'].get('segmentation_num_classes', 2) # Default to 2 if missing
        
        segmentation_mask = run_segmentation_inference(
            seg_interpreter, seg_input_details, seg_output_details,
            image_path, seg_input_size, depth_map.shape, # Pass depth map shape for resizing mask
            num_classes=seg_num_classes # Pass num_classes
        )
        if segmentation_mask is None:
            logging.error("Segmentation failed.")
            return None
        # Ensure mask is boolean
        segmentation_mask = segmentation_mask.astype(bool)
        # Avoid storing large mask in results dict, just store shape for info
        results['segmentation_mask_shape'] = segmentation_mask.shape
        logging.info(f"Segmentation successful. Mask shape: {segmentation_mask.shape}")
        timing['segmentation'] = time.time() - t0
    except Exception as e:
        logging.exception(f"Error during segmentation: {e}") # Log traceback
        return None

    # 4. Run Classification 
    try:
        t0 = time.time()
        #  run_classification_inference takes interpreter, details, image_path, target_size, labels
        # and returns the top class label and confidence score
        target_clf_size = tuple(config['model_params']['classification_input_size'])
        food_label, confidence = run_classification_inference(
            clf_interpreter, clf_input_details, clf_output_details,
            image_path, target_clf_size, class_labels
        )
        if food_label is None:
            logging.error("Classification failed.")
            
            return None
        # Ensure label is JSON serializable (str or int)
        results['food_label'] = int(food_label) if isinstance(food_label, (np.integer, int)) else food_label
        results['confidence'] = float(confidence) # Ensure float
        logging.info(f"Classification successful: {food_label} (Confidence: {confidence:.4f})")
        timing['classification'] = time.time() - t0
    except Exception as e:
        logging.exception(f"Error during classification: {e}") # Log traceback
        return None

    # 5. Estimate Volume 
    try:
        t0 = time.time()
        intrinsics = config['camera_intrinsics']
        volume_params = config.get('volume_params', {}) # Use .get for safety
        min_depth = volume_params.get('min_depth', None) # Allow None if not specified
        max_depth = volume_params.get('max_depth', None)
        
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        # Convert depth map to points using the segmentation mask, applying depth filters
        masked_points_mm = depth_map_to_masked_points(
            depth_map, segmentation_mask, 
            fx, fy, cx, cy, 
            min_depth_m=min_depth, max_depth_m=max_depth # Pass min/max depth
        )
 
        if masked_points_mm is None:
            logging.warning("Could not generate points from depth map and mask for volume estimation.")
            volume_mm3 = 0.0
        else:
            # Estimate volume using convex hull
            volume_mm3 = estimate_volume_convex_hull(masked_points_mm)

        # Convert volume from mm³ to cm³ (1 cm³ = 1000 mm³)
        volume_cm3 = volume_mm3 / 1000.0
        results['volume_cm3'] = volume_cm3
        logging.info(f"Estimated volume: {volume_cm3:.2f} cm³")
        timing['volume_estimation'] = time.time() - t0
    except KeyError as e:
         logging.error(f"Missing camera intrinsics ('fx', 'fy', 'cx', 'cy') in config: {e}")
         return None
    except Exception as e:
        logging.exception(f"Error during volume estimation: {e}") # Log traceback
        # Allow continuation without volume? For now, stop.
        return None

    # 6. Look Up Density
    density_g_cm3 = None
    if food_label: # Only lookup if classification was successful
        try:
            t0 = time.time()
            # Pass the API key if available
            density_g_cm3 = lookup_density(food_label, api_key=usda_api_key)
            if density_g_cm3 is not None:
                results['density_g_cm3'] = density_g_cm3
                logging.info(f"Found density for {food_label}: {density_g_cm3} g/cm³")
            else:
                results['density_g_cm3'] = None # Explicitly store None if not found
                logging.warning(f"Density not found for {food_label}.")
            timing['density_lookup'] = time.time() - t0
        except Exception as e:
            logging.exception(f"Error during density lookup for {food_label}: {e}") # Log traceback
            results['density_g_cm3'] = None # Store None on error
    else:
        logging.warning("Skipping density lookup because classification failed.")
        results['density_g_cm3'] = None

    # 7. Estimate Mass 
    estimated_mass_g = None
    if volume_cm3 > 0 and density_g_cm3 is not None and density_g_cm3 > 0:
        t0 = time.time()
        estimated_mass_g = volume_cm3 * density_g_cm3
        results['estimated_mass_g'] = estimated_mass_g
        logging.info(f"Estimated mass: {estimated_mass_g:.2f} g")
        timing['mass_calculation'] = time.time() - t0
    else:
        results['estimated_mass_g'] = None
        logging.warning("Could not estimate mass (volume or density unavailable/invalid).")

    total_time = time.time() - start_time
    logging.info(f"Total analysis time: {total_time:.2f} seconds")
    logging.debug(f"Timing breakdown: {timing}")

    return results