import numpy as np
import os
import logging
import yaml
from pathlib import Path
import time 
import cv2

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
    from volume_helpers.volume_helpers import depth_map_to_masked_points, estimate_volume_convex_hull, estimate_volume_from_mesh
    from volume_helpers.density_lookup import lookup_density
except ImportError as e:
     logging.error(f"Failed to import local helpers: {e}. Ensure volume_helpers and density_lookup exist.")
     def depth_map_to_masked_points(*args): raise NotImplementedError("depth_map_to_masked_points not implemented/imported.")
     def estimate_volume_convex_hull(*args): raise NotImplementedError("estimate_volume_convex_hull not implemented/imported.")
     def estimate_volume_from_mesh(*args): raise NotImplementedError("estimate_volume_from_mesh not implemented/imported.")
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
    config: dict,
    depth_map_path: str | None = None,  
    mesh_file_path: str | None = None,  
    known_food_class: str | None = None, 
    usda_api_key: str | None = None
) -> dict | None:
    """
    Runs the full food analysis pipeline: segmentation, classification, volume, density, mass.

    Args:
        image_path (str): Path to the input RGB image.
        config (dict): Loaded pipeline configuration dictionary.
        depth_map_path (str | None, optional): Path to the corresponding depth map. Defaults to None.
        mesh_file_path (str | None, optional): Path to a 3D mesh file for volume calculation. Defaults to None.
        known_food_class (str | None, optional): If the food class is already known, provide it here. Defaults to None.
        usda_api_key (str | None, optional): USDA API key for density lookup fallback. Defaults to None.

    Returns:
        dict | None: A dictionary containing analysis results ('food_label', 'confidence',
                     'volume_cm3', 'density_g_cm3', 'estimated_mass_g', 'segmentation_mask_shape'),
                     or None if a critical step fails.
    """
    results = {}
    timing = {} 
    project_root = _get_project_root()
    start_time = time.time()

    #1. Load Input Data
    try:
        t0 = time.time()
        # Load image first to get dimensions for dummy depth map if needed
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to load input image: {image_path}")
            return None
        logging.debug(f"Loaded image from {image_path}, shape: {image.shape}") 

        # Attempt to load depth map
        depth_map = None
        if depth_map_path and os.path.exists(depth_map_path):
            try:
                if depth_map_path.lower().endswith('.npy'):
                    depth_map = np.load(depth_map_path)
                    logging.debug(f"Loaded .npy depth map from {depth_map_path}, shape: {depth_map.shape if depth_map is not None else 'None'}")
                else: 
                    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
                    logging.debug(f"Loaded image-based depth map from {depth_map_path}, shape: {depth_map.shape if depth_map is not None else 'None'}")

                if depth_map is None: 
                    logging.warning(f"Failed to load depth map from existing file {depth_map_path} (e.g. unsupported format or corrupted). Will use default.")
            except Exception as e:
                logging.warning(f"Error loading depth map {depth_map_path}: {e}. Will use default.")
                depth_map = None 

        if depth_map is None: 
            logging.warning(
                f"Depth map at '{depth_map_path}' not found or failed to load. "
                f"Using a default dummy depth map (all pixels at 1m depth, matching image size {image.shape[0]}x{image.shape[1]})."
            )
            depth_map = np.ones((image.shape[0], image.shape[1]), dtype=np.uint16) * 1000 

        # Basic validation for the final depth_map (either loaded or dummy)
        if not isinstance(depth_map, np.ndarray) or depth_map.ndim != 2:
             logging.error(f"Final depth map is not a 2D NumPy array (type: {type(depth_map)}, ndim: {depth_map.ndim if isinstance(depth_map, np.ndarray) else 'N/A'}). Cannot proceed.")
             return None
        logging.info(f"Using depth map of shape: {depth_map.shape}") 
        timing['load_inputs'] = time.time() - t0
    except Exception as e: 
        logging.error(f"Critical error during input data loading: {e}", exc_info=True)
        return None

    #2. Load Models 
    try:
        t0 = time.time()
        seg_model_path = os.path.join(project_root, config['models']['segmentation_tflite'])
        seg_interpreter, seg_input_details, seg_output_details = load_segmentation_model(seg_model_path)

        clf_model_path = os.path.join(project_root, config['models']['classification_tflite'])
        class_labels_path_rel = config['models'].get('classification_labels') 
        class_labels_path = None
        if class_labels_path_rel:
            class_labels_path = os.path.join(project_root, class_labels_path_rel) 
        
        clf_interpreter, clf_input_details, clf_output_details, class_labels = load_classification_model(clf_model_path, labels_path=class_labels_path)
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
        seg_input_size = tuple(config['model_params']['segmentation_input_size'])
        seg_num_classes = config['model_params'].get('segmentation_num_classes', 2) 
        
        segmentation_mask = run_segmentation_inference(
            seg_interpreter, seg_input_details, seg_output_details,
            image_path, seg_input_size, depth_map.shape, 
            num_classes=seg_num_classes 
        )
        if segmentation_mask is None:
            logging.error("Segmentation failed.")
            return None
        segmentation_mask = segmentation_mask.astype(bool)
        results['segmentation_mask_shape'] = segmentation_mask.shape
        logging.info(f"Segmentation successful. Mask shape: {segmentation_mask.shape}")
        timing['segmentation'] = time.time() - t0
    except Exception as e:
        logging.exception(f"Error during segmentation: {e}") 
        return None

    # 4. Determine Food Class 
    food_label = None
    confidence = 0.0
    try:
        t0 = time.time()
        if known_food_class:
            food_label = known_food_class
            confidence = 1.0  
            logging.info(f"Using known food class: {food_label}")
        else:
            target_clf_size = tuple(config['model_params']['classification_input_size'])
            classified_label, classified_confidence = run_classification_inference(
                clf_interpreter, clf_input_details, clf_output_details,
                image_path, target_clf_size, class_labels
            )
            if classified_label is None:
                logging.error("Image-based classification failed to return a label.")
                return None 
            food_label = classified_label
            confidence = classified_confidence
            logging.info(f"Image-based classification successful: {food_label} (Confidence: {confidence:.4f})")
        
        results['food_label'] = str(food_label) 
        results['confidence'] = float(confidence) 
        timing['classification'] = time.time() - t0
    except Exception as e:
        logging.exception(f"Error during food class determination: {e}") 
        return None

    # 5. Estimate Volume 
    volume_cm3 = 0.0  
    try:
        t0 = time.time()
        volume_calculation_method = "N/A"

        if mesh_file_path and os.path.exists(mesh_file_path):
            logging.info(f"Attempting volume estimation from mesh file: {mesh_file_path}")
            volume_from_mesh = estimate_volume_from_mesh(mesh_file_path)
            if volume_from_mesh is not None:
                volume_cm3 = volume_from_mesh
                volume_calculation_method = f"Mesh file ({os.path.basename(mesh_file_path)})"
                logging.info(f"Volume successfully calculated from mesh: {volume_cm3:.2f} cm³")
            else:
                logging.warning(f"Failed to estimate volume from mesh file {mesh_file_path}. Volume remains {volume_cm3:.2f} cm³.")
        
        if volume_cm3 == 0.0 and depth_map is not None and segmentation_mask is not None: 
            logging.info("Attempting volume estimation from depth map and segmentation mask.")
            intrinsics = config['camera_intrinsics']
            volume_params = config.get('volume_params', {}) 

            points = depth_map_to_masked_points(depth_map, segmentation_mask, intrinsics)
            if points is None or points.shape[0] < 4: 
                logging.warning(f"Not enough points ({points.shape[0] if points is not None else 0}) from depth map and mask for convex hull. Volume remains {volume_cm3:.2f} cm³.")
            else:
                qhull_options = volume_params.get('qhull_options', 'QJ') 
                volume_from_depth = estimate_volume_convex_hull(points, qhull_options=qhull_options)
                if volume_from_depth is not None:
                    volume_cm3 = volume_from_depth
                    volume_calculation_method = "Depth map (Convex Hull)"
                    logging.info(f"Volume successfully calculated from depth map: {volume_cm3:.2f} cm³")
                else:
                    logging.warning(f"Convex hull volume estimation from depth map returned None. Volume remains {volume_cm3:.2f} cm³.")
        elif volume_cm3 == 0.0: 
             logging.warning("Could not estimate volume from mesh or depth map. Volume set to 0.0 cm³.")
             volume_calculation_method = "Unavailable (defaulted to 0)"
        
        results['volume_cm3'] = float(volume_cm3)
        results['volume_method'] = volume_calculation_method 
        logging.info(f"Final estimated volume: {results['volume_cm3']:.2f} cm³ (Method: {results['volume_method']})")
        timing['volume_estimation'] = time.time() - t0
    except Exception as e:
        logging.exception(f"Error during volume estimation: {e}") 
        return None

    # 6. Look Up Density
    density_g_cm3 = None
    if food_label: 
        try:
            t0 = time.time()
            density_g_cm3 = lookup_density(food_label, api_key=usda_api_key)
            if density_g_cm3 is not None:
                results['density_g_cm3'] = density_g_cm3
                logging.info(f"Found density for {food_label}: {density_g_cm3} g/cm³")
            else:
                results['density_g_cm3'] = None 
                logging.warning(f"Density not found for {food_label}.")
            timing['density_lookup'] = time.time() - t0
        except Exception as e:
            logging.exception(f"Error during density lookup for {food_label}: {e}") 
            results['density_g_cm3'] = None 
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