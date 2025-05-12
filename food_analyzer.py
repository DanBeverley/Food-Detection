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
    def load_classification_model(path): return None, None, None, None # type: ignore
    def run_classification_inference(*args): return None, 0.0 # type: ignore

try:
    from volume_helpers.volume_helpers import depth_map_to_masked_points, estimate_volume_convex_hull, estimate_volume_from_mesh
    from volume_helpers.density_lookup import lookup_nutritional_info
except ImportError as e:
    logging.error(f"Failed to import helper functions: {e}")
    # Define dummy functions or re-raise to prevent execution if critical
    def depth_map_to_masked_points(*args, **kwargs): return None # type: ignore
    def estimate_volume_convex_hull(*args, **kwargs): return None # type: ignore
    def estimate_volume_from_mesh(*args, **kwargs): return None # type: ignore
    def lookup_nutritional_info(*args, **kwargs): return None # and ensure it matches signature

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
    point_cloud_file_path: str | None = None,  
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
        point_cloud_file_path (str | None, optional): Path to a 3D point cloud file (e.g., .ply) for volume calculation. Defaults to None.
        known_food_class (str | None, optional): If the food class is already known, provide it here. Defaults to None.
        usda_api_key (str | None, optional): USDA API key for density lookup fallback. Defaults to None.

    Returns:
        dict | None: A dictionary containing analysis results, or None if a critical step fails.
                     Keys include: 'food_label', 'confidence', 'volume_cm3', 'volume_method',
                     'density_g_cm3', 'estimated_mass_g', 'calories_kcal_per_100g',
                     'estimated_total_calories', 'segmentation_mask_shape'.
    """
    results = {
        'food_label': None,
        'confidence': 0.0,
        'volume_cm3': 0.0,
        'volume_method': "N/A", 
        'density_g_cm3': None,
        'estimated_mass_g': None,
        'calories_kcal_per_100g': None,  
        'estimated_total_calories': None, 
        'segmentation_mask_shape': None,
        'segmentation_mask_path': None, 
        'error_message': None,
        'timing': {}
    }
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
                    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
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
    volume_cm3 = None
    try:
        t0 = time.time()
        if point_cloud_file_path and os.path.exists(point_cloud_file_path):
            logging.info(f"Attempting volume estimation from point cloud file: {point_cloud_file_path}")
            volume_cm3 = estimate_volume_from_mesh(point_cloud_file_path) 
            if volume_cm3 is not None and volume_cm3 > 0:
                results['volume_method'] = "point_cloud_file"
                logging.info(f"Volume from point cloud file: {volume_cm3:.2f} cm³")
            else:
                logging.warning(f"Failed to get volume from point cloud file {point_cloud_file_path} or volume was zero. Trying other methods.")
                volume_cm3 = None # Ensure it's None to try next method

        if volume_cm3 is None and mesh_file_path and os.path.exists(mesh_file_path):
            logging.info(f"Attempting volume estimation from mesh file: {mesh_file_path}")
            volume_cm3 = estimate_volume_from_mesh(mesh_file_path)
            if volume_cm3 is not None and volume_cm3 > 0:
                results['volume_method'] = "mesh_file"
                logging.info(f"Volume from mesh file: {volume_cm3:.2f} cm³")
            else:
                logging.warning(f"Failed to get volume from mesh file {mesh_file_path} or volume was zero. Trying depth map.")
                volume_cm3 = None # Ensure it's None to try next method
        
        if volume_cm3 is None:
            logging.info("Attempting volume estimation from depth map and segmentation mask.")
            cam_intrinsics = config.get('camera_intrinsics')
            if not cam_intrinsics:
                logging.error("Camera intrinsics not found in config. Cannot estimate volume from depth map.")
                results['error_message'] = "Camera intrinsics missing for depth-based volume."
            else:
                points_from_depth = depth_map_to_masked_points(
                    depth_map, segmentation_mask,
                    fx=cam_intrinsics['fx'], fy=cam_intrinsics['fy'],
                    cx=cam_intrinsics['cx'], cy=cam_intrinsics['cy'],
                    min_depth_m=config['volume_params'].get('min_depth_m'),
                    max_depth_m=config['volume_params'].get('max_depth_m'),
                    depth_scale_factor=config['volume_params'].get('depth_scale_factor', 1.0)
                )
                if points_from_depth is not None and points_from_depth.shape[0] > 0:
                    # Convert volume from mm³ (from estimate_volume_convex_hull) to cm³
                    volume_mm3 = estimate_volume_convex_hull(points_from_depth)
                    if volume_mm3 is not None and volume_mm3 > 0:
                        volume_cm3 = volume_mm3 / 1000.0
                        results['volume_method'] = "depth_map_convex_hull"
                        logging.info(f"Volume from depth map (convex hull): {volume_cm3:.2f} cm³ ({volume_mm3:.2f} mm³)")
                    else:
                        logging.warning("Volume from depth map was zero or None.")
                else:
                    logging.warning("Failed to get 3D points from depth map.")
        
        if volume_cm3 is not None and volume_cm3 > 0:
            results['volume_cm3'] = volume_cm3
        else:
            results['volume_cm3'] = 0.0 # Default to 0 if no method succeeded
            results['volume_method'] = "N/A"
            logging.warning("All volume estimation methods failed or yielded zero volume.")
        timing['volume_estimation'] = time.time() - t0
    except Exception as e:
        logging.exception(f"Error during volume estimation: {e}")
        results['error_message'] = f"Volume estimation error: {e}"
        results['volume_cm3'] = 0.0 # Ensure volume is 0 on error
        results['volume_method'] = "Error"

    # 6. Look Up Nutritional Info (Density & Calories)
    density_g_cm3 = None
    calories_kcal_per_100g = None 
    if food_label: 
        try:
            t0 = time.time()
            logging.info(f"Looking up nutritional info for: {food_label}")
            # Use the API key passed to analyze_food_item
            nutritional_info = lookup_nutritional_info(food_label, api_key=usda_api_key) 

            if nutritional_info:
                density_g_cm3 = nutritional_info.get('density')
                calories_kcal_per_100g = nutritional_info.get('calories_kcal_per_100g')
                
                if density_g_cm3 is not None:
                    results['density_g_cm3'] = float(density_g_cm3)
                    logging.info(f"Density for {food_label}: {density_g_cm3:.2f} g/cm³")
                else:
                    logging.warning(f"Density not found for {food_label}.")
                
                if calories_kcal_per_100g is not None: 
                    results['calories_kcal_per_100g'] = float(calories_kcal_per_100g)
                    logging.info(f"Calories for {food_label}: {calories_kcal_per_100g:.2f} kcal/100g")
                else:
                    logging.warning(f"Calories (kcal/100g) not found for {food_label}.")
            else:
                logging.warning(f"Nutritional info (density/calories) lookup returned None for {food_label}.")
            timing['nutritional_lookup'] = time.time() - t0 
        except Exception as e:
            logging.exception(f"Error during nutritional info lookup for {food_label}: {e}")
            results['error_message'] = results.get('error_message', "") + f"NutritionalLookupError: {e}; "

    # 7. Estimate Mass and Total Calories
    estimated_mass_g = None
    estimated_total_calories = None 
    try:
        t0 = time.time()
        if density_g_cm3 is not None and results['volume_cm3'] is not None and results['volume_cm3'] > 0:
            estimated_mass_g = density_g_cm3 * results['volume_cm3']
            results['estimated_mass_g'] = float(estimated_mass_g)
            logging.info(f"Estimated mass for {food_label}: {estimated_mass_g:.2f} g")

            if calories_kcal_per_100g is not None: 
                estimated_total_calories = (estimated_mass_g / 100.0) * calories_kcal_per_100g
                results['estimated_total_calories'] = float(estimated_total_calories)
                logging.info(f"Estimated total calories for {food_label}: {estimated_total_calories:.2f} kcal")
            else:
                logging.info(f"Cannot estimate total calories for {food_label} as calories/100g is unknown.")

        elif results['volume_cm3'] == 0.0:
             logging.info(f"Cannot estimate mass or total calories for {food_label} as volume is 0 cm³.")
        else:
            logging.info(f"Cannot estimate mass or total calories for {food_label} as density or volume is unknown/invalid.")
        timing['mass_calorie_estimation'] = time.time() - t0 
    except Exception as e:
        logging.exception(f"Error during mass and calorie estimation: {e}")
        results['error_message'] = results.get('error_message', "") + f"MassCalorieError: {e}; "

    total_time = time.time() - start_time
    logging.info(f"Total analysis time: {total_time:.2f} seconds")
    logging.debug(f"Timing breakdown: {timing}")

    return results