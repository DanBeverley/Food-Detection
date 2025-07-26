import numpy as np
import os
import logging
import yaml
from pathlib import Path
import time 
import cv2
import json

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
    from volume_helpers.volume_estimator import estimate_volume_from_depth
except ImportError as e:
    logging.error(f"Failed to import helper functions: {e}")
    # Define dummy functions or re-raise to prevent execution if critical
    def depth_map_to_masked_points(*args, **kwargs): return None # type: ignore
    def estimate_volume_convex_hull(*args, **kwargs): return None # type: ignore
    def estimate_volume_from_mesh(*args, **kwargs): return None # type: ignore
    def lookup_nutritional_info(*args, **kwargs): return None # and ensure it matches signature
    def estimate_volume_from_depth(*args, **kwargs): raise NotImplementedError("estimate_volume_from_depth not imported.")

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

# Helper functions for analyze_food_item

def _load_input_data(image_path: str, depth_map_path: str | None, image_basename: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load and validate input image and depth map."""
    timing = {}
    error_messages = []
    
    try:
        t0_inputs = time.time()
        # Load image first to get dimensions for dummy depth map if needed
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Image: {image_basename} - Failed to load input image: {image_path}")
            error_messages.append("Failed to load input image.")
            raise ValueError("Failed to load input image")
        logging.debug(f"Image: {image_basename} - Loaded image, shape: {image.shape}") 

        # Attempt to load depth map
        depth_map = None
        if depth_map_path and os.path.exists(depth_map_path):
            try:
                if depth_map_path.lower().endswith('.npy'):
                    depth_map = np.load(depth_map_path)
                    logging.debug(f"Image: {image_basename} - Loaded .npy depth map from {depth_map_path}, shape: {depth_map.shape if depth_map is not None else 'None'}")
                else: 
                    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
                    logging.debug(f"Image: {image_basename} - Loaded image-based depth map from {depth_map_path}, shape: {depth_map.shape if depth_map is not None else 'None'}")

                if depth_map is None: 
                    logging.warning(f"Image: {image_basename} - Failed to load depth map from existing file {depth_map_path}. Will use default.")
            except Exception as e:
                logging.warning(f"Image: {image_basename} - Error loading depth map {depth_map_path}: {e}. Will use default.")
                depth_map = None 

        if depth_map is None: 
            logging.warning(
                f"Image: {image_basename} - Depth map at '{depth_map_path}' not found or failed to load. "
                f"Using a default dummy depth map (all pixels at 1m depth, matching image size {image.shape[0]}x{image.shape[1]})."
            )
            depth_map = np.ones((image.shape[0], image.shape[1]), dtype=np.uint16) * 1000 

        # Basic validation for the final depth_map (either loaded or dummy)
        if not isinstance(depth_map, np.ndarray) or depth_map.ndim != 2:
             logging.error(f"Image: {image_basename} - Final depth map is not a 2D NumPy array. Cannot proceed.")
             error_messages.append("Invalid final depth map.")
             raise ValueError("Invalid final depth map")
        logging.info(f"Image: {image_basename} - Using depth map of shape: {depth_map.shape}") 
        timing['load_inputs'] = time.time() - t0_inputs
        
        return image, depth_map, {'timing': timing, 'error_messages': error_messages}
        
    except Exception as e: 
        logging.error(f"Image: {image_basename} - Critical error during input data loading: {e}", exc_info=True)
        error_messages.append(f"Critical input loading error: {e}")
        raise

def _perform_segmentation(image: np.ndarray, config: dict, mask_path: str | None, 
                         image_basename: str, save_steps: bool, output_dir: str | None) -> tuple[np.ndarray | None, str, dict]:
    """Perform food segmentation."""
    timing = {}
    error_messages = []
    
    segmentation_mask = None
    segmentation_source = "unknown"
    t0_seg_overall = time.time()
    
    try:
        if mask_path and os.path.exists(mask_path):
            # Load pre-computed mask
            segmentation_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if segmentation_mask is not None:
                segmentation_source = "provided"
                logging.info(f"Image: {image_basename} - Using provided segmentation mask from: {mask_path}")
            else:
                logging.warning(f"Image: {image_basename} - Failed to load provided mask from {mask_path}. Running inference.")
        
        if segmentation_mask is None:
            # Run segmentation inference
            t0_seg = time.time()
            model_params = config.get('model_params', {})
            input_size = tuple(model_params.get('segmentation_input_size', [256, 256]))
            threshold = model_params.get('segmentation_threshold', 0.5)
            segmentation_tflite_path = config.get('models', {}).get('segmentation_tflite')
            
            if not segmentation_tflite_path:
                logging.error(f"Image: {image_basename} - No segmentation TFLite model path specified in config.")
                error_messages.append("No segmentation model path in config.")
                raise ValueError("No segmentation model path")
            
            project_root = _get_project_root()
            if not os.path.isabs(segmentation_tflite_path):
                segmentation_tflite_path = os.path.join(project_root, segmentation_tflite_path)
            
            if not os.path.exists(segmentation_tflite_path):
                logging.error(f"Image: {image_basename} - Segmentation TFLite model not found: {segmentation_tflite_path}")
                error_messages.append(f"Segmentation model not found: {segmentation_tflite_path}")
                raise FileNotFoundError(f"Segmentation model not found: {segmentation_tflite_path}")
            
            # Load model and run inference
            model, input_details, output_details = load_segmentation_model(segmentation_tflite_path)
            segmentation_mask, predicted_prob = run_segmentation_inference(
                model, input_details, output_details, image, input_size, threshold
            )
            segmentation_source = "inference"
            timing['segmentation_inference'] = time.time() - t0_seg
            logging.info(f"Image: {image_basename} - Segmentation inference completed in {timing['segmentation_inference']:.2f}s")
            
            # Save segmentation mask if enabled
            if save_steps and output_dir and segmentation_mask is not None:
                mask_filename = f"{os.path.splitext(image_basename)[0]}_segmentation_mask.png"
                mask_path_save = os.path.join(output_dir, mask_filename)
                try:
                    cv2.imwrite(mask_path_save, segmentation_mask)
                    logging.info(f"Image: {image_basename} - Segmentation mask saved to: {mask_path_save}")
                except Exception as e:
                    logging.warning(f"Image: {image_basename} - Failed to save segmentation mask: {e}")
        
        timing['segmentation_total'] = time.time() - t0_seg_overall
        return segmentation_mask, segmentation_source, {'timing': timing, 'error_messages': error_messages}
        
    except Exception as e:
        logging.error(f"Image: {image_basename} - Error during segmentation: {e}", exc_info=True)
        error_messages.append(f"Segmentation error: {e}")
        raise

# Core Analysis Function

def analyze_food_item(
    image_path: str,
    config: dict, 
    depth_map_path: str | None = None,  
    point_cloud_path: str | None = None, 
    mesh_file_path: str | None = None,  
    output_dir: str | None = None, 
    save_steps: bool = False, 
    display_results: bool = True,
    known_food_class: str | None = None, 
    usda_api_key: str | None = None, 
    mask_path: str | None = None,
    volume_estimation_method: str = 'mesh', 
    camera_intrinsics_key: str = 'default',
    custom_camera_intrinsics: dict | None = None,
    volume_estimation_config: dict | None = None
) -> dict | None:
    """
    Runs the full food analysis pipeline: segmentation, classification, volume, density, mass.

    Args:
        image_path (str): Path to the input RGB image.
        config (dict): Pipeline configuration dictionary.
        depth_map_path (str | None, optional): Path to the corresponding depth map. Defaults to None.
        point_cloud_path (str | None, optional): Path to a 3D point cloud file (e.g., .ply). Defaults to None.
        mesh_file_path (str | None, optional): Path to a 3D mesh file for volume calculation. Defaults to None.
        output_dir (str | None, optional): Directory to save intermediate outputs. Defaults to None.
        save_steps (bool, optional): Whether to save intermediate steps. Defaults to False.
        display_results (bool, optional): Whether to display results (e.g., images). Defaults to True.
        known_food_class (str | None, optional): If the food class is already known. Defaults to None.
        usda_api_key (str | None, optional): USDA API key. Defaults to None.
        mask_path (str | None, optional): Path to a pre-computed segmentation mask. Defaults to None.
        volume_estimation_method (str, optional): 'mesh' or 'depth'. Defaults to 'mesh'.
        camera_intrinsics_key (str, optional): Key for camera intrinsics. Defaults to 'default'.
        custom_camera_intrinsics (dict | None, optional): Custom camera intrinsics. Defaults to None.
        volume_estimation_config (dict | None, optional): Custom config for volume estimator. Defaults to None.

    Returns:
        dict | None: A dictionary containing analysis results, or None if a critical step fails.
    """
    results = {
        'food_label': None,
        'confidence': 0.0,
        'classification_status': 'N/A', 
        'volume_cm3': 0.0,
        'volume_method': "N/A", 
        'density_g_cm3': None,
        'estimated_mass_g': None,
        'calories_kcal_per_100g': None,  
        'estimated_total_calories': None, 
        'segmentation_mask_shape': None,
        'error_messages': [], # New list for collecting error messages
        'timing': {
            'total_pipeline': 0.0, # Overall time
            'load_inputs': 0.0,
            'segmentation_overall': 0.0,
            'segmentation_load_model': 0.0,
            'segmentation_inference': 0.0,
            'segmentation_mask_load_precomputed': 0.0,
            'segmentation_mask_resize': 0.0,
            'classification_overall': 0.0,
            'classification_load_model': 0.0,
            'classification_image_preprocessing': 0.0,
            'classification_inference': 0.0,
            'volume_estimation_overall': 0.0,
            'volume_mesh_load_calc': 0.0,
            'volume_depth_points_calc': 0.0, 
            'volume_depth_convexhull_calc': 0.0, 
            'volume_depth_voxel_calc': 0.0, 
            'nutrition_lookup': 0.0
        }
    }
    project_root = _get_project_root()
    start_time_total_pipeline = time.time()
    image_basename = os.path.basename(image_path) # For contextual logging

    logging.info(f"Analyzing food item: {image_basename}") 
    logging.info(f"Volume estimation method: {volume_estimation_method}")
    logging.info(f"Received camera_intrinsics_key = '{camera_intrinsics_key}' (Type: {type(camera_intrinsics_key)})")

    # Ensure output_dir exists if save_steps is True
    if save_steps and output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Output directory set to: {output_dir}")
        except Exception as e:
            logging.warning(f"Could not create output directory {output_dir}: {e}. Will not save intermediate steps.")
            save_steps = False # Disable saving if dir creation fails
    elif save_steps and not output_dir:
        logging.warning("save_steps is True, but no output_dir provided. Will not save intermediate steps.")
        save_steps = False

    # 1. Load Input Data
    try:
        image, depth_map, load_data_results = _load_input_data(image_path, depth_map_path, image_basename)
        results['timing'].update(load_data_results['timing'])
        results['error_messages'].extend(load_data_results['error_messages'])
        
        # Log depth map info
        if depth_map_path and os.path.exists(depth_map_path):
            logging.debug(f"Image: {image_basename} - Depth map loaded with shape: {depth_map.shape}, min_val: {depth_map.min()}, max_val: {depth_map.max()}")
        else:
            logging.info(f"Image: {image_basename} - No depth map path provided or file not found.")
    except Exception:
        return results

    # 2. Food Segmentation
    try:
        segmentation_mask, segmentation_source, seg_results = _perform_segmentation(
            image, config, mask_path, image_basename, save_steps, output_dir
        )
        results['timing'].update(seg_results['timing'])
        results['error_messages'].extend(seg_results['error_messages'])
        
        # Store segmentation results
        if segmentation_mask is not None:
            results['segmentation_mask_shape'] = segmentation_mask.shape
            results['segmentation_pixels'] = np.count_nonzero(segmentation_mask) if segmentation_mask is not None else 0
        else:
            results['segmentation_mask_shape'] = None
            results['segmentation_pixels'] = 0
            
        logging.info(f"Image: {image_basename} - Segmentation complete. Source: {segmentation_source}, Final Mask Shape: {results['segmentation_mask_shape']}")
    except Exception as e:
        logging.error(f"Image: {image_basename} - Segmentation failed: {e}")
        results['error_messages'].append(f"Segmentation failed: {e}")
        segmentation_mask = None
        segmentation_source = "failed"

    # 3. Food Classification
    food_label = None
    confidence = 0.0
    t0_class_overall = time.time()

    t0_class_img_prep = time.time()
    cropped_image_for_classification = None
    try:
        img_for_clf = cv2.imread(image_path)
        if img_for_clf is None:
            logging.error(f"Image: {image_basename} - Failed to load image for classification from {image_path}")
            return None
        img_for_clf_rgb = cv2.cvtColor(img_for_clf, cv2.COLOR_BGR2RGB)

        if segmentation_mask is not None:
            resized_segmentation_mask_for_clf = cv2.resize(
                segmentation_mask.astype(np.uint8), 
                (img_for_clf_rgb.shape[1], img_for_clf_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            coords = np.where(resized_segmentation_mask_for_clf)
            
            if len(coords[0]) > 0 and len(coords[1]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                padding = 10
                h, w = img_for_clf_rgb.shape[:2]
                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)
                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)
                
                cropped_image_for_classification = img_for_clf_rgb[y_min:y_max, x_min:x_max]
                logging.info(f"Image: {image_basename} - Cropped to bounding box: ({x_min},{y_min}) to ({x_max},{y_max})")
            else:
                logging.warning(f"Image: {image_basename} - Empty segmentation mask. Using full image for classification.")
                cropped_image_for_classification = img_for_clf_rgb
        else:
            logging.warning(f"Image: {image_basename} - No segmentation mask. Using full image.")
            cropped_image_for_classification = img_for_clf_rgb

    except Exception as e_crop:
        logging.error(f"Image: {image_basename} - Error during image preparation for classification: {e_crop}", exc_info=True)
        results['error_messages'].append(f"ClassImgPrepError: {e_crop};")
    results['timing']['classification_image_preprocessing'] = time.time() - t0_class_img_prep

    if known_food_class:
        food_label = known_food_class
        confidence = 1.0
        logging.info(f"Image: {image_basename} - Using known food class: {food_label}")
        results['classification_status'] = "Known (Pre-defined)"
    elif cropped_image_for_classification is not None:
        logging.info(f"Image: {image_basename} - Attempting classification on prepared image.")
        try:
            models_config = config.get('models', {})
            model_params_config = config.get('model_params', {})
            class_model_path_rel = models_config.get('classification_tflite')
            class_labels_path_rel = model_params_config.get('classification_labels')
            class_input_size_config = model_params_config.get('classification_input_size')
            confidence_threshold = model_params_config.get('classification_confidence_threshold', 0.6)
            
            if not all([class_model_path_rel, class_labels_path_rel, class_input_size_config]):
                logging.error(f"Image: {image_basename} - Classification model config missing.")
                results['error_messages'].append("ClassModelConfigMissing")
            else:
                class_model_path = str(project_root / class_model_path_rel)
                class_label_map_path = str(project_root / class_labels_path_rel)
                
                from models.classification.predict_classification import load_classification_model, run_classification_inference
                
                class_model, class_input_details, class_output_details, class_labels = load_classification_model(
                    class_model_path, class_label_map_path
                )
                
                if class_model and class_labels:
                    t0_class_inference = time.time()
                    classified_label, classified_confidence = run_classification_inference(
                        class_model, class_input_details, class_output_details,
                        model_input_size_hw=tuple(class_input_size_config),
                        architecture="mobilenet_v2",
                        class_labels=class_labels,
                        image_data=cropped_image_for_classification
                    )
                    results['timing']['classification_inference'] = time.time() - t0_class_inference

                    if classified_label is not None and classified_confidence is not None:
                        if classified_confidence < confidence_threshold:
                            food_label = f"Uncertain: {classified_label}"
                            confidence = float(classified_confidence)
                            results['classification_status'] = f"BelowThreshold ({confidence:.2f})"
                        else:
                            food_label = classified_label  
                            confidence = float(classified_confidence)
                            results['classification_status'] = f"Confident ({confidence:.2f})"
                    else:
                        results['classification_status'] = "Error (ModelReturnNone)"
                else:
                    results['classification_status'] = "Error (ModelLoadFail)"

        except Exception as e_class_model:
            logging.exception(f"Image: {image_basename} - Error during classification: {e_class_model}")
            results['classification_status'] = "Error (Exception)" 
    else:
        logging.warning(f"Image: {image_basename} - Skipping classification - no image prepared.")
        results['classification_status'] = "Skipped (NoImageForClf)"

    results['food_label'] = food_label
    results['confidence'] = confidence
    results['timing']['classification_overall'] = time.time() - t0_class_overall
    logging.info(f"Image: {image_basename} - Classification complete: '{food_label}', Confidence: {confidence:.2f}")

    # 5. Volume Estimation
    t0_vol_overall = time.time()
    calculated_volume_cm3 = 0.0
    volume_method_used = "N/A"

    if volume_estimation_method == 'depth':
        t0_vol_overall = time.time()
        calculated_volume_cm3 = 0.0
        volume_method_used = "N/A"

    if volume_estimation_method == 'depth':
        if depth_map is not None and segmentation_mask is not None and segmentation_mask.shape[:2] == depth_map.shape[:2]:
            logging.info(f"Image: {image_basename} - Attempting volume estimation using 'depth' method.")
            t0_vol_depth_voxel = time.time()
            try:
                # Prepare volume estimation config, merging defaults with overrides
                # Use volume processing params from pipeline config for better modularity
                vol_est_params = volume_estimation_config if volume_estimation_config is not None else config.get('volume_estimation', {}).get('processing_params', {})
                
                # Determine debug_output_path for estimate_volume_from_depth
                debug_output_path_volume = None
                if save_steps and output_dir:
                    debug_output_name = f"{Path(image_path).stem}_volume_debug"
                    debug_output_path_volume = os.path.join(output_dir, debug_output_name)
                    os.makedirs(debug_output_path_volume, exist_ok=True)
                    logging.info(f"Volume estimator debug output will be saved to: {debug_output_path_volume}")

            
                calculated_volume_cm3 = estimate_volume_from_depth(
                    depth_map=depth_map, 
                    segmentation_mask=segmentation_mask,
                    camera_intrinsics_key=camera_intrinsics_key,
                    custom_intrinsics=custom_camera_intrinsics,
                    all_camera_intrinsics=config.get('camera_intrinsics', {}), # Pass all known intrinsics
                    config=vol_est_params, 
                )
                if calculated_volume_cm3 is not None and calculated_volume_cm3 > 0:
                    volume_method_used = "depth_point_cloud_voxel"
                    logging.info(f"Image: {image_basename} - Volume (depth_point_cloud_voxel): {calculated_volume_cm3:.2f} cm³")
                else:
                    logging.warning(f"Image: {image_basename} - Depth-based volume estimation returned {calculated_volume_cm3}. Check inputs/params.")
                    calculated_volume_cm3 = 0.0 # Ensure it's a float
            except Exception as e:
                logging.error(f"Image: {image_basename} - Error during depth-based volume estimation: {e}", exc_info=True)
                calculated_volume_cm3 = 0.0
            results['timing']['volume_depth_voxel_calc'] = time.time() - t0_vol_depth_voxel
        else:
            logging.warning(f"Image: {image_basename} - Cannot use 'depth' volume estimation: depth_map ({depth_map is not None}), segmentation_mask ({segmentation_mask is not None}), or shapes mismatch.")
    
    elif volume_estimation_method == 'mesh':
        if mesh_file_path and os.path.exists(mesh_file_path):
            logging.info(f"Image: {image_basename} - Attempting volume estimation using 'mesh' method with file: {mesh_file_path}")
            t0_vol_mesh = time.time()
            try:
                calculated_volume_cm3 = estimate_volume_from_mesh(mesh_file_path)
                if calculated_volume_cm3 is not None and calculated_volume_cm3 > 0:
                    volume_method_used = "mesh_direct"
                    logging.info(f"Image: {image_basename} - Volume (mesh_direct): {calculated_volume_cm3:.2f} cm³")
                else:
                    logging.warning(f"Image: {image_basename} - Mesh-based volume estimation returned {calculated_volume_cm3}.")
                    calculated_volume_cm3 = 0.0
            except Exception as e:
                logging.error(f"Image: {image_basename} - Error during mesh volume estimation: {e}", exc_info=True)
                calculated_volume_cm3 = 0.0
            results['timing']['volume_mesh_load_calc'] = time.time() - t0_vol_mesh
        # Fallback to point cloud convex hull if mesh failed or not provided
        elif point_cloud_path and os.path.exists(point_cloud_path) and depth_map is not None and segmentation_mask is not None:
            logging.info(f"Image: {image_basename} - 'mesh' method selected, but no mesh file. Trying point cloud file: {point_cloud_path} with convex hull.")
            t0_vol_depth_hull = time.time()
            try:
                logging.warning(f"Image: {image_basename} - Using convex hull from point cloud file as fallback for 'mesh' method.")
                logging.warning(f"Image: {image_basename} - Fallback to convex hull from PLY for 'mesh' method is not fully implemented. Please use 'depth' method or provide a valid .obj file for 'mesh' method.")
                calculated_volume_cm3 = 0.0 
                volume_method_used = "point_cloud_convex_hull_fallback_unsupported"
            except Exception as e:
                logging.error(f"Image: {image_basename} - Error during point cloud convex hull volume estimation: {e}", exc_info=True)
                calculated_volume_cm3 = 0.0
            results['timing']['volume_depth_convexhull_calc'] = time.time() - t0_vol_depth_hull
        else:
            logging.warning(f"Image: {image_basename} - 'mesh' method selected, but no mesh_file_path or suitable point_cloud_path provided, or other required data missing.")
    else:
        logging.warning(f"Image: {image_basename} - Unknown volume_estimation_method: {volume_estimation_method}. No volume calculated.")

    results['volume_cm3'] = calculated_volume_cm3 if calculated_volume_cm3 is not None else 0.0
    results['volume_method'] = volume_method_used
    results['timing']['volume_estimation_overall'] = time.time() - t0_vol_overall
    logging.info(f"Image: {image_basename} - Volume estimation phase complete. Calculated Volume: {calculated_volume_cm3:.2f} cm³ (Method: {volume_method_used})")

    # 6. Nutritional Information Lookup
    t0_nutrition = time.time()
    nutritional_info = None
    results['nutritional_info_status'] = "Not Attempted"

    logging.info(f"Image: {image_basename} - Preparing for nutritional lookup. Food Label: '{food_label}', Volume: {calculated_volume_cm3:.2f} cm³")

    can_lookup_nutrition = True
    skip_reason = ""
    if food_label is None or "Uncertain:" in food_label or "Unknown" in food_label:
        can_lookup_nutrition = False
        skip_reason += "Food label is uncertain or unknown. "
    if calculated_volume_cm3 <= 0:
        can_lookup_nutrition = False
        skip_reason += "Calculated volume is zero or less." 

    if can_lookup_nutrition:
        logging.info(f"Image: {image_basename} - Attempting nutritional lookup for '{food_label}' with volume {calculated_volume_cm3:.2f} cm³.")
        try:
            # Ensure food_label passed to lookup is the clean version if it was uncertain
            clean_food_label_for_lookup = food_label.replace("Uncertain: ", "") if food_label else None

            nutritional_info = lookup_nutritional_info(
                food_item=clean_food_label_for_lookup,
                api_key=usda_api_key
            )
            if nutritional_info:
                # Calculate total calories based on volume and calories per 100g
                calories_per_100g = nutritional_info.get('calories_kcal_per_100g')
                density_g_cm3 = nutritional_info.get('density')
                
                # Calculate mass and total calories if we have the necessary data
                estimated_mass_g = None
                estimated_total_calories = None
                
                if density_g_cm3 is not None and calculated_volume_cm3 > 0:
                    estimated_mass_g = density_g_cm3 * calculated_volume_cm3
                    results['density_g_cm3'] = density_g_cm3
                    results['estimated_mass_g'] = estimated_mass_g
                    
                    if calories_per_100g is not None:
                        estimated_total_calories = (estimated_mass_g / 100.0) * calories_per_100g
                        results['estimated_total_calories'] = estimated_total_calories
                        results['calories_kcal_per_100g'] = calories_per_100g
                        
                        logging.info(f"Image: {image_basename} - Nutritional lookup successful for '{clean_food_label_for_lookup}'. "
                                   f"Density: {density_g_cm3:.2f} g/cm³, Mass: {estimated_mass_g:.2f}g, "
                                   f"Calories: {calories_per_100g:.2f} kcal/100g, Total: {estimated_total_calories:.2f} kcal.")
                    else:
                        logging.info(f"Image: {image_basename} - Found density ({density_g_cm3:.2f} g/cm³) but no calorie information for '{clean_food_label_for_lookup}'.")
                else:
                    if calories_per_100g is not None:
                        results['calories_kcal_per_100g'] = calories_per_100g
                        logging.info(f"Image: {image_basename} - Found calorie information ({calories_per_100g:.2f} kcal/100g) but no density for '{clean_food_label_for_lookup}'.")
                    else:
                        logging.info(f"Image: {image_basename} - Nutritional lookup returned data but missing key information for '{clean_food_label_for_lookup}'.")
                
                results['nutritional_info_status'] = f"Success (USDA API)"
            else:
                logging.warning(f"Image: {image_basename} - Nutritional lookup for '{clean_food_label_for_lookup}' returned no information.")
                results['nutritional_info_status'] = "NoInfoReturned"
        except Exception as e_nutrition:
            logging.exception(f"Image: {image_basename} - Error during nutritional lookup for '{food_label}': {e_nutrition}")
            results['error_messages'].append(f"NutritionError: {e_nutrition};")
            results['nutritional_info_status'] = f"Error ({e_nutrition})"
    else:
        logging.warning(f"Image: {image_basename} - Skipping nutritional lookup. Reason: {skip_reason.strip()}")
        results['nutritional_info_status'] = f"Skipped ({skip_reason.strip()})"

    results['nutritional_info'] = nutritional_info
    results['timing']['nutritional_lookup'] = time.time() - t0_nutrition


    results['timing']['total_pipeline'] = time.time() - start_time_total_pipeline
    logging.info(f"Image: {image_basename} - Food analysis pipeline completed in {results['timing']['total_pipeline']:.2f} seconds.")
    

    # Display results if enabled (basic console print for now)
    if display_results:
        print("\n--- Analysis Results ---")
        for key, value in results.items():
            if key == 'timing': # Special handling for timing dict
                print(f"  Timing Information:")
                for t_key, t_value in value.items():
                    print(f"    {t_key}: {t_value:.4f} s")
            elif isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print("------------------------\n")

    # Finalize error message from list
    if results['error_messages']:
        results['error_message'] = " | ".join(results['error_messages'])

    # --- PRODUCTION SUMMARY LOG ---
    summary_food_label = results.get('food_label', 'N/A')
    summary_confidence = results.get('confidence', 0.0) 
    summary_volume_cm3 = results.get('volume_cm3', 0.0)
    
    summary_calories_per_100g = results.get('calories_kcal_per_100g', 'N/A')
    summary_nutrition_source = "USDA API" if results.get('nutritional_info_status', '').startswith('Success') else "N/A"
    summary_total_calories = results.get('estimated_total_calories', 'N/A')
    
    if isinstance(summary_total_calories, (float, int)):
        summary_total_calories = f"{summary_total_calories:.2f}"
    if isinstance(summary_calories_per_100g, (float, int)):
        summary_calories_per_100g = f"{summary_calories_per_100g:.2f}"

    logging.info(f"--- PRODUCTION SUMMARY LOG [{image_basename}] ---"
                 f"\n  Food: {summary_food_label} (Confidence: {summary_confidence:.2f})"
                 f"\n  Volume: {summary_volume_cm3:.2f} cm³"
                 f"\n  Nutrition Source: {summary_nutrition_source}"
                 f"\n  Calories/100g: {summary_calories_per_100g} kcal"
                 f"\n  Estimated Total Calories: {summary_total_calories} kcal"
                 f"\n--- END SUMMARY ---")

    return results

# Example usage (for direct script execution, if needed for testing)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Food Analyzer')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input RGB image')
    parser.add_argument('--depth_map_path', type=str, help='Path to the corresponding depth map')
    parser.add_argument('--config_path', type=str, help='Path to the pipeline configuration file')
    parser.add_argument('--point_cloud_path', type=str, help='Path to a 3D point cloud file (e.g., .ply)')
    parser.add_argument('--mesh_file_path', type=str, help='Path to a 3D mesh file for volume calculation')
    parser.add_argument('--output_dir', type=str, help='Directory to save intermediate outputs')
    parser.add_argument('--save_steps', action='store_true', help='Whether to save intermediate steps')
    parser.add_argument('--no_display', action='store_true', help='Whether to display results')
    parser.add_argument('--known_food_class', type=str, help='If the food class is already known')
    parser.add_argument('--usda_api_key', type=str, help='USDA API key')
    parser.add_argument('--mask_path', type=str, help='Path to a pre-computed segmentation mask')
    parser.add_argument('--volume_estimation_method', type=str, default='mesh', help='Volume estimation method (mesh or depth)')
    parser.add_argument('--camera_intrinsics_key', type=str, default='default', help='Key for camera intrinsics')
    parser.add_argument('--custom_camera_intrinsics_json', type=str, help='Custom camera intrinsics as JSON')
    parser.add_argument('--volume_estimation_config_json', type=str, help='Custom config for volume estimator as JSON')

    args = parser.parse_args()

    if args.config_path is None:
        print("Error: --config_path is required for direct execution of food_analyzer.py for loading the config dictionary.")
        print("Alternatively, this example section needs to be updated to construct/load a config dict.")
        exit()

    try:
        config_dict_for_direct_run = load_pipeline_config(args.config_path) # For direct run
    except Exception:
        print(f"Failed to load config from {args.config_path} for direct script run. Exiting.")
        exit()

    analysis = analyze_food_item(
        image_path=args.image_path,
        config=config_dict_for_direct_run, # Pass loaded config dict
        depth_map_path=args.depth_map_path,
        point_cloud_path=args.point_cloud_path,
        mesh_file_path=args.mesh_file_path,
        output_dir=args.output_dir, # from argparse
        save_steps=args.save_steps, # from argparse
        display_results=not args.no_display, # from argparse
        known_food_class=args.known_food_class,
        usda_api_key=args.usda_api_key,
        mask_path=args.mask_path,
        # Pass new args for direct script testing
        volume_estimation_method=args.volume_estimation_method, 
        camera_intrinsics_key=args.camera_intrinsics_key,
        custom_camera_intrinsics=json.loads(args.custom_camera_intrinsics_json) if args.custom_camera_intrinsics_json else None,
        volume_estimation_config=json.loads(args.volume_estimation_config_json) if args.volume_estimation_config_json else None
    )

    if analysis:
        print("Analysis completed successfully.")
    else:
        print("Analysis failed.")