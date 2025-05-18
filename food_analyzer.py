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
    config_path: str,
    image_path: str,
    depth_map_path: str | None = None,  
    mesh_file_path: str | None = None,  
    point_cloud_file_path: str | None = None,  
    known_food_class: str | None = None, 
    usda_api_key: str | None = None,
    mask_path: str | None = None
) -> dict | None:
    """
    Runs the full food analysis pipeline: segmentation, classification, volume, density, mass.

    Args:
        config_path (str): Path to the pipeline configuration file.
        image_path (str): Path to the input RGB image.
        depth_map_path (str | None, optional): Path to the corresponding depth map. Defaults to None.
        mesh_file_path (str | None, optional): Path to a 3D mesh file for volume calculation. Defaults to None.
        point_cloud_file_path (str | None, optional): Path to a 3D point cloud file (e.g., .ply) for volume calculation. Defaults to None.
        known_food_class (str | None, optional): If the food class is already known, provide it here. Defaults to None.
        usda_api_key (str | None, optional): USDA API key for density lookup fallback. Defaults to None.
        mask_path (str | None, optional): Path to a pre-computed segmentation mask. Defaults to None.

    Returns:
        dict | None: A dictionary containing analysis results, or None if a critical step fails.
                     Keys include: 'food_label', 'confidence', 'volume_cm3', 'volume_method',
                     'density_g_cm3', 'estimated_mass_g', 'calories_kcal_per_100g',
                     'estimated_total_calories', 'segmentation_mask_shape', 'segmentation_source',
                     'classification_status'.
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
        'error_message': None,
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
            'nutrition_lookup': 0.0
        }
    }
    project_root = _get_project_root()
    start_time_total_pipeline = time.time()
    image_basename = os.path.basename(image_path) # For contextual logging

    # Load configuration early
    try:
        config = load_pipeline_config(config_path)
        if config is None: # load_pipeline_config might raise, or return None in some hypothetical future version
            logging.error(f"Image: {image_basename} - Failed to load pipeline configuration. Cannot proceed.")
            results['error_message'] = "Failed to load pipeline configuration."
            return results
    except Exception as e: # Handles errors from load_pipeline_config (e.g. FileNotFoundError)
        logging.error(f"Image: {image_basename} - Critical error loading configuration: {e}. Cannot proceed.")
        results['error_message'] = f"Critical error loading configuration: {e}"
        return results

    #1. Load Input Data
    try:
        t0_inputs = time.time()
        # Load image first to get dimensions for dummy depth map if needed
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Image: {image_basename} - Failed to load input image: {image_path}")
            results['error_message'] = "Failed to load input image."
            return results # Critical failure
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
             results['error_message'] = "Invalid final depth map."
             return results # Critical failure
        logging.info(f"Image: {image_basename} - Using depth map of shape: {depth_map.shape}") 
        results['timing']['load_inputs'] = time.time() - t0_inputs
    except Exception as e: 
        logging.error(f"Image: {image_basename} - Critical error during input data loading: {e}", exc_info=True)
        results['error_message'] = f"Critical input loading error: {e}"
        return results

    # 2. Load Depth Map (if provided)
    if depth_map_path and os.path.exists(depth_map_path):
        logging.debug(f"Image: {image_basename} - Depth map loaded with shape: {depth_map.shape}, min_val: {depth_map.min()}, max_val: {depth_map.max()}")
    else:
        logging.info(f"Image: {image_basename} - No depth map path provided or file not found.")

    # 3. Food Segmentation
    segmentation_mask = None
    segmentation_source = "unknown" # Initialize segmentation source
    t0_seg_overall = time.time()

    if mask_path and os.path.exists(mask_path):
        t0_seg_mask_load = time.time()
        try:
            loaded_mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if loaded_mask_gray is not None:
                _, thresholded_mask = cv2.threshold(loaded_mask_gray, 1, 255, cv2.THRESH_BINARY)
                segmentation_mask = thresholded_mask.astype(bool)
                logging.info(f"Image: {image_basename} - Successfully loaded pre-computed mask from: {mask_path} with initial shape {segmentation_mask.shape}")
                segmentation_source = f"precomputed_mask_file: {os.path.basename(mask_path)}"
            else:
                logging.warning(f"Image: {image_basename} - cv2.imread returned None for pre-computed mask {mask_path}. Falling back to model generation.")
        except Exception as e_load_mask:
            logging.warning(f"Image: {image_basename} - Error loading pre-computed mask from {mask_path}: {e_load_mask}. Falling back to model generation.")
        results['timing']['segmentation_mask_load_precomputed'] = time.time() - t0_seg_mask_load

    if segmentation_mask is None: # Fallback if not loaded or path not provided or load failed
        logging.info(f"Image: {image_basename} - No valid pre-computed mask, attempting to generate mask using model.")
        try:
            models_config = config.get('models', {})
            seg_model_path_rel = models_config.get('segmentation_tflite')
            model_params_config = config.get('model_params', {})
            seg_input_size_config = model_params_config.get('segmentation_input_size')

            if not seg_model_path_rel or not seg_input_size_config:
                logging.error(f"Image: {image_basename} - Segmentation model path or input size missing in config. Skipping model-based segmentation.")
                results['error_message'] = results.get('error_message', '') + " SegModelConfigMissing;"
            else:
                t0_seg_load_model = time.time()
                seg_model_path = str(project_root / seg_model_path_rel)
                seg_interpreter, seg_input_details, seg_output_details = load_segmentation_model(seg_model_path)
                results['timing']['segmentation_load_model'] = time.time() - t0_seg_load_model
                
                target_seg_size = tuple(seg_input_size_config)
                
                t0_seg_inference = time.time()
                segmentation_mask = run_segmentation_inference(
                    seg_interpreter, seg_input_details, seg_output_details,
                    image_path, target_seg_size
                )
                results['timing']['segmentation_inference'] = time.time() - t0_seg_inference
                logging.info(f"Image: {image_basename} - Generated mask using model, shape: {segmentation_mask.shape if segmentation_mask is not None else 'None'}")
                if segmentation_mask is not None:
                    segmentation_source = "model_generated"
                else:
                    logging.warning(f"Image: {image_basename} - Model-based segmentation returned None.")
                    results['error_message'] = results.get('error_message', '') + " SegModelReturnedNone;"

        except Exception as e_seg_model:
            logging.exception(f"Image: {image_basename} - Error during model-based segmentation: {e_seg_model}")
            results['error_message'] = results.get('error_message', '') + f" SegModelError: {e_seg_model};"

    t0_seg_mask_resize = time.time()
    if segmentation_mask is not None:
        target_h, target_w = -1, -1
        resize_info = "original image shape"
        if depth_map is not None:
            target_h, target_w = depth_map.shape[:2]
            resize_info = "depth map shape"
        else: 
            temp_img_for_shape = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if temp_img_for_shape is not None:
                target_h, target_w = temp_img_for_shape.shape[:2]
            else:
                logging.warning(f"Image: {image_basename} - Could not read image to determine target shape for mask resizing (depth map absent). Using mask's current shape.")
                target_h, target_w = segmentation_mask.shape[:2] # Default to current shape if image load fails
                resize_info = "mask's current shape (image load failed for fallback resizing)"

        if segmentation_mask.shape[0] != target_h or segmentation_mask.shape[1] != target_w:
            logging.info(f"Image: {image_basename} - Resizing segmentation mask from {segmentation_mask.shape} to ({target_h}, {target_w}) based on {resize_info}.")
            segmentation_mask = cv2.resize(
                segmentation_mask.astype(np.uint8) * 255, # Convert boolean to 0/255 for resize
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        results['segmentation_mask_shape'] = segmentation_mask.shape
        logging.info(f"Image: {image_basename} - Final segmentation mask shape: {segmentation_mask.shape} (Source: {segmentation_source})")
    else: # segmentation_mask was None from the start (e.g. model error and no precomputed)
        logging.error(f"Image: {image_basename} - Segmentation mask is None. Cannot proceed.")
        results['error_message'] = (results.get('error_message') or "") + " NoSegmentationMaskAtAll;"
        results['timing']['segmentation_overall'] = time.time() - t0_seg_overall
        results['timing']['total_pipeline'] = time.time() - start_time_total_pipeline
        return results
    results['timing']['segmentation_mask_resize'] = time.time() - t0_seg_mask_resize
    results['timing']['segmentation_overall'] = time.time() - t0_seg_overall
    results['segmentation_source'] = segmentation_source
    logging.info(f"Image: {image_basename} - Segmentation complete. Source: {segmentation_source}, Final Mask Shape: {results['segmentation_mask_shape']}")

    # 4. Food Classification
    food_label = None
    confidence = 0.0
    t0_class_overall = time.time() 

    # Prepare image for classification (cropping)
    t0_class_img_prep = time.time()
    cropped_image_for_classification = None
    try:
        # Load image for classification
        img_for_clf = cv2.imread(image_path)
        if img_for_clf is None:
            logging.error(f"Image: {image_basename} - Failed to load image for classification from {image_path}")
            return None
        img_for_clf_rgb = cv2.cvtColor(img_for_clf, cv2.COLOR_BGR2RGB)

        # Resize segmentation_mask to match img_for_clf_rgb dimensions
        # segmentation_mask is (H_mask, W_mask), img_for_clf_rgb is (H_img, W_img, C)
        # We need to resize mask to (H_img, W_img)
        resized_segmentation_mask_for_clf = cv2.resize(
            segmentation_mask.astype(np.uint8), 
            (img_for_clf_rgb.shape[1], img_for_clf_rgb.shape[0]), # (W, H) for cv2.resize
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        if not np.any(resized_segmentation_mask_for_clf):
            logging.warning(f"Image: {image_basename} - Resized segmentation mask for classification is empty. Classification might be unreliable.")
            # Optional: use original image if mask is empty, or skip classification
            # For now, proceed with potentially empty masked_img_for_clf which will be black

        masked_img_for_clf = np.zeros_like(img_for_clf_rgb)
        masked_img_for_clf[resized_segmentation_mask_for_clf] = img_for_clf_rgb[resized_segmentation_mask_for_clf]
        cropped_image_for_classification = masked_img_for_clf
        logging.debug(f"Image: {image_basename} - Successfully created masked image for classification.")

    except Exception as e_crop:
        logging.error(f"Image: {image_basename} - Error during image preparation for classification: {e_crop}", exc_info=True)
        results['error_message'] = (results.get('error_message') or "") + f" ClassImgPrepError: {e_crop};"
    results['timing']['classification_image_preprocessing'] = time.time() - t0_class_img_prep

    if known_food_class:
        food_label = known_food_class
        confidence = 1.0  
        logging.info(f"Image: {image_basename} - Using known food class: {food_label}")
        results['classification_status'] = "Known (Pre-defined)"
    elif cropped_image_for_classification is not None:
        logging.info(f"Image: {image_basename} - Attempting classification on the cropped/masked image.")
        try:
            models_config = config.get('models', {})
            class_model_path_rel = models_config.get('classification_tflite')
            model_params_config = config.get('model_params', {})
            class_input_size_config = model_params_config.get('classification_input_size')
            class_label_map_path_rel = model_params_config.get('classification_label_map')
            # Get the confidence threshold, default to 0.6 if not in config
            confidence_threshold = model_params_config.get('classification_confidence_threshold', 0.6)
            # Get architecture from the classification's own config file
            classification_config_rel_path = model_params_config.get('classification_config_path', 'models/classification/config.yaml')
            classification_config_abs_path = project_root / classification_config_rel_path
            clf_architecture = "Unknown"
            if os.path.exists(classification_config_abs_path):
                try:
                    with open(classification_config_abs_path, 'r') as f_clf_cfg:
                        clf_cfg_content = yaml.safe_load(f_clf_cfg)
                        clf_architecture = clf_cfg_content.get('model',{}).get('architecture', 'Unknown')
                    logging.info(f"Image: {image_basename} - Determined classification architecture: {clf_architecture} from {classification_config_abs_path}")
                except Exception as e_clf_cfg:
                    logging.warning(f"Image: {image_basename} - Error loading classification config {classification_config_abs_path} to get architecture: {e_clf_cfg}. Defaulting to 'Unknown'.")
            else:
                logging.warning(f"Image: {image_basename} - Classification config {classification_config_abs_path} not found. Defaulting architecture to 'Unknown'.")

            if not class_model_path_rel or not class_input_size_config or not class_label_map_path_rel:
                logging.error(f"Image: {image_basename} - Classification model path, input size, or label map missing in config. Skipping classification.")
                results['error_message'] = results.get('error_message', '') + " ClassModelConfigMissing;"
                results['classification_status'] = "Skipped (ConfigMissing)"
            else:
                t0_class_load_model = time.time()
                class_model_path = str(project_root / class_model_path_rel)
                class_label_map_path = str(project_root / class_label_map_path_rel)

                class_model, class_input_details, class_output_details, class_labels = load_classification_model(
                    class_model_path,
                    class_label_map_path
                )
                results['timing']['classification_load_model'] = time.time() - t0_class_load_model
                target_class_size = tuple(class_input_size_config)

                if class_model and class_labels is not None:
                    t0_class_inference = time.time()
                    classified_label, classified_confidence = run_classification_inference(
                        class_model, class_input_details, class_output_details, class_labels,
                        image_data=cropped_image_for_classification, 
                        model_input_size_hw=target_class_size, # Renamed from target_size_hw for clarity
                        architecture=clf_architecture # pass architecture
                    )
                    results['timing']['classification_inference'] = time.time() - t0_class_inference

                    if classified_label and classified_confidence is not None:
                        food_label = classified_label
                        confidence = float(classified_confidence)
                        logging.info(f"Image: {image_basename} - Classification result: {food_label} (Confidence: {confidence:.2f})")

                        if confidence < confidence_threshold:
                            logging.warning(f"Image: {image_basename} - Classification confidence {confidence:.2f} is below threshold {confidence_threshold}. Label '{food_label}' marked as uncertain.")
                            food_label = f"Uncertain: {food_label}"
                            results['classification_status'] = f"BelowConfidenceThreshold (Score: {confidence:.2f})"
                        else:
                            results['classification_status'] = f"Confident (Score: {confidence:.2f})"
                    else:
                        logging.warning(f"Image: {image_basename} - Classification model returned None for label or confidence.")
                        results['error_message'] = results.get('error_message', '') + " ClassModelReturnNone;"
                        results['classification_status'] = "Error (ModelReturnNone)"
                else:
                    logging.error(f"Image: {image_basename} - Failed to load classification model or labels. Skipping classification.")
                    results['error_message'] = results.get('error_message', '') + " ClassModelLoadFail;"
                    results['classification_status'] = "Error (ModelLoadFail)"

        except Exception as e_class_model:
            logging.exception(f"Image: {image_basename} - Error during classification: {e_class_model}")
            results['error_message'] = results.get('error_message', '') + f" ClassModelError: {e_class_model};"
            results['classification_status'] = f"Error ({e_class_model})"
    else: # known_food_class was not provided AND cropped_image_for_classification is None
        logging.warning(f"Image: {image_basename} - Skipping classification as no valid image for classification and no known_food_class provided.")
        results['classification_status'] = "Skipped (NoImageForClf)"

    results['food_label'] = food_label
    results['confidence'] = confidence
    results['timing']['classification_overall'] = time.time() - t0_class_overall

    # 5. Volume Estimation
    volume_cm3 = None
    volume_method = "N/A"
    t0_vol_overall = time.time()

    if mesh_file_path and os.path.exists(mesh_file_path):
        t0_vol_mesh = time.time()
        try:
            volume_cm3, volume_method = estimate_volume_from_mesh(mesh_file_path)
            if volume_cm3 is not None:
                logging.info(f"Image: {image_basename} - Volume estimated from mesh {mesh_file_path}: {volume_cm3:.2f} cm³")
            else:
                logging.warning(f"Image: {image_basename} - estimate_volume_from_mesh returned None for {mesh_file_path}")
        except Exception as e_mesh_vol:
            logging.error(f"Image: {image_basename} - Error estimating volume from mesh {mesh_file_path}: {e_mesh_vol}", exc_info=True)
            results['error_message'] = (results.get('error_message') or "") + f" MeshVolumeError: {e_mesh_vol};"
        results['timing']['volume_mesh_load_calc'] = time.time() - t0_vol_mesh
    
    if volume_cm3 is None and segmentation_mask is not None and depth_map is not None:
        logging.info(f"Image: {image_basename} - Attempting volume estimation from depth map and segmentation mask.")
        cam_intrinsics = config.get('camera_intrinsics', {})
        depth_processing_config = config.get('depth_processing', {})
        min_depth_mm_config = depth_processing_config.get('min_depth_mm') # Now mm
        max_depth_mm_config = depth_processing_config.get('max_depth_mm') # Now mm

        if not all(cam_intrinsics.get(k) for k in ['fx', 'fy', 'cx', 'cy']):
            logging.error(f"Image: {image_basename} - Incomplete camera intrinsics in config. Cannot estimate volume from depth map.")
            results['error_message'] = (results.get('error_message') or "") + " IncompleteCameraIntrinsics;"
        else:
            t0_vol_depth_points = time.time()
            points_from_depth = depth_map_to_masked_points(
                depth_map,
                segmentation_mask, 
                cam_intrinsics,
                min_depth_mm=min_depth_mm_config, 
                max_depth_mm=max_depth_mm_config
            )
            results['timing']['volume_depth_points_calc'] = time.time() - t0_vol_depth_points

            if points_from_depth is not None and len(points_from_depth) > 3:
                t0_vol_depth_hull = time.time()
                vol_details = estimate_volume_convex_hull(points_from_depth)
                results['timing']['volume_depth_convexhull_calc'] = time.time() - t0_vol_depth_hull
                if vol_details:
                    volume_cm3 = vol_details.get('volume_cm3')
                    volume_method = vol_details.get('method_description', "depth_convex_hull")
                    if volume_cm3 is not None:
                        logging.info(f"Image: {image_basename} - Volume from depth (convex hull): {volume_cm3:.2f} cm³")
                    else:
                        logging.warning(f"Image: {image_basename} - Convex hull volume estimation from depth returned None for volume_cm3.")
                else:
                    logging.warning(f"Image: {image_basename} - Convex hull volume estimation from depth returned None for details.")
            elif points_from_depth is not None:
                logging.warning(f"Image: {image_basename} - Not enough points ({len(points_from_depth)}) from depth map for convex hull volume estimation (need >3).")
            else:
                logging.warning(f"Image: {image_basename} - Failed to get points from depth map for volume estimation.")
    elif volume_cm3 is None: # Mesh not used/failed, and (mask or depth missing for depth_volume)
        logging.warning(f"Image: {image_basename} - Volume estimation skipped. No mesh provided/usable, or mask/depth unavailable for depth-based method.")

    results['volume_cm3'] = volume_cm3 if volume_cm3 is not None else 0.0
    results['volume_method'] = volume_method
    results['timing']['volume_estimation_overall'] = time.time() - t0_vol_overall

    # 6. Density & Nutritional Lookup
    density_g_cm3 = None
    calories_kcal_per_100g = None
    estimated_total_calories = None
    t0_nutrition = time.time()

    if food_label and not food_label.startswith("Uncertain:") and volume_cm3 is not None and volume_cm3 > 0:
        logging.info(f"Image: {image_basename} - Looking up nutritional info for: {food_label}")
        databases_config = config.get('databases', {})
        custom_db_path_rel = databases_config.get('custom_density_db')
        if not custom_db_path_rel:
            logging.warning(f"Image: {image_basename} - Custom density database path not in config. USDA/fallback might be limited.")
            custom_db_path = None
        else:
            custom_db_path = str(project_root / custom_db_path_rel)

        nutritional_info = lookup_nutritional_info(food_label, custom_db_path, usda_api_key)
        if nutritional_info:
            density_g_cm3 = nutritional_info.get('density')
            calories_kcal_per_100g = nutritional_info.get('calories_kcal_per_100g')
            if density_g_cm3 is not None:
                logging.info(f"Image: {image_basename} - Found density for {food_label}: {density_g_cm3} g/cm³")
                estimated_mass_g = density_g_cm3 * volume_cm3
                results['estimated_mass_g'] = estimated_mass_g
                if calories_kcal_per_100g is not None and estimated_mass_g is not None:
                    logging.info(f"Image: {image_basename} - Found calories for {food_label}: {calories_kcal_per_100g} kcal/100g")
                    estimated_total_calories = (estimated_mass_g / 100.0) * calories_kcal_per_100g
                    results['estimated_total_calories'] = estimated_total_calories
                elif calories_kcal_per_100g is None:
                    logging.info(f"Image: {image_basename} - Calories per 100g not found for {food_label} in nutritional info.")
            else:
                logging.info(f"Image: {image_basename} - Density not found for {food_label} in nutritional info.")
        else:
            logging.info(f"Image: {image_basename} - Nutritional info not found for {food_label}.")
        results['density_g_cm3'] = density_g_cm3
        results['calories_kcal_per_100g'] = calories_kcal_per_100g
        results['timing']['nutrition_lookup'] = time.time() - t0_nutrition
    elif food_label and food_label.startswith("Uncertain:"):
        logging.info(f"Image: {image_basename} - Skipping nutritional lookup for uncertain classification: {food_label}")
    elif not (volume_cm3 is not None and volume_cm3 > 0):
        logging.info(f"Image: {image_basename} - Skipping nutritional lookup for {food_label} due to zero or invalid volume.")
    else: # food_label is None
        logging.info(f"Image: {image_basename} - Skipping nutritional lookup as food label is not determined.")

    results['timing']['total_pipeline'] = time.time() - start_time_total_pipeline
    logging.info(f"Image: {image_basename} - Food analysis pipeline completed in {results['timing']['total_pipeline']:.2f} seconds.")
    return results