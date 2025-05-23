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
    logging.info(f"FOOD_ANALYZER_DEBUG: Received camera_intrinsics_key = '{camera_intrinsics_key}' (Type: {type(camera_intrinsics_key)})") # Cascade: Modified for debugging

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

    #1. Load Input Data
    try:
        t0_inputs = time.time()
        # Load image first to get dimensions for dummy depth map if needed
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Image: {image_basename} - Failed to load input image: {image_path}")
            results['error_messages'].append("Failed to load input image.")
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
             results['error_messages'].append("Invalid final depth map.")
             return results # Critical failure
        logging.info(f"Image: {image_basename} - Using depth map of shape: {depth_map.shape}") 
        results['timing']['load_inputs'] = time.time() - t0_inputs
    except Exception as e: 
        logging.error(f"Image: {image_basename} - Critical error during input data loading: {e}", exc_info=True)
        results['error_messages'].append(f"Critical input loading error: {e}")
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
                # --- SEG_DEBUG: Log loaded mask properties ---
                if segmentation_mask is not None:
                    logging.info(f"SEG_DEBUG (Loaded Mask): Shape={segmentation_mask.shape}, dtype={segmentation_mask.dtype}, "
                                 f"Min={segmentation_mask.min()}, Max={segmentation_mask.max()}, "
                                 f"Non-zero pixels={np.count_nonzero(segmentation_mask)}")
                # --- End SEG_DEBUG ---
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
            seg_num_classes_config = model_params_config.get('segmentation_num_classes', 1) # Default to 1 if not found
            seg_threshold_config = model_params_config.get('segmentation_threshold', 0.5) # Default to 0.5

            if not seg_model_path_rel or not seg_input_size_config:
                logging.error(f"Image: {image_basename} - Segmentation model path or input size missing in config. Skipping model-based segmentation.")
                results['error_messages'].append("SegModelConfigMissing;")
            else:
                t0_seg_load_model = time.time()
                seg_model_path = str(project_root / seg_model_path_rel)
                seg_interpreter, seg_input_details, seg_output_details = load_segmentation_model(seg_model_path, tuple(seg_input_size_config))
                results['timing']['segmentation_load_model'] = time.time() - t0_seg_load_model
                
                model_input_size_hw = tuple(seg_input_size_config)
                
                # Determine the target shape for the output segmentation mask
                output_target_shape_hw = None
                resize_info_log = "unknown"
                if depth_map is not None and hasattr(depth_map, 'shape') and len(depth_map.shape) >= 2 and depth_map.shape[0] > 0 and depth_map.shape[1] > 0:
                    output_target_shape_hw = depth_map.shape[:2]
                    resize_info_log = "depth map shape"
                elif image is not None and hasattr(image, 'shape') and len(image.shape) >= 2 and image.shape[0] > 0 and image.shape[1] > 0:
                    output_target_shape_hw = image.shape[:2]
                    resize_info_log = "original RGB image shape"
                else:
                    # Fallback: This case should ideally not be reached if image loading was successful.
                    # Using model_input_size_hw as a last resort.
                    logging.warning(f"Image: {image_basename} - Could not determine a valid target output shape from depth_map or image. Using model input size {model_input_size_hw} as fallback for output resize.")
                    output_target_shape_hw = model_input_size_hw 
                    resize_info_log = "model input size (fallback)"
                logging.info(f"Image: {image_basename} - Segmentation mask will be produced at target shape: {output_target_shape_hw} (based on {resize_info_log}).")
                
                t0_seg_inference = time.time()
                segmentation_mask = run_segmentation_inference(
                    seg_interpreter, seg_input_details, seg_output_details,
                    image_path, 
                    model_input_size_hw=model_input_size_hw, # Explicitly named
                    output_resize_shape_hw=output_target_shape_hw, # New argument
                    num_classes=seg_num_classes_config, # Pass num_classes
                    threshold=seg_threshold_config # Pass threshold
                )
                results['timing']['segmentation_inference'] = time.time() - t0_seg_inference
                # --- SEG_DEBUG: Log model-generated mask properties ---
                if segmentation_mask is not None:
                    logging.info(f"SEG_DEBUG (Model Output Raw - from run_segmentation_inference): "
                                 f"Shape={segmentation_mask.shape}, dtype={segmentation_mask.dtype}, "
                                 f"Min={segmentation_mask.min()}, Max={segmentation_mask.max()}, "
                                 f"Non-zero pixels={np.count_nonzero(segmentation_mask)}")
                else:
                    logging.info("SEG_DEBUG (Model Output Raw - from run_segmentation_inference): Mask is None.")
                # --- End SEG_DEBUG ---
                logging.info(f"Image: {image_basename} - Generated mask using model, shape: {segmentation_mask.shape if segmentation_mask is not None else 'None'}")
                if segmentation_mask is not None:
                    segmentation_source = "model_generated"
                else:
                    logging.warning(f"Image: {image_basename} - Model-based segmentation returned None.")
                    results['error_messages'].append("SegModelReturnedNone;")
        except Exception as e_seg_model:
            logging.exception(f"Image: {image_basename} - Error during model-based segmentation: {e_seg_model}")
            results['error_messages'].append(f"SegModelError: {e_seg_model};")

    if segmentation_mask is not None:
        results['segmentation_mask_shape'] = segmentation_mask.shape
        logging.info(f"Image: {image_basename} - Final segmentation mask shape: {segmentation_mask.shape} (Source: {segmentation_source})")
        if save_steps and output_dir:
            try:
                mask_filename = os.path.join(output_dir, f"{Path(image_path).stem}_final_mask.png")
                # Ensure mask is in a savable format (e.g., 0-255 uint8)
                saveable_mask = (segmentation_mask.astype(np.uint8) * 255) if segmentation_mask.dtype == bool else segmentation_mask.astype(np.uint8)
                cv2.imwrite(mask_filename, saveable_mask)
                logging.info(f"Image: {image_basename} - Saved final segmentation mask to {mask_filename}")
            except Exception as e_save_mask:
                logging.warning(f"Image: {image_basename} - Failed to save final segmentation mask: {e_save_mask}")
    else:
        logging.warning(f"Image: {image_basename} - No segmentation mask available to record shape or save.")
        results['segmentation_mask_shape'] = None

    results['timing']['segmentation_overall'] = time.time() - t0_seg_overall
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
        results['error_messages'].append(f"ClassImgPrepError: {e_crop};")
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
            class_labels_path_rel = model_params_config.get('classification_labels') # Corrected to use 'classification_labels'
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

            # Debugging classification config
            logging.info(f"Image: {image_basename} - Checking classification config values:")
            logging.info(f"  class_model_path_rel: '{class_model_path_rel}' (type: {type(class_model_path_rel)}) (Exists relative to project: {Path(project_root / class_model_path_rel).exists() if class_model_path_rel else False})")
            logging.info(f"  class_input_size_config: {class_input_size_config} (type: {type(class_input_size_config)}) (Valid: {isinstance(class_input_size_config, list) and len(class_input_size_config) == 2})")
            logging.info(f"  class_labels_path_rel: '{class_labels_path_rel}' (type: {type(class_labels_path_rel)}) (Exists relative to project: {Path(project_root / class_labels_path_rel).exists() if class_labels_path_rel else False})")

            config_check_class_model = bool(class_model_path_rel and Path(project_root / class_model_path_rel).exists())
            config_check_input_size = bool(class_input_size_config and isinstance(class_input_size_config, list) and len(class_input_size_config) == 2)
            config_check_labels = bool(class_labels_path_rel and Path(project_root / class_labels_path_rel).exists())
            
            if not (config_check_class_model and config_check_input_size and config_check_labels):
                logging.error(f"Image: {image_basename} - Classification model path, input size, or label map missing/invalid in config. Skipping classification.")
                results['error_messages'].append("ClassModelConfigMissingOrInvalid;")
            else:
                t0_class_load_model = time.time()
                class_model_path = str(project_root / class_model_path_rel)
                class_label_map_path = str(project_root / class_labels_path_rel)

                class_model, class_input_details, class_output_details, class_labels = load_classification_model(
                    class_model_path,
                    class_label_map_path
                )
                results['timing']['classification_load_model'] = time.time() - t0_class_load_model
                target_class_size = tuple(class_input_size_config)

                if class_model and class_labels is not None:
                    t0_class_inference = time.time()
                    classified_label, classified_confidence = run_classification_inference(
                        class_model,                    # interpreter
                        class_input_details,            # input_details
                        class_output_details,           # output_details
                        target_class_size,              # model_input_size_hw (positional)
                        clf_architecture,               # architecture (positional)
                        class_labels=class_labels,      # class_labels (keyword)
                        image_data=cropped_image_for_classification # image_data (keyword)
                    )
                    results['timing']['classification_inference'] = time.time() - t0_class_inference

                    if classified_label and classified_confidence is not None:
                        raw_food_label = classified_label
                        raw_confidence = float(classified_confidence)
                        logging.info(f"Image: {image_basename} - Raw Classification: '{raw_food_label}' (Confidence: {raw_confidence:.2f})")

                        if raw_confidence < confidence_threshold:
                            food_label = f"Uncertain: {raw_food_label}"
                            confidence = raw_confidence # Store raw confidence even if uncertain
                            logging.warning(f"Image: {image_basename} - Classification confidence {raw_confidence:.2f} is below threshold {confidence_threshold}. Final label: '{food_label}'")
                            results['classification_status'] = f"BelowConfidenceThreshold (Score: {raw_confidence:.2f})"
                        else:
                            food_label = raw_food_label
                            confidence = raw_confidence
                            results['classification_status'] = f"Confident (Score: {confidence:.2f})"
                            logging.info(f"Image: {image_basename} - Final Classification: '{food_label}' (Confidence: {confidence:.2f})")
                    else:
                        logging.warning(f"Image: {image_basename} - Classification model returned None for label or confidence.")
                        results['error_messages'].append("ClassModelReturnNone;")
                        results['classification_status'] = "Error (ModelReturnNone)"
                else:
                    logging.error(f"Image: {image_basename} - Failed to load classification model or labels. Skipping classification.")
                    results['error_messages'].append("ClassModelLoadFail;")
                    results['classification_status'] = "Error (ModelLoadFail)"

        except Exception as e_class_model:
            logging.exception(f"Image: {image_basename} - Error during classification: {e_class_model}")
            results['error_messages'].append(f"ClassModelError: {e_class_model};")
            results['classification_status'] = f"Error ({e_class_model})"
    else: # known_food_class was not provided AND cropped_image_for_classification is None
        logging.warning(f"Image: {image_basename} - Skipping classification as no valid image for classification and no known_food_class provided.")
        results['classification_status'] = "Skipped (NoImageForClf)"

    results['food_label'] = food_label
    results['confidence'] = confidence
    results['timing']['classification_overall'] = time.time() - t0_class_overall
    logging.info(f"Image: {image_basename} - Classification phase complete. Determined Food Label: '{food_label}', Confidence: {confidence:.2f}")

    # 5. Volume Estimation
    t0_vol_overall = time.time()
    calculated_volume_cm3 = 0.0
    volume_method_used = "N/A"

    if volume_estimation_method == 'depth':
        if depth_map is not None and segmentation_mask is not None and segmentation_mask.shape[:2] == depth_map.shape[:2]:
            logging.info(f"Image: {image_basename} - Attempting volume estimation using 'depth' method.")
            t0_vol_depth_voxel = time.time()
            try:
                # Prepare volume estimation config, merging defaults with overrides
                vol_est_params = volume_estimation_config if volume_estimation_config is not None else {}
                
                # Determine debug_output_path for estimate_volume_from_depth
                debug_output_path_volume = None
                if save_steps and output_dir:
                    debug_output_name = f"{Path(image_path).stem}_volume_debug"
                    debug_output_path_volume = os.path.join(output_dir, debug_output_name)
                    os.makedirs(debug_output_path_volume, exist_ok=True)
                    logging.info(f"Volume estimator debug output will be saved to: {debug_output_path_volume}")

                # --- Debugging: Log mask properties before passing to volume estimator ---
                if segmentation_mask is not None:
                    logging.info(f"FOOD_ANALYZER_DEBUG: Passing segmentation_mask to volume_estimator with: "
                                 f"Shape={segmentation_mask.shape}, dtype={segmentation_mask.dtype}, "
                                 f"Non-zero pixels={np.count_nonzero(segmentation_mask)}")
                else:
                    logging.info("FOOD_ANALYZER_DEBUG: segmentation_mask is None before calling volume_estimator.")
                # --- End Debugging ---

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
        # Fallback to point cloud convex hull if mesh failed or not provided, but method is 'mesh'
        # This part might need review: if method is 'mesh', should it only use mesh?
        # For now, keeping existing potential fallback to point_cloud_path if mesh is primary but fails/missing.
        elif point_cloud_path and os.path.exists(point_cloud_path) and depth_map is not None and segmentation_mask is not None:
            logging.info(f"Image: {image_basename} - 'mesh' method selected, but no mesh file. Trying point cloud file: {point_cloud_path} with convex hull.")
            # This reuses the old convex hull logic. Consider if this is desired for 'mesh' method.
            # Or if 'mesh' should strictly mean .obj files.
            # The following is effectively the old 'depth_map_to_masked_points' + 'estimate_volume_convex_hull'
            # which might be redundant if 'depth' method is preferred for non-mesh scenarios.
            t0_vol_depth_hull = time.time()
            try:
                # Assuming camera_intrinsics are needed for depth_map_to_masked_points if it's used.
                # The original depth_map_to_masked_points might not have used full intrinsics.
                # This part needs careful review against volume_helpers.py content.
                # For now, let's assume it can proceed or we log a warning.
                # Placeholder for actual camera intrinsics if needed by depth_map_to_masked_points
                # This is a bit tricky as the old convex hull method might not have used full intrinsics from config
                # For simplicity, if we reach here, it implies a configuration that might be suboptimal
                # as the new 'depth' method is more robust for depth map based calculation.
                logging.warning(f"Image: {image_basename} - Using convex hull from point cloud file as fallback for 'mesh' method. This path might be deprecated in favor of 'depth' method for non-OBJ inputs.")
                # This part of the logic might be simplified or removed if 'depth' method is the sole path for non-mesh volume.
                # The original call was: points = depth_map_to_masked_points(depth_map, segmentation_mask, camera_params_from_config)
                # And then: calculated_volume_cm3 = estimate_volume_convex_hull(points)
                # This is complex to replicate here without knowing exact camera_params_from_config structure.
                # For now, let's log it as not fully supported in this refactor if mesh is missing.
                logging.warning(f"Image: {image_basename} - Fallback to convex hull from PLY for 'mesh' method is not fully implemented with new intrinsics flow. Please use 'depth' method or provide a valid .obj file for 'mesh' method.")
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
                food_item_label=clean_food_label_for_lookup,
                volume_cm3=calculated_volume_cm3,
                density_db_path=config.get('density_db_path', str(project_root / 'data' / 'databases' / 'food_density_db.json')),
                usda_api_key=usda_api_key,
                cache_dir=str(project_root / '.cache' / 'usda_api_cache')
            )
            if nutritional_info:
                logging.info(f"Image: {image_basename} - Nutritional lookup successful for '{clean_food_label_for_lookup}'. Calories: {nutritional_info.get('total_calories', 'N/A')} kcal.")
                # Log more details if needed, e.g., nutritional_info['source_used']
                results['nutritional_info_status'] = f"Success (Source: {nutritional_info.get('source_used', 'Unknown')})"
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

    # ... (existing logic for density lookup and mass calculation) ...

    results['timing']['total_pipeline'] = time.time() - start_time_total_pipeline
    logging.info(f"Image: {image_basename} - Food analysis pipeline completed in {results['timing']['total_pipeline']:.2f} seconds.")
    
    # Save final mask if save_steps is True and mask exists
    if save_steps and output_dir and segmentation_mask is not None:
        try:
            mask_filename = os.path.join(output_dir, f"{Path(image_path).stem}_final_mask.png")
            # Ensure mask is in a savable format (e.g., 0-255 uint8)
            saveable_mask = (segmentation_mask.astype(np.uint8) * 255) if segmentation_mask.dtype == bool else segmentation_mask.astype(np.uint8)
            cv2.imwrite(mask_filename, saveable_mask)
            logging.info(f"Image: {image_basename} - Saved final segmentation mask to {mask_filename}")
        except Exception as e:
            logging.warning(f"Image: {image_basename} - Failed to save final segmentation mask: {e}")

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
    
    nut_info = results.get('nutritional_info')
    summary_calories_per_100g = "N/A"
    summary_nutrition_source = "N/A"
    summary_total_calories = "N/A"

    if nut_info:
        summary_calories_per_100g = nut_info.get('calories_kcal_per_100g', 'N/A')
        summary_nutrition_source = nut_info.get('source_used', 'N/A')
        summary_total_calories = nut_info.get('total_calories', 'N/A')
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