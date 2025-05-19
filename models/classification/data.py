import json
import os
import tensorflow as tf
from typing import Tuple, List, Dict, Optional
import numpy as np # Not strictly used in the refactored load_classification_data
import logging
from sklearn.model_selection import train_test_split
import pathlib
# import glob # No longer needed
from tqdm import tqdm # Import tqdm
import trimesh # For point cloud processing
import traceback # For detailed error traceback
import random # Import for random sampling
from collections import Counter # Ensure Counter is imported

# Dynamic import for preprocess_input
_PREPROCESS_FN_CACHE = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent.parent

def compute_class_weights(y_true: List[int], num_classes: int) -> Optional[Dict[int, float]]:
    """
    Computes class weights for imbalanced datasets.
    Args:
        y_true: List of true labels (integers).
        num_classes: Total number of unique classes.
    Returns:
        A dictionary mapping class indices to their calculated weights, or None if input is empty.
    """
    if not y_true:
        logger.warning("Cannot compute class weights: y_true is empty.")
        return None

    counts = Counter(y_true)
    total_samples = len(y_true)
    class_weights = {}

    for class_idx in range(num_classes):
        if counts[class_idx] > 0:
            weight = total_samples / (num_classes * counts[class_idx])
            class_weights[class_idx] = weight
        else:
            # If a class is not present in y_true, its weight is not added.
            # Keras handles missing keys in class_weight dict by defaulting to a weight of 1.
            # Alternatively, one could assign a default weight (e.g., 1.0 or 0.0)
            # logger.debug(f"Class {class_idx} not found in y_true, weight not computed.")
            pass 
            
    if not class_weights: # Should not happen if y_true is not empty and num_classes > 0
        logger.warning("Class weights dictionary is empty after computation. Check input y_true and num_classes.")
        return None
        
    return class_weights

def _get_preprocess_fn(architecture: str):
    """Dynamically imports and returns the correct preprocess_input function."""
    global _PREPROCESS_FN_CACHE
    if architecture in _PREPROCESS_FN_CACHE:
        return _PREPROCESS_FN_CACHE[architecture]

    preprocess_input_fn = None # Initialize to ensure it's always defined
    if architecture == "MobileNet": # MobileNetV1
        from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_input_fn
    elif architecture == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_fn
    elif architecture == "MobileNetV3Small" or architecture == "MobileNetV3Large":
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_input_fn
    elif architecture.startswith("EfficientNetV2"):
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_input_fn
    elif architecture.startswith("EfficientNet"): # EfficientNetB0-B7
            from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_fn
    elif architecture.startswith("ResNet") and "V2" not in architecture: # ResNet50, ResNet101, ResNet152
            from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_fn
    elif architecture.startswith("ResNet") and "V2" in architecture: # ResNet50V2 etc.
            from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_fn
    elif architecture.startswith("ConvNeXt"):
            from tensorflow.keras.applications.convnext import preprocess_input as preprocess_input_fn
    # Add other architectures explicitly if needed for future use
    else:
        logger.warning(f"Preprocessing function not explicitly defined or imported for {architecture}. Using generic scaling (image / 255.0). This may be suboptimal.")
        preprocess_input_fn = lambda x: x / 255.0 # Fallback generic scaling
    
    _PREPROCESS_FN_CACHE[architecture] = preprocess_input_fn
    return preprocess_input_fn

def _build_augmentation_pipeline(config: Dict) -> Optional[tf.keras.Sequential]:
    """Builds a data augmentation pipeline from the config."""
    if not config.get('augmentation', {}).get('enabled', False):
        return None

    aug_config = config['augmentation']
    pipeline = tf.keras.Sequential(name="augmentation_pipeline")
    pipeline.add(tf.keras.layers.Input(shape=(*config['image_size'], 3)))

    if aug_config.get('horizontal_flip', False):
        pipeline.add(tf.keras.layers.RandomFlip("horizontal"))
    if aug_config.get('rotation_range', 0) > 0:
        factor = aug_config['rotation_range'] / 360.0
        pipeline.add(tf.keras.layers.RandomRotation(factor))
    if aug_config.get('shear_range', 0) > 0:
        # pipeline.add(tf.keras.layers.RandomShear(intensity=aug_config['shear_range']))
        logger.warning("RandomShear augmentation is currently commented out due to potential TF version incompatibility.")
    if aug_config.get('zoom_range', 0) > 0:
        pipeline.add(tf.keras.layers.RandomZoom(height_factor=aug_config['zoom_range'], width_factor=aug_config['zoom_range']))
    if 'brightness_range' in aug_config and aug_config['brightness_range']:
        if isinstance(aug_config['brightness_range'], list) and len(aug_config['brightness_range']) == 2:
            factor = max(abs(1.0 - aug_config['brightness_range'][0]), abs(aug_config['brightness_range'][1] - 1.0))
        else:
            factor = aug_config['brightness_range'] 
        pipeline.add(tf.keras.layers.RandomBrightness(factor=factor))

    if aug_config.get("width_shift_range", 0) > 0:
        pipeline.add(tf.keras.layers.RandomTranslation(height_factor=0, width_factor=aug_config['width_shift_range']))
    if aug_config.get("height_shift_range", 0) > 0:
        pipeline.add(tf.keras.layers.RandomTranslation(height_factor=aug_config['height_shift_range'], width_factor=0))
    pipeline.add(tf.keras.layers.Resizing(config['image_size'][0], config['image_size'][1]))
    return pipeline

# MixUp function (to be applied after batching)
@tf.function
def mixup(batch_images: tf.Tensor, batch_labels: tf.Tensor, alpha: float, num_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies MixUp augmentation to a batch of images and labels.

    Args:
        batch_images: A batch of images (batch_size, height, width, channels).
        batch_labels: A batch of labels (batch_size,). Assumed to be integer class indices.
        alpha: The alpha parameter for the Beta distribution (controls mixing strength).
        num_classes: The total number of classes for one-hot encoding the labels.

    Returns:
        A tuple of (mixed_images, mixed_labels).
    """
    batch_size = tf.shape(batch_images)[0]
    # Ensure labels are one-hot encoded for mixing
    # If labels are already one-hot, this tf.one_hot will need to be adjusted or skipped.
    # Assuming batch_labels are sparse (indices) based on sparse_categorical_crossentropy typical usage.
    labels_one_hot = tf.one_hot(tf.cast(batch_labels, dtype=tf.int32), depth=num_classes)

    # Sample lambda from a Beta distribution
    # Gamma(alpha, 1) / (Gamma(alpha,1) + Gamma(alpha,1)) if alpha > 0 else 1
    # tf.random.gamma requires alpha > 0.
    # If alpha is 0, it implies no mixing, so lambda is 1.
    if alpha > 0.0:
        beta_dist = tf.compat.v1.distributions.Beta(alpha, alpha)
        lambda_val = beta_dist.sample(batch_size)
        # Reshape lambda for broadcasting: (batch_size,) -> (batch_size, 1, 1, 1) for images, (batch_size, 1) for labels
        lambda_img = tf.reshape(lambda_val, [batch_size, 1, 1, 1])
        lambda_lbl = tf.reshape(lambda_val, [batch_size, 1])
    else: # No mixing if alpha is 0
        lambda_img = tf.ones((batch_size, 1, 1, 1), dtype=tf.float32)
        lambda_lbl = tf.ones((batch_size, 1), dtype=tf.float32)

    # Shuffle the batch to create pairs for mixing
    # Use tf.random.shuffle to ensure reproducibility with global/op seeds if set
    shuffled_indices = tf.random.shuffle(tf.range(batch_size))
    
    shuffled_images = tf.gather(batch_images, shuffled_indices)
    shuffled_labels_one_hot = tf.gather(labels_one_hot, shuffled_indices)

    # Perform MixUp
    mixed_images = lambda_img * batch_images + (1.0 - lambda_img) * shuffled_images
    mixed_labels = lambda_lbl * labels_one_hot + (1.0 - lambda_lbl) * shuffled_labels_one_hot
    
    return mixed_images, mixed_labels

# CutMix function (to be applied after batching)
@tf.function
def cutmix(batch_images: tf.Tensor, batch_labels: tf.Tensor, alpha: float, num_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies CutMix augmentation to a batch of images and labels.

    Args:
        batch_images: A batch of images (batch_size, height, width, channels).
        batch_labels: A batch of labels. Assumed to be one-hot if coming after MixUp,
                      or sparse (indices) if MixUp is disabled.
        alpha: The alpha parameter for the Beta distribution (controls patch size).
        num_classes: The total number of classes for one-hot encoding if labels are sparse.

    Returns:
        A tuple of (mixed_images, mixed_labels).
    """
    batch_size = tf.shape(batch_images)[0]
    image_h = tf.shape(batch_images)[1]
    image_w = tf.shape(batch_images)[2]

    # Ensure labels are one-hot. If MixUp was applied, labels are already one-hot.
    # If MixUp was not applied, batch_labels are likely sparse.
    if len(tf.shape(batch_labels)) == 1: # Sparse labels (batch_size,)
        labels_one_hot = tf.one_hot(tf.cast(batch_labels, dtype=tf.int32), depth=num_classes)
    else: # Already one-hot (batch_size, num_classes)
        labels_one_hot = batch_labels

    # Sample lambda from a Beta distribution
    if alpha > 0.0:
        beta_dist = tf.compat.v1.distributions.Beta(alpha, alpha)
        lambda_val = beta_dist.sample(1)[0] # Sample a single lambda for the batch
    else: # No mixing if alpha is 0
        return batch_images, labels_one_hot # Or original batch_labels if it was sparse

    # Shuffle the batch to create pairs for mixing
    shuffled_indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(batch_images, shuffled_indices)
    shuffled_labels_one_hot = tf.gather(labels_one_hot, shuffled_indices)

    # Calculate patch coordinates
    cut_ratio = tf.math.sqrt(1.0 - lambda_val) # Proportional to patch size
    cut_h = tf.cast(cut_ratio * tf.cast(image_h, tf.float32), tf.int32)
    cut_w = tf.cast(cut_ratio * tf.cast(image_w, tf.float32), tf.int32)

    # Uniformly sample the center of the patch
    center_y = tf.random.uniform(shape=[], minval=0, maxval=image_h, dtype=tf.int32)
    center_x = tf.random.uniform(shape=[], minval=0, maxval=image_w, dtype=tf.int32)

    y1 = tf.clip_by_value(center_y - cut_h // 2, 0, image_h)
    y2 = tf.clip_by_value(center_y + cut_h // 2, 0, image_h)
    x1 = tf.clip_by_value(center_x - cut_w // 2, 0, image_w)
    x2 = tf.clip_by_value(center_x + cut_w // 2, 0, image_w)

    # Create mask
    # Mask has 1s where the patch from the shuffled image should be, 0s otherwise.
    patch_height = y2 - y1
    patch_width = x2 - x1

    # Ensure patch area is not zero before division
    actual_patch_area = tf.cast(patch_height * patch_width, tf.float32)
    total_area = tf.cast(image_h * image_w, tf.float32)
    
    # Adjust lambda based on the actual patch area that fits within the image
    # lambda_val here represents the proportion of the first image in the mix
    lambda_adjusted = 1.0 - (actual_patch_area / tf.maximum(total_area, 1e-8)) # Avoid division by zero if total_area is somehow 0

    # Create the mask for pasting
    mask_shape = (batch_size, image_h, image_w, 1) # Mask will be broadcast to channels
    padding = [[y1, image_h - y2], [x1, image_w - x2]]
    mask_patch = tf.ones([batch_size, patch_height, patch_width, 1], dtype=tf.float32)
    # Pad the patch to create the full image mask. Padded areas are 0.
    # Correct padding for tf.pad: it pads *around* the patch.
    # We want to create a mask that is 1 in the patch area and 0 outside.
    # A more direct way to create the mask:
    mask_y = tf.logical_and(tf.range(image_h)[:, tf.newaxis] >= y1, tf.range(image_h)[:, tf.newaxis] < y2)
    mask_x = tf.logical_and(tf.range(image_w)[tf.newaxis, :] >= x1, tf.range(image_w)[tf.newaxis, :] < x2)
    mask_2d = tf.cast(tf.logical_and(mask_y, mask_x), tf.float32) # (H, W)
    mask = mask_2d[tf.newaxis, :, :, tf.newaxis] # (1, H, W, 1)
    mask = tf.tile(mask, [batch_size, 1, 1, 1]) # (B, H, W, 1)

    # Apply CutMix
    # Original image where mask is 0, shuffled image where mask is 1.
    cutmix_images = batch_images * (1.0 - mask) + shuffled_images * mask
    cutmix_labels = labels_one_hot * lambda_adjusted + shuffled_labels_one_hot * (1.0 - lambda_adjusted)

    return cutmix_images, cutmix_labels

def load_classification_data(config: Dict) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset], int, Dict[int, str], Optional[Dict[int, float]]]:
    """
    Load and prepare classification data using a metadata.json file.
    Refactored to read from metadata_path, use label_map_path from paths config.
    Args:
        config: Dictionary from classification/config.yaml.
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, num_classes, index_to_label_map, class_weights_dict).
    """
    project_root = _get_project_root()

    try:
        # Essential paths from config
        paths_config = config['paths']
        data_cfg = config['data']
        model_cfg = config['model']

        # Construct metadata_path and label_map_path using paths_config
        # Ensure paths are absolute if they are relative to project root or a specific data_dir
        base_data_dir = project_root / paths_config.get('data_dir', 'data/classification')

        metadata_filename = paths_config.get('metadata_filename', 'metadata.json')
        metadata_path = base_data_dir / metadata_filename

        label_map_filename = paths_config.get('label_map_filename', 'label_map.json')
        label_map_path = base_data_dir / label_map_filename

        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            return None, None, None, 0, {}, {}
        if not label_map_path.exists():
            logger.error(f"Label map file not found: {label_map_path}")
            return None, None, None, 0, {}, {}

        with open(metadata_path, 'r') as f:
            all_items = json.load(f)
        with open(label_map_path, 'r') as f:
            label_map_loaded = json.load(f)
            # Invert the map and convert index to int: {class_name: index_int}
            label_to_index = {name: int(idx) for idx, name in label_map_loaded.items()}
            index_to_label = {int(idx): name for idx, name in label_map_loaded.items()}
    
        num_classes_from_map = len(label_to_index)
        # Define the authoritative number of classes for the model and one-hot encoding
        model_configured_num_classes = model_cfg.get('num_classes', num_classes_from_map)
        logger.info(f"Model will be configured for {model_configured_num_classes} classes (derived from map: {num_classes_from_map}, config override: {model_cfg.get('num_classes')}).")

        image_size = tuple(data_cfg['image_size'])
        batch_size = data_cfg['batch_size']
        split_ratio = data_cfg.get('split_ratio', 0.2) # Default to 0.2 if not specified

        # New: Configs for additional modalities
        use_depth_map = data_cfg.get('use_depth_map', False)
        depth_map_dir_name = data_cfg.get('depth_map_dir_name', 'depth')
        use_point_cloud = data_cfg.get('use_point_cloud', False)
        point_cloud_root_dir = data_cfg.get('point_cloud_root_dir', '')
        pc_sampling_rate_dir = data_cfg.get('point_cloud_sampling_rate_dir', '')
        pc_suffix = data_cfg.get('point_cloud_suffix', '_sampled_1.ply')

        # Prepare lists for all modalities
        all_rgb_paths = []
        all_labels_numeric = [] # Initialize all_labels_numeric
        all_depth_paths = []
        all_pc_paths = []    

        skipped_missing_path = 0
        skipped_missing_label = 0
        processed_count = 0

        # Progress bar setup
        progress_bar = tqdm(all_items, desc="Processing metadata items", unit="item")

        for item in progress_bar:
            # Check for explicit image_path first
            rgb_path_str = item['image_path']
            # Basic validation: ensure rgb_path_str is not empty and exists
            if not rgb_path_str or not pathlib.Path(rgb_path_str).exists():
                logger.warning(f"RGB path missing or file not found: '{rgb_path_str}' for item {item.get('instance_name', 'N/A')}. Skipping item.")
                skipped_missing_path += 1
                continue
            
            all_rgb_paths.append(rgb_path_str)
            class_name = item.get('class_name')
            current_label = None
            if class_name and class_name in label_to_index:
                current_label = label_to_index[class_name]
            elif class_name:
                logger.warning(f"Class name '{class_name}' found in metadata item {rgb_path_str} but not in label_map. Skipping item.")
                skipped_missing_label += 1
                # Remove the already added rgb_path if label is invalid
                if all_rgb_paths and all_rgb_paths[-1] == rgb_path_str:
                    all_rgb_paths.pop()
                continue 
            else:
                logger.warning(f"Missing 'class_name' in metadata for item {rgb_path_str}. Skipping item.")
                skipped_missing_label += 1
                if all_rgb_paths and all_rgb_paths[-1] == rgb_path_str:
                    all_rgb_paths.pop()
                continue
            all_labels_numeric.append(current_label)
            
            # instance_id or instance_name is used for constructing depth/pc paths if they follow a pattern like .../class_name/instance_id_or_name/...
            # Prioritize 'instance_id', then 'instance_name', then a default if neither exists (though usually one should for multi-modal)
            instance_identifier = item.get('instance_id', item.get('instance_name', ''))
            # all_instance_ids.append(instance_identifier) # Though this list itself isn't directly used for stratified split anymore

            depth_path_str = "" # Default to empty string
            if use_depth_map:
                # Path for depth map: data_root / class_name / instance_name / depth_dir_name / frame_number.png (example)
                # We need a robust way to get the frame_filename or identifier from rgb_path_str
                frame_filename = pathlib.Path(rgb_path_str).name
                # Assuming depth maps are in a subfolder relative to the RGB image's folder, or a parallel folder structure
                # Example: E:\_MetaFood3D_new_RGBD_videos\RGBD_videos\Carrot\carrot_2\original\72.jpg
                # Depth:   E:\_MetaFood3D_new_RGBD_videos\RGBD_videos\Carrot\carrot_2\depth\72.png (if .png)
                # This logic assumes depth map has the same stem but possibly different suffix and parent dir named depth_map_dir_name
                potential_depth_path = pathlib.Path(rgb_path_str).parent.parent / depth_map_dir_name / frame_filename
                # More flexible: allow depth_map_dir_name to be relative to instance folder or class folder
                # This example assumes depth_map_dir_name is a sibling to 'original' if rgb_path_str contains 'original'
                # This might need adjustment based on exact dataset structure.
                # For now, let's assume a simpler structure for classification where depth might be alongside RGB or given directly.
                # If 'depth_path' is in metadata, use it. Otherwise, try to derive.
                if 'depth_path' in item and item['depth_path']:
                    potential_depth_path = pathlib.Path(item['depth_path'])
                elif instance_identifier: # Try to derive if instance_identifier is present
                     # This derivation is a placeholder and likely needs to match your *actual* structure for classification data
                    potential_depth_path = pathlib.Path(rgb_path_str).parent.parent / depth_map_dir_name / frame_filename # Simplified assumption
                else: # Cannot derive without instance identifier or explicit path
                    potential_depth_path = None

                if potential_depth_path and potential_depth_path.exists():
                    depth_path_str = str(potential_depth_path)
                else:
                    logger.debug(f"Depth map not found for {rgb_path_str} (tried {potential_depth_path}). Will use zeros.")
            all_depth_paths.append(depth_path_str)

            pc_path_str = "" # Default to empty string
            if use_point_cloud:
                food_class_from_label = item['class_name'] 
                # PC path construction logic (assuming it's correct as per previous steps)
                # Point clouds are often per-instance rather than per-frame.
                # The metadata should ideally link an RGB frame to its corresponding (potentially single) instance point cloud.
                if 'point_cloud_path' in item and item['point_cloud_path']:
                    potential_pc_path = pathlib.Path(item['point_cloud_path'])
                elif instance_identifier: # Try to derive if instance_identifier is present
                    potential_pc_path = pathlib.Path(point_cloud_root_dir) / pc_sampling_rate_dir / food_class_from_label / instance_identifier / (instance_identifier + pc_suffix)
                else:
                    potential_pc_path = None
                
                if potential_pc_path and potential_pc_path.exists():
                    pc_path_str = str(potential_pc_path)
                else:
                    logger.debug(f"Point cloud not found for {rgb_path_str} (food_class: {food_class_from_label}, instance: {instance_identifier}, tried {potential_pc_path}). Will use zeros.")
            all_pc_paths.append(pc_path_str)

            processed_count += 1
            if processed_count % 20000 == 0: # Log progress every 20000 items for full dataset
                logger.info(f"Processed {processed_count}/{len(all_items)} metadata items...")

        logger.info(f"Finished processing metadata. {processed_count} items prepared for dataset creation.")
        logger.info(f"Skipped {skipped_missing_path} items due to missing RGB path.")
        logger.info(f"Skipped {skipped_missing_label} items due to missing or invalid class_name.")

        if not all_rgb_paths or not all_labels_numeric:
            logger.error("No valid data items found after processing metadata. Aborting.")
            return None, None, None, 0, {}, {}

        # --- (Optional) Subset Sampling for Debugging/Testing ---
        debug_max_total_samples = data_cfg.get('debug_max_total_samples', None)
        if debug_max_total_samples and isinstance(debug_max_total_samples, int) and debug_max_total_samples > 0:
            if debug_max_total_samples < len(all_rgb_paths):
                logger.info(f"Debug mode: Sampling {debug_max_total_samples} images from the total {len(all_rgb_paths)} images for quick testing.")
                paired_data = list(zip(all_rgb_paths, all_labels_numeric, all_depth_paths, all_pc_paths))
                # Ensure sample size is not larger than population
                actual_sample_size = min(debug_max_total_samples, len(paired_data))
                if actual_sample_size < debug_max_total_samples:
                    logger.warning(f"Requested debug_max_total_samples {debug_max_total_samples} but only {len(paired_data)} samples available. Using {actual_sample_size}.")
                
                if actual_sample_size > 0:
                    sampled_pairs = random.sample(paired_data, actual_sample_size)
                    all_rgb_paths, all_labels_numeric, all_depth_paths, all_pc_paths = zip(*sampled_pairs)
                    all_rgb_paths = list(all_rgb_paths)
                    all_labels_numeric = list(all_labels_numeric)
                    all_depth_paths = list(all_depth_paths)
                    all_pc_paths = list(all_pc_paths)
                    logger.info(f"Sampled down to {len(all_rgb_paths)} images for debug run.")

                    # --- Stratification Fix for Small Subsets ---
                    if len(all_labels_numeric) > 0: # Proceed only if there are labels to count
                        label_counts = Counter(all_labels_numeric)
                        min_samples_per_class_for_split = 2 # For StratifiedShuffleSplit or train_test_split with stratification
                        
                        classes_to_remove = {label for label, count in label_counts.items() if count < min_samples_per_class_for_split}
                        
                        if classes_to_remove:
                            logger.warning(
                                f"Debug mode: To ensure stratification is possible, removing classes with < {min_samples_per_class_for_split} samples from the debug subset."
                            )
                            logger.warning(f"Classes to remove: {classes_to_remove}")
                            
                            # Create new lists excluding the underrepresented classes
                            filtered_rgb_paths = []
                            filtered_labels_numeric = []
                            filtered_depth_paths = []
                            filtered_pc_paths = []
                            
                            for i in range(len(all_labels_numeric)):
                                if all_labels_numeric[i] not in classes_to_remove:
                                    filtered_rgb_paths.append(all_rgb_paths[i])
                                    filtered_labels_numeric.append(all_labels_numeric[i])
                                    filtered_depth_paths.append(all_depth_paths[i])
                                    filtered_pc_paths.append(all_pc_paths[i])
                            
                            if len(filtered_labels_numeric) < len(all_labels_numeric):
                                logger.info(f"Removed {len(all_labels_numeric) - len(filtered_labels_numeric)} samples belonging to underrepresented classes.")
                                all_rgb_paths = filtered_rgb_paths
                                all_labels_numeric = filtered_labels_numeric
                                all_depth_paths = filtered_depth_paths
                                all_pc_paths = filtered_pc_paths
                                logger.info(f"Final debug subset size after filtering for stratification: {len(all_rgb_paths)} samples.")
                            else:
                                logger.info("No classes needed removal for stratification.")
                        else:
                            logger.info("Debug subset has sufficient samples per class for stratification.")
                    else:
                        logger.warning("Debug subset has no labels after sampling. Stratification checks skipped.")
                else:
                    logger.warning(f"Debug mode: No samples available after attempting to sample {debug_max_total_samples}. Skipping subset processing.")
            else:
                logger.info(f"debug_max_total_samples ({debug_max_total_samples}) is >= total images ({len(all_rgb_paths)}). Using all images.")
        # --- End Subset Sampling & Stratification Fix ---

        if not all_rgb_paths or not all_labels_numeric:
            logger.error("No data remains after potential debug sampling and filtering. Aborting data loading.")
            return None, None, None, 0, {}, {}

        # Determine if this is a debug run for split logic
        is_debug_run = data_cfg.get('debug_max_total_samples', None) is not None

        # Convert to TensorFlow constants for splitting
        all_rgb_paths_tf = tf.constant(all_rgb_paths)

        path_tuples = list(zip(all_rgb_paths, all_depth_paths, all_pc_paths))
        all_labels_numeric_tf = tf.constant(all_labels_numeric, dtype=tf.int32)

        # Initial split: Train vs. Temp (Validation + Test)
        # Stratify by all_labels_numeric if possible
        num_unique_classes_total = len(set(all_labels_numeric))
        
        # Calculate actual split sizes for the first split
        first_split_temp_size_abs = int(len(path_tuples) * split_ratio)
        # Ensure temp_size is at least 1 if len(path_tuples) * split_ratio is very small but non-zero, and total samples > 1
        if len(path_tuples) > 1 and first_split_temp_size_abs == 0 and (len(path_tuples) * split_ratio) > 0:
            first_split_temp_size_abs = 1 
        first_split_train_size_abs = len(path_tuples) - first_split_temp_size_abs

        stratify_for_first_split = all_labels_numeric
        if is_debug_run and (first_split_temp_size_abs < num_unique_classes_total or first_split_train_size_abs < num_unique_classes_total):
            logger.warning(
                f"Debug mode: Disabling stratification for train/temp split. TEMP_SIZE ({first_split_temp_size_abs}) or TRAIN_SIZE ({first_split_train_size_abs}) "
                f"is smaller than N_UNIQUE_CLASSES ({num_unique_classes_total})."
            )
            stratify_for_first_split = None
        elif first_split_temp_size_abs == 0 or first_split_train_size_abs == 0: # Handles cases where one split would be empty
             logger.warning(
                f"Warning: A split part (train or temp) would be empty (temp_size={first_split_temp_size_abs}, train_size={first_split_train_size_abs}). "
                f"Disabling stratification for train/temp split."
            )
             stratify_for_first_split = None

        if len(path_tuples) < 2: # Cannot split if less than 2 samples
            logger.warning(f"Total samples ({len(path_tuples)}) less than 2. Assigning all to train, val/test will be empty.")
            train_indices = np.arange(len(path_tuples))
            temp_indices = []
        else:
            try:
                train_indices, temp_indices = train_test_split(
                    np.arange(len(path_tuples)),
                    test_size=split_ratio, 
                    random_state=data_cfg.get('random_seed', 42),
                    stratify=stratify_for_first_split
                )
            except ValueError as e:
                logger.error(f"Error during first train_test_split (train/temp): {e}. Defaulting to non-stratified split.")
                train_indices, temp_indices = train_test_split(
                    np.arange(len(path_tuples)),
                    test_size=split_ratio, 
                    random_state=data_cfg.get('random_seed', 42),
                    stratify=None
                )

        # Second split: Validation vs. Test from Temp set
        if len(temp_indices) > 1:
            temp_labels_for_stratify = [all_labels_numeric[i] for i in temp_indices]
            num_unique_classes_in_temp = len(set(temp_labels_for_stratify))

            # Calculate actual split sizes for the second split (0.5 for test_size)
            second_split_test_size_abs = int(len(temp_indices) * 0.5)
            if len(temp_indices) > 1 and second_split_test_size_abs == 0 : # Ensure test_size is at least 1 if temp_indices has e.g. 1 element and 0.5*1=0
                 second_split_test_size_abs = 1 if len(temp_indices) > 1 else 0 # if len is 1, test size should be 0, val gets 1.
            second_split_val_size_abs = len(temp_indices) - second_split_test_size_abs
            
            # Correction for len(temp_indices) == 1 where test_size 0.5 would make second_split_test_size_abs = 0
            # We want val to get the sample if only 1 sample in temp.
            if len(temp_indices) == 1:
                second_split_val_size_abs = 1
                second_split_test_size_abs = 0

            stratify_for_second_split = temp_labels_for_stratify
            if is_debug_run and (second_split_test_size_abs < num_unique_classes_in_temp or second_split_val_size_abs < num_unique_classes_in_temp):
                logger.warning(
                    f"Debug mode: Disabling stratification for val/test split. TEST_SIZE ({second_split_test_size_abs}) or VAL_SIZE ({second_split_val_size_abs}) "
                    f"is smaller than N_UNIQUE_CLASSES_IN_TEMP ({num_unique_classes_in_temp})."
                )
                stratify_for_second_split = None
            elif second_split_test_size_abs == 0 or second_split_val_size_abs == 0: # Handles cases where one split would be empty and other has data
                if not (second_split_test_size_abs == 0 and second_split_val_size_abs == 0): # Avoid warning if temp_indices is empty
                    logger.warning(
                        f"Warning: A split part (val or test) would be empty (val_size={second_split_val_size_abs}, test_size={second_split_test_size_abs}). "
                        f"Disabling stratification for val/test split."
                    )
                stratify_for_second_split = None
            
            try:
                val_indices, test_indices = train_test_split(
                    temp_indices,
                    test_size=0.5, # Splitting the temp set into 50% validation, 50% test
                    random_state=data_cfg.get('random_seed', 42),
                    stratify=stratify_for_second_split
                )
            except ValueError as e:
                logger.error(f"Error during second train_test_split (val/test): {e}. Defaulting to non-stratified split.")
                val_indices, test_indices = train_test_split(
                    temp_indices,
                    test_size=0.5, 
                    random_state=data_cfg.get('random_seed', 42),
                    stratify=None
                )
        elif len(temp_indices) == 1:
            logger.warning("Only one sample in temp set. Assigning to validation set, test set will be empty.")
            val_indices = temp_indices
            test_indices = []
        else: # len(temp_indices) == 0
            val_indices, test_indices = [], []

        train_path_tuples = [path_tuples[i] for i in train_indices]
        
        val_path_tuples = [path_tuples[i] for i in val_indices]
        val_labels = [all_labels_numeric_tf[i] for i in val_indices]

        test_path_tuples = [path_tuples[i] for i in test_indices]
        test_labels = [all_labels_numeric_tf[i] for i in test_indices]

        # Create tf.data.Dataset objects
        # For training dataset
        if train_path_tuples:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_path_tuples, [all_labels_numeric_tf[i] for i in train_indices]))
            train_dataset = train_dataset.shuffle(buffer_size=len(train_path_tuples), seed=42, reshuffle_each_iteration=True) # Shuffle before mapping
            augmentation_pipeline = _build_augmentation_pipeline(data_cfg)
            preprocess_fn_rgb = _get_preprocess_fn(model_cfg['architecture'])

            depth_prep_cfg = data_cfg.get('modalities_preprocessing', {}).get('depth_map', {})
            pc_prep_cfg = data_cfg.get('modalities_preprocessing', {}).get('point_cloud', {})

            # Helper function for point cloud processing (to be wrapped by tf.py_function)
            def _load_and_preprocess_point_cloud_py(pc_path_bytes: bytes, num_points_target_tensor: tf.Tensor, normalization_method_tensor: tf.Tensor) -> np.ndarray:
                pc_path = pc_path_bytes.numpy().decode('utf-8')
                num_points_target = num_points_target_tensor.numpy() # Convert num_points_target tensor to Python int
                current_normalization_method = normalization_method_tensor.numpy().decode('utf-8') # Convert normalization_method tensor to Python string
                try:
                    if not pc_path or not pathlib.Path(pc_path).exists():
                        return np.zeros((num_points_target, 3), dtype=np.float32)

                    mesh_or_points = trimesh.load(pc_path, process=False) 
                    
                    if isinstance(mesh_or_points, trimesh.Trimesh):
                        if mesh_or_points.vertices.shape[0] == 0:
                            return np.zeros((num_points_target, 3), dtype=np.float32)
                        points = mesh_or_points.vertices 
                    elif isinstance(mesh_or_points, trimesh.points.PointCloud):
                        points = mesh_or_points.vertices
                    else:
                        return np.zeros((num_points_target, 3), dtype=np.float32)

                    if points.shape[0] == 0:
                        return np.zeros((num_points_target, 3), dtype=np.float32)

                    # Subsample or pad to num_points_target
                    if points.shape[0] > num_points_target:
                        indices = np.random.choice(points.shape[0], num_points_target, replace=False)
                        points = points[indices]
                    elif points.shape[0] < num_points_target:
                        if points.shape[0] == 0: 
                            return np.zeros((num_points_target, 3), dtype=np.float32)
                        indices = np.random.choice(points.shape[0], num_points_target, replace=True)
                        points = points[indices]
                
                    # Normalize coordinates
                    points = points.astype(np.float32) 
                    points_mean = np.mean(points, axis=0)
                    points_centered = points - points_mean

                    if current_normalization_method == 'unit_sphere':
                        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
                        if max_dist < 1e-6: max_dist = 1.0 
                        points_normalized = points_centered / max_dist
                    elif current_normalization_method == 'unit_cube':
                        # Scale to fit within a [-1, 1] cube, maintaining aspect ratio
                        max_abs_coord = np.max(np.abs(points_centered))
                        if max_abs_coord < 1e-6: max_abs_coord = 1.0
                        points_normalized = points_centered / max_abs_coord
                        points_normalized = np.clip(points_normalized, -1.0, 1.0) 
                    elif current_normalization_method == 'centered_only': 
                        points_normalized = points_centered
                    else: # Default or 'none'
                        points_normalized = points 
            
                    return points_normalized.astype(np.float32)

                except Exception as e:
                    return np.zeros((num_points_target, 3), dtype=np.float32)

            # Updated load_and_preprocess signature and logic
            def load_and_preprocess(input_paths: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], label: tf.Tensor, augment: bool = False) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
                rgb_path = input_paths[0]
                depth_path_tensor = input_paths[1]
                pc_path_tensor = input_paths[2]

                # --- RGB Image Processing ---
                image_string = tf.io.read_file(rgb_path)
                rgb_image = tf.image.decode_image(image_string, channels=3, expand_animations=False, dtype=tf.uint8)
                rgb_image = tf.image.resize(rgb_image, image_size)
                rgb_image_float = tf.cast(rgb_image, tf.float32)
                rgb_image_float.set_shape([*image_size, 3])

                if augment and augmentation_pipeline is not None:
                    img_for_aug = tf.expand_dims(rgb_image_float, axis=0)
                    img_aug = augmentation_pipeline(img_for_aug, training=True)
                    rgb_image_float = tf.squeeze(img_aug, axis=0)
                    rgb_image_float = tf.clip_by_value(rgb_image_float, 0.0, 255.0)
        
                rgb_image_preprocessed = preprocess_fn_rgb(rgb_image_float)
                inputs = {'rgb_input': rgb_image_preprocessed}

                # --- Depth Map Processing ---
                if use_depth_map:
                    # Check if depth_path_tensor is not an empty string
                    depth_exists_cond = tf.strings.length(depth_path_tensor) > 0
                    
                    def load_and_process_depth():
                        depth_string = tf.io.read_file(depth_path_tensor)
                        # Try decoding as PNG, then JPEG if PNG fails, common for depth maps
                        try:
                            depth_image_decoded = tf.image.decode_png(depth_string, channels=1, dtype=tf.uint8) # Or tf.uint16 if applicable
                        except tf.errors.InvalidArgumentError:
                            depth_image_decoded = tf.image.decode_jpeg(depth_string, channels=1)
                    
                        depth_image_resized = tf.image.resize(depth_image_decoded, image_size)
                        depth_image_float = tf.cast(depth_image_resized, tf.float32)
                    
                        norm_method = depth_prep_cfg.get('normalization', 'min_max_local')
                        if norm_method == 'min_max_local':
                            min_val = tf.reduce_min(depth_image_float)
                            max_val = tf.reduce_max(depth_image_float)
                            denominator = max_val - min_val
                            depth_normalized = tf.cond(denominator < 1e-6, 
                                                       lambda: tf.zeros_like(depth_image_float), 
                                                       lambda: (depth_image_float - min_val) / denominator)
                        elif norm_method == 'fixed_range':
                            fixed_min = float(depth_prep_cfg.get('fixed_min_val', 0.0))
                            fixed_max = float(depth_prep_cfg.get('fixed_max_val', 255.0)) # Assuming 8-bit like range if not specified
                            denominator = fixed_max - fixed_min
                            if denominator < 1e-6: denominator = 1.0 # Avoid division by zero, treat as no normalization
                            depth_normalized = (depth_image_float - fixed_min) / denominator
                            depth_normalized = tf.clip_by_value(depth_normalized, 0.0, 1.0)
                        else: # Default or 'none'
                            depth_normalized = depth_image_float / 255.0 # Assuming 0-255 range needs scaling to 0-1

                        depth_normalized.set_shape([*image_size, 1])
                        return depth_normalized
                
                    def zeros_for_depth():
                        return tf.zeros([*image_size, 1], dtype=tf.float32)

                    inputs['depth_input'] = tf.cond(depth_exists_cond, load_and_process_depth, zeros_for_depth)

                # --- Point Cloud Processing ---
                if use_point_cloud:
                    num_points_target = int(pc_prep_cfg.get('num_points', 4096))
                    normalization_method_str = pc_prep_cfg.get('normalization', 'unit_sphere')
                
                    pc_exists_cond = tf.strings.length(pc_path_tensor) > 0

                    def load_and_process_pc():
                        pc_data = tf.py_function(_load_and_preprocess_point_cloud_py, 
                                                 inp=[pc_path_tensor, num_points_target, normalization_method_str], 
                                                 Tout=tf.float32)
                        pc_data.set_shape([num_points_target, 3])
                        return pc_data

                    def zeros_for_pc():
                        return tf.zeros([num_points_target, 3], dtype=tf.float32)
                
                    inputs['pc_input'] = tf.cond(pc_exists_cond, load_and_process_pc, zeros_for_pc)
            
                return inputs, label

            AUTOTUNE = tf.data.AUTOTUNE

            train_dataset = train_dataset.map(lambda x, y: load_and_preprocess(x, y, augment=data_cfg.get('augmentation', {}).get('enabled', False)), num_parallel_calls=AUTOTUNE)
            logger.info("Applying .cache() to the training dataset after mapping. This will use more RAM but speed up subsequent epochs.")
            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

        else:
            train_dataset = None
            logger.info("Training dataset is empty after split.")

        # Factory function to create the one-hot encoding mapper
        # This helps ensure num_classes is correctly captured for AutoGraph.
        def one_hot_encode_eval_labels_factory(num_classes_to_use: int):
            @tf.function # Explicitly mark for AutoGraph if needed, or let map handle it
            def _map_fn(features, label):
                return features, tf.one_hot(tf.cast(label, tf.int32), depth=num_classes_to_use)
            return _map_fn

        # Create the mapper function using the final num_classes value
        # This num_classes is determined after potential debug filtering
        # For one-hot encoding depth, always use model_configured_num_classes
        one_hot_mapper_for_eval = one_hot_encode_eval_labels_factory(model_configured_num_classes)

        # For validation dataset
        if val_path_tuples:
            val_dataset = tf.data.Dataset.from_tensor_slices((val_path_tuples, val_labels))
            val_dataset = val_dataset.map(lambda x, y: load_and_preprocess(x, y, augment=False), num_parallel_calls=AUTOTUNE)
            val_dataset = val_dataset.map(one_hot_mapper_for_eval, num_parallel_calls=AUTOTUNE)
            logger.info("Applying .cache() to the validation dataset after mapping and one-hot encoding labels.")
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
        else:
            val_dataset = None
            logger.info("Validation dataset is empty after split.")

        # For test dataset
        if test_path_tuples:
            test_dataset = tf.data.Dataset.from_tensor_slices((test_path_tuples, test_labels))
            test_dataset = test_dataset.map(lambda x, y: load_and_preprocess(x, y, augment=False), num_parallel_calls=AUTOTUNE)
            test_dataset = test_dataset.map(one_hot_mapper_for_eval, num_parallel_calls=AUTOTUNE)
            logger.info("Applying .cache() to the test dataset after mapping and one-hot encoding labels.")
            test_dataset = test_dataset.cache()
            test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
            test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
        else:
            test_dataset = None
            logger.info("Test dataset is empty after split.")

        # Apply MixUp and CutMix after batching
        # These augmentations operate on batches and can change label distributions (e.g., to soft labels).
        # The structure of train_dataset elements (tensor vs dict) depends on whether multi-modal is enabled.
        
        is_multimodal_enabled = data_cfg.get('modalities_config', {}).get('enabled', False)

        mixup_config = data_cfg.get('augmentation', {}).get('mixup', {})
        if train_dataset and mixup_config.get('enabled', False) and mixup_config.get('alpha', 0.0) > 0:
            mixup_alpha = float(mixup_config.get('alpha', 0.2))
            logger.info(f"Applying MixUp augmentation with alpha={mixup_alpha} and num_classes={model_configured_num_classes}.")
            
            def apply_mixup_map(features_input, labels_input):
                rgb_to_process = features_input['rgb_input'] # load_and_preprocess always returns a dict
                mixed_rgb, mixed_labels = mixup(rgb_to_process, labels_input, mixup_alpha, model_configured_num_classes)
                if is_multimodal_enabled:
                    # Update the dict for multi-modal case
                    updated_features = features_input.copy() # Make a copy to modify
                    updated_features['rgb_input'] = mixed_rgb
                    return updated_features, mixed_labels
                else:
                    # Return only the mixed RGB tensor for single-modal case
                    return mixed_rgb, mixed_labels
            train_dataset = train_dataset.map(apply_mixup_map, num_parallel_calls=AUTOTUNE)
        elif mixup_config.get('enabled', False):
            logger.info("MixUp is enabled in config but alpha is 0.0 or not specified correctly. No MixUp will be applied.")

        cutmix_config = data_cfg.get('augmentation', {}).get('cutmix', {})
        if train_dataset and cutmix_config.get('enabled', False) and cutmix_config.get('alpha', 0.0) > 0:
            cutmix_alpha = float(cutmix_config.get('alpha', 1.0))
            logger.info(f"Applying CutMix augmentation with alpha={cutmix_alpha} and num_classes={model_configured_num_classes}.")

            def apply_cutmix_map(features_input, labels_input):
                if is_multimodal_enabled:
                    # features_input is a dict from MixUp (or load_and_preprocess if MixUp was skipped)
                    rgb_to_process = features_input['rgb_input']
                else:
                    # features_input is an RGB tensor from MixUp (or features_input['rgb_input'] from load_and_preprocess if MixUp was skipped)
                    # This path assumes if MixUp was skipped and single-modal, the structure might still be a dict initially.
                    # However, the MixUp stage for single-modal returns a tensor if applied.
                    if isinstance(features_input, dict):
                        rgb_to_process = features_input['rgb_input'] # Handle case where MixUp was skipped
                    else:
                        rgb_to_process = features_input # Is a tensor if MixUp (single-modal) was applied
                
                cutmixed_rgb, cutmixed_labels = cutmix(rgb_to_process, labels_input, cutmix_alpha, model_configured_num_classes)
                
                if is_multimodal_enabled:
                    # Update the dict for multi-modal case
                    # features_input here is already a copy if it came from the multi-modal MixUp path
                    # If MixUp was skipped, make a copy.
                    if not isinstance(features_input, dict):
                         # This case should ideally not be hit if logic is consistent
                         # but as a safeguard if prev stage was single-modal mixup returning tensor.
                         updated_features = {'rgb_input': cutmixed_rgb} # Potential loss of other modalities if not careful
                         logger.warning("CutMix in multi-modal path received tensor, potential loss of other modalities.")
                    else:
                        updated_features = features_input.copy() 
                    updated_features['rgb_input'] = cutmixed_rgb
                    return updated_features, cutmixed_labels
                else:
                    # Return only the cutmixed RGB tensor for single-modal case
                    return cutmixed_rgb, cutmixed_labels
            train_dataset = train_dataset.map(apply_cutmix_map, num_parallel_calls=AUTOTUNE)
        elif cutmix_config.get('enabled', False):
            logger.info("CutMix is enabled in config but alpha is 0.0 or not specified correctly. No CutMix will be applied.")

        if train_dataset:
            train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
            logger.info("Training dataset created and prefetched.")
    
        if val_dataset:
            val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
            logger.info("Validation dataset created and prefetched.")
    
        class_weights_dict = compute_class_weights(all_labels_numeric, model_configured_num_classes)
        if class_weights_dict:
            logger.info(f"Computed class weights for {len(class_weights_dict)} classes based on {model_configured_num_classes} configured classes.")
        else:
            logger.info("Class weights not computed (e.g., empty labels or uniform distribution implied).")

        logger.info("Data loading and preprocessing complete.")
        return train_dataset, val_dataset, test_dataset, model_configured_num_classes, index_to_label, class_weights_dict

    except FileNotFoundError as e:
        logger.error(f"Configuration or data file not found: {e}")
        traceback.print_exc()
        return None, None, None, 0, {}, {}
    except KeyError as e:
        logger.error(f"Missing expected key in configuration: {e}")
        traceback.print_exc()
        return None, None, None, 0, {}, {}
    except ValueError as e:
        logger.error(f"ValueError during data loading/splitting: {e}")
        traceback.print_exc()
        return None, None, None, 0, {}, {}
    except Exception as e:
        logger.error(f"Failed to load classification data: {e}")
        traceback.print_exc() 
        return None, None, None, 0, {}, {}
