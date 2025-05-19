import json
import os
import tensorflow as tf
from typing import Tuple, List, Dict, Optional
# import numpy as np # Not strictly used in the refactored load_classification_data
import logging
from sklearn.model_selection import train_test_split
import pathlib
# import glob # No longer needed
from tqdm import tqdm # Import tqdm
import trimesh # For point cloud processing

# Dynamic import for preprocess_input
_PREPROCESS_FN_CACHE = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent.parent

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

def load_classification_data(config: Dict) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], int, Dict[int, str]]:
    """
    Load and prepare classification data using a metadata.json file.
    Refactored to read from metadata_path, use label_map_path from paths config.
    Args:
        config: Dictionary from classification/config.yaml.
    Returns:
        Tuple of (train_dataset, val_dataset, num_classes, index_to_label_map).
    """
    project_root = _get_project_root()

    try:
        data_cfg = config['data']
        paths_cfg = config['paths']
        model_cfg = config['model']
        architecture = model_cfg['architecture']
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

        # Construct metadata_path and label_map_path using paths_cfg
        # Ensure paths are absolute if they are relative to project root or a specific data_dir
        project_root = _get_project_root()
        base_data_dir = project_root / paths_cfg.get('data_dir', 'data/classification')

        metadata_filename = paths_cfg.get('metadata_filename', 'metadata.json')
        metadata_path = base_data_dir / metadata_filename

        label_map_filename = paths_cfg.get('label_map_filename', 'label_map.json')
        label_map_path = base_data_dir / label_map_filename # Corrected to use base_data_dir

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not label_map_path.exists():
            raise FileNotFoundError(f"Label map file not found: {label_map_path}")

        with open(metadata_path, 'r') as f:
            all_items = json.load(f)
        with open(label_map_path, 'r') as f:
            label_to_index = json.load(f)
    
        index_to_label = {v: k for k, v in label_to_index.items()}
        num_classes = len(label_to_index)

        logger.info(f"Found {len(all_items)} total image items in metadata.")
        logger.info(f"Number of classes: {num_classes} from {label_map_path}.")

        # Prepare lists for all modalities
        all_rgb_paths = []
        all_depth_paths = [] # Store paths or empty strings
        all_pc_paths = []    # Store paths or empty strings
        all_labels_numeric = []
        all_instance_ids = []

        for item in tqdm(all_items, desc="Processing metadata items"):
            rgb_path_str = item['path']
            # Basic validation: ensure rgb_path_str is not empty and exists
            if not rgb_path_str or not pathlib.Path(rgb_path_str).exists():
                logger.warning(f"RGB path missing or file not found: '{rgb_path_str}' for item {item.get('instance_id', 'N/A')}. Skipping item.")
                continue
            all_rgb_paths.append(rgb_path_str)
            all_labels_numeric.append(label_to_index[item['label']])
            instance_id = item['instance_id']
            all_instance_ids.append(instance_id)

            depth_path_str = "" # Default to empty string
            if use_depth_map:
                rgb_path_obj = pathlib.Path(rgb_path_str)
                # Depth path construction logic (assuming it's correct as per previous steps)
                potential_depth_path = rgb_path_obj.parent.parent / depth_map_dir_name / rgb_path_obj.name
                if potential_depth_path.exists():
                    depth_path_str = str(potential_depth_path)
                else:
                    logger.debug(f"Depth map not found for {rgb_path_str} at {potential_depth_path}. Will use empty path.")
            all_depth_paths.append(depth_path_str)

            pc_path_str = "" # Default to empty string
            if use_point_cloud:
                food_class_from_label = item['label'] 
                # PC path construction logic (assuming it's correct as per previous steps)
                potential_pc_path = pathlib.Path(point_cloud_root_dir) / pc_sampling_rate_dir / food_class_from_label / instance_id / (instance_id + pc_suffix)
                if potential_pc_path.exists():
                    pc_path_str = str(potential_pc_path)
                else:
                    logger.debug(f"Point cloud not found for instance {instance_id} (class {food_class_from_label}) at {potential_pc_path}. Will use empty path.")
            all_pc_paths.append(pc_path_str)

        if not all_rgb_paths:
            raise ValueError("No image paths were loaded. Check metadata file and paths.")

        # Stratified split if split_ratio > 0 and < 1
        if 0 < split_ratio < 1:
            # Create a unique ID for each item for consistent splitting if metadata doesn't have one
            # Using instance_id for splitting to keep frames from same instance together if possible,
            # but stratification is on labels.
            # For proper instance-level split, group by instance_id first.
            # Current split is per-image, stratified by label.

            # Generate indices for splitting
            indices = list(range(len(all_rgb_paths)))
        
            # Stratified split based on labels
            # Ensure all_labels_numeric is suitable for stratify argument (e.g., list or array-like)
            train_indices, val_indices = train_test_split(
                indices, 
                test_size=split_ratio, 
                random_state=42, # for reproducibility
                stratify=all_labels_numeric
            )

            train_rgb_paths = [all_rgb_paths[i] for i in train_indices]
            train_depth_paths = [all_depth_paths[i] for i in train_indices]
            train_pc_paths = [all_pc_paths[i] for i in train_indices]
            train_labels = [all_labels_numeric[i] for i in train_indices]

            val_rgb_paths = [all_rgb_paths[i] for i in val_indices]
            val_depth_paths = [all_depth_paths[i] for i in val_indices]
            val_pc_paths = [all_pc_paths[i] for i in val_indices]
            val_labels = [all_labels_numeric[i] for i in val_indices]
        
            logger.info(f"Training samples: {len(train_rgb_paths)}, Validation samples: {len(val_rgb_paths)}")
            val_dataset = None # Initialize, will be created if val_rgb_paths is not empty

        elif split_ratio == 0: # Use all data for training, no validation set
            logger.info("Split ratio is 0. Using all data for training, validation set will be None.")
            train_rgb_paths = all_rgb_paths
            train_depth_paths = all_depth_paths
            train_pc_paths = all_pc_paths
            train_labels = all_labels_numeric
            val_rgb_paths, val_depth_paths, val_pc_paths, val_labels = [], [], [], []
            val_dataset = None
        elif split_ratio == 1: # Use all data for validation, no training set (uncommon)
            logger.info("Split ratio is 1. Using all data for validation, training set will be None.")
            val_rgb_paths = all_rgb_paths
            val_depth_paths = all_depth_paths
            val_pc_paths = all_pc_paths
            val_labels = all_labels_numeric
            train_rgb_paths, train_depth_paths, train_pc_paths, train_labels = [], [], [], []
            train_dataset = None # Should not proceed to train if no training data
            if not train_rgb_paths:
                logger.warning("Training dataset is empty because split_ratio is 1.")
        else:
            raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")


        augmentation_pipeline = _build_augmentation_pipeline(data_cfg)
        preprocess_fn_rgb = _get_preprocess_fn(architecture)

        depth_prep_cfg = data_cfg.get('modalities_preprocessing', {}).get('depth_map', {})
        pc_prep_cfg = data_cfg.get('modalities_preprocessing', {}).get('point_cloud', {})

        # Helper function for point cloud processing (to be wrapped by tf.py_function)
        def _load_and_preprocess_point_cloud_py(pc_path_bytes: bytes, num_points_target: int, normalization_method: str) -> np.ndarray:
            pc_path = pc_path_bytes.decode('utf-8')
            try:
                if not pc_path or not pathlib.Path(pc_path).exists():
                    # logger.debug(f"Point cloud file not found or path empty: {pc_path}. Returning zeros.") # Called too often from tf.py_function
                    return np.zeros((num_points_target, 3), dtype=np.float32)

                # Load point cloud using trimesh
                # For .ply, trimesh.load usually returns a Trimesh object or PointCloud object
                mesh_or_points = trimesh.load(pc_path, process=False) # process=False to avoid early processing
                
                if isinstance(mesh_or_points, trimesh.Trimesh):
                    # If it's a mesh, sample points from its surface, or use vertices if sampling fails/not preferred
                    if mesh_or_points.vertices.shape[0] == 0:
                        # logger.warning(f"Mesh {pc_path} has no vertices. Returning zeros.")
                        return np.zeros((num_points_target, 3), dtype=np.float32)
                    # Option 1: Use vertices directly if number is reasonable or sampling is not desired
                    # points = mesh_or_points.vertices
                    # Option 2: Sample points from the surface (more uniform for complex meshes)
                    # points, _ = trimesh.sample.sample_surface(mesh_or_points, num_points_target * 2) # Sample more initially
                    # if points.shape[0] == 0:
                    points = mesh_or_points.vertices # Fallback to vertices if sampling returns nothing
                elif isinstance(mesh_or_points, trimesh.points.PointCloud):
                    points = mesh_or_points.vertices
                else:
                    # logger.warning(f"Unsupported geometry type from {pc_path}: {type(mesh_or_points)}. Returning zeros.")
                    return np.zeros((num_points_target, 3), dtype=np.float32)

                if points.shape[0] == 0:
                    # logger.debug(f"No points found in {pc_path}. Returning zeros.")
                    return np.zeros((num_points_target, 3), dtype=np.float32)

                # Subsample or pad to num_points_target
                if points.shape[0] > num_points_target:
                    indices = np.random.choice(points.shape[0], num_points_target, replace=False)
                    points = points[indices]
                elif points.shape[0] < num_points_target:
                    if points.shape[0] == 0: # Should be caught above, but defensive
                        return np.zeros((num_points_target, 3), dtype=np.float32)
                    indices = np.random.choice(points.shape[0], num_points_target, replace=True)
                    points = points[indices]
            
                # Normalize coordinates
                points = points.astype(np.float32) # Ensure float32 for calculations
                points_mean = np.mean(points, axis=0)
                points_centered = points - points_mean

                if normalization_method == 'unit_sphere':
                    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
                    if max_dist < 1e-6: max_dist = 1.0 # Avoid division by zero
                    points_normalized = points_centered / max_dist
                elif normalization_method == 'unit_cube':
                    # Scale to fit within a [-1, 1] cube, maintaining aspect ratio
                    max_abs_coord = np.max(np.abs(points_centered))
                    if max_abs_coord < 1e-6: max_abs_coord = 1.0 # Avoid division by zero
                    points_normalized = points_centered / max_abs_coord
                    points_normalized = np.clip(points_normalized, -1.0, 1.0) # Ensure strictly within cube
                elif normalization_method == 'centered_only': # Only center, no scaling
                    points_normalized = points_centered
                else: # 'none' or unknown
                    points_normalized = points # Use original points (potentially after centering if desired for 'none')
            
                return points_normalized.astype(np.float32)

            except Exception as e:
                # logger.error(f"Error processing point cloud {pc_path}: {e}. Returning zeros.") # Called too often
                # This print helps during debugging when logger isn't configured for tf.py_function context
                # print(f"CASCADE_DEBUG: Error in _load_and_preprocess_point_cloud_py for {pc_path}: {e}")
                return np.zeros((num_points_target, 3), dtype=np.float32)

        # Updated load_and_preprocess signature and logic
        def load_and_preprocess(input_paths: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], label: tf.Tensor, augment: bool = False) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
            rgb_path, depth_path_tensor, pc_path_tensor = input_paths

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

        if not train_rgb_paths:
            logger.warning("Training dataset is empty. Cannot proceed with training.")
            # Return empty/None datasets or raise error, depending on desired behavior
            # For now, will lead to error later if train_dataset is used when None.
            train_dataset = None
        else:
            train_input_slices = (tf.constant(train_rgb_paths, dtype=tf.string),
                                  tf.constant(train_depth_paths, dtype=tf.string),
                                  tf.constant(train_pc_paths, dtype=tf.string))
        
            train_dataset = tf.data.Dataset.from_tensor_slices((train_input_slices, tf.constant(train_labels, dtype=tf.int32)))
            train_dataset = train_dataset.map(lambda paths_tuple, lbl: load_and_preprocess(paths_tuple, lbl, augment=data_cfg.get('augmentation', {}).get('enabled', False)), num_parallel_calls=AUTOTUNE)
            logger.info("Applying .cache() to the training dataset after mapping. This will use more RAM but speed up subsequent epochs.")
            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(buffer_size=max(1000, len(train_rgb_paths)))
            train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

        if not val_rgb_paths:
            logger.info("Validation dataset is empty or not created (split_ratio might be 0 or 1).")
            val_dataset = None
        else:
            val_input_slices = (tf.constant(val_rgb_paths, dtype=tf.string),
                                tf.constant(val_depth_paths, dtype=tf.string),
                                tf.constant(val_pc_paths, dtype=tf.string))
            val_dataset = tf.data.Dataset.from_tensor_slices((val_input_slices, tf.constant(val_labels, dtype=tf.int32)))
            val_dataset = val_dataset.map(lambda paths_tuple, lbl: load_and_preprocess(paths_tuple, lbl, augment=False), num_parallel_calls=AUTOTUNE)
            logger.info("Applying .cache() to the validation dataset after mapping.")
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.batch(batch_size, drop_remainder=False) # No drop_remainder for validation

        # Apply MixUp if enabled (after batching)
        # Note: MixUp and CutMix currently operate on a single image input.
        # They will need to be adapted if they should affect depth/PC data or if model expects dict input.
        # For now, they will apply to 'rgb_input' if the dataset yields dictionaries from load_and_preprocess.
        # This part might need significant rework depending on how MixUp/CutMix should interact with multiple modalities.
        mixup_config = data_cfg.get('augmentation', {}).get('mixup', {})
        if train_dataset and mixup_config.get('enabled', False):
            mixup_alpha = float(mixup_config.get('alpha', 0.2))
            if mixup_alpha > 0.0:
                logger.info(f"Applying MixUp augmentation to training data with alpha={mixup_alpha}.")
                if not label_to_index: 
                    raise ValueError("Label map is required for MixUp to determine num_classes.")
            
                # Adapting MixUp for dictionary inputs
                def apply_mixup_to_dict(inputs_dict, labels_sparse):
                    rgb_images = inputs_dict['rgb_input']
                    mixed_rgb_images, mixed_labels_one_hot = mixup(rgb_images, labels_sparse, alpha=mixup_alpha, num_classes=num_classes)
                    # inputs_dict['rgb_input'] = mixed_rgb_images # This would modify the dict in place if not careful
                    # Create a new dict for output to avoid issues with tensor immutability / graph mode
                    updated_inputs = inputs_dict.copy() # Shallow copy is usually fine for this structure
                    updated_inputs['rgb_input'] = mixed_rgb_images
                    return updated_inputs, mixed_labels_one_hot
            
                train_dataset = train_dataset.map(apply_mixup_to_dict, num_parallel_calls=AUTOTUNE)
            else:
                logger.info("MixUp is enabled in config but alpha is 0.0. No MixUp will be applied.")

        # Apply CutMix if enabled (after batching, and after MixUp if MixUp was applied)
        cutmix_config = data_cfg.get('augmentation', {}).get('cutmix', {})
        if train_dataset and cutmix_config.get('enabled', False):
            cutmix_alpha = float(cutmix_config.get('alpha', 1.0))
            if cutmix_alpha > 0.0:
                logger.info(f"Applying CutMix augmentation to training data with alpha={cutmix_alpha}.")
                if not label_to_index:
                    raise ValueError("Label map is required for CutMix to determine num_classes.")

                # Adapting CutMix for dictionary inputs
                def apply_cutmix_to_dict(inputs_dict, labels_mixed_or_sparse):
                    rgb_images = inputs_dict['rgb_input']
                    # If mixup was applied, labels_mixed_or_sparse are one-hot. Otherwise, they are sparse.
                    # CutMix internal logic handles this via `if len(tf.shape(batch_labels)) == 1:`
                    cutmixed_rgb_images, cutmixed_labels = cutmix(rgb_images, labels_mixed_or_sparse, alpha=cutmix_alpha, num_classes=num_classes)
                    updated_inputs = inputs_dict.copy()
                    updated_inputs['rgb_input'] = cutmixed_rgb_images
                    return updated_inputs, cutmixed_labels
            
                train_dataset = train_dataset.map(apply_cutmix_to_dict, num_parallel_calls=AUTOTUNE)
            else:
                logger.info("CutMix is enabled in config but alpha is 0.0. No CutMix will be applied.")

        # Prefetch for performance
        if train_dataset:
            train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
            logger.info("Training dataset created and prefetched.")
    
        if val_dataset:
            val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
            logger.info("Validation dataset created and prefetched.")
    
        return train_dataset, val_dataset, num_classes, index_to_label

# --- load_test_data and other functions will be refactored in subsequent steps --- 