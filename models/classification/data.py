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
        metadata_path_str = data_cfg['metadata_path']
        image_size = tuple(data_cfg['image_size'])
        batch_size = data_cfg['batch_size']
        split_ratio = data_cfg['split_ratio']
        random_seed = data_cfg.get('random_seed', 42)
        label_map_filename = paths_cfg['label_map_filename']
        # Ensure we use the explicitly defined label_map_dir from config
        label_map_dir_str = paths_cfg['label_map_dir'] 
        architecture = model_cfg['architecture']
    except KeyError as e:
        raise ValueError(f"Configuration error: missing key {e} in classification config.")

    metadata_file = project_root / metadata_path_str
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Metadata JSON file not found: {metadata_file}")

    label_map_file = project_root / label_map_dir_str / label_map_filename
    logger.info(f"DEBUG: Attempting to load label map from resolved path: {str(label_map_file)}")
    if not label_map_file.is_file():
        logger.error(f"DEBUG: Label map file components: project_root='{project_root}', label_map_dir_str='{label_map_dir_str}', label_map_filename='{label_map_filename}'")
        raise FileNotFoundError(f"Label map file not found: {label_map_file}. Run prepare_classification_dataset.py first.")

    logger.info(f"Loading label map from: {label_map_file}")
    with open(label_map_file, 'r') as f:
        loaded_label_map = json.load(f)
    index_to_label = {int(k): str(v) for k, v in loaded_label_map.items()}
    label_to_index = {v: k for k, v in index_to_label.items()}
    num_classes = len(index_to_label)
    logger.info(f"Loaded {num_classes} classes.")
    logger.info(f"DEBUG: First 3 entries in index_to_label: {dict(list(index_to_label.items())[:3])}")
    logger.info(f"DEBUG: First 3 entries in label_to_index: {dict(list(label_to_index.items())[:3])}")

    if num_classes == 0:
        raise ValueError("Label map is empty.")

    logger.info(f"Loading data from metadata: {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata_list = json.load(f)
    
    all_items = []
    logger.info(f"DEBUG: Processing metadata_list. Total items: {len(metadata_list)}. First 3 items raw:")
    for i, item_raw in enumerate(metadata_list[:3]):
        logger.info(f"DEBUG: Raw metadata item {i}: {item_raw}")

    logger.info("Processing metadata entries from metadata.json...")
    for item in tqdm(metadata_list, desc="Validating image paths from metadata.json"):
        class_name = item.get('class_name')
        img_path = item.get('image_path')
        instance_name = item.get('instance_name')

        if not (class_name and img_path and instance_name):
            logger.warning(f"Skipping incomplete metadata item: {item}")
            continue
        
        class_in_map = class_name in label_to_index
        if not class_in_map:
            logger.warning(f"Class '{class_name}' in metadata (img: {img_path}) not in label_map. Skipping.")
            continue
        
        img_file_exists = pathlib.Path(img_path).is_file()
        if not img_file_exists:
             logger.warning(f"Image file not found: {img_path} (from metadata). Skipping.")
             continue
        all_items.append({
            'path': str(img_path),
            'label': label_to_index[class_name],
            'instance_id': f"{class_name}_{instance_name}"
        })

    if not all_items:
        raise ValueError(f"No valid image data loaded from metadata file: {metadata_file}. Check content and label_map.")
    logger.info(f"Found {len(all_items)} total valid image entries from metadata.")

    train_paths, val_paths = [], []
    train_labels, val_labels = [], []

    if split_ratio > 0.0:
        unique_instance_ids = sorted(list(set(d['instance_id'] for d in all_items)))
        if len(unique_instance_ids) < 2 and len(all_items) > 1:
             logger.warning(f"Only {len(unique_instance_ids)} unique instance(s) found for {len(all_items)} items. Instance-aware split might behave like random split.")
             # Fallback to stratified split on labels if too few instances for meaningful instance split
             all_paths = [d['path'] for d in all_items]
             all_labels = [d['label'] for d in all_items]
             if len(set(all_labels)) > 1: # Stratify only if multiple classes exist
                 train_paths, val_paths, train_labels, val_labels = train_test_split(
                    all_paths, all_labels, test_size=split_ratio, random_state=random_seed, stratify=all_labels)
             else: # Single class, simple split is fine
                 train_paths, val_paths, train_labels, val_labels = train_test_split(
                    all_paths, all_labels, test_size=split_ratio, random_state=random_seed)
        elif len(unique_instance_ids) < 2:
            logger.warning("Not enough unique instances for instance-aware split. Using all data for training, validation will be empty.")
            train_paths = [d['path'] for d in all_items]
            train_labels = [d['label'] for d in all_items]
        else:
            train_instance_ids, val_instance_ids = train_test_split(
                unique_instance_ids, test_size=split_ratio, random_state=random_seed)
            logger.info(f"Instance split: {len(train_instance_ids)} train, {len(val_instance_ids)} val instances.")
            for item in all_items:
                if item['instance_id'] in train_instance_ids:
                    train_paths.append(item['path'])
                    train_labels.append(item['label'])
                elif item['instance_id'] in val_instance_ids:
                    val_paths.append(item['path'])
                    val_labels.append(item['label'])
    else: # split_ratio is 0 or not specified for validation
        logger.info("Split ratio is 0. Using all data for training, validation set will be None.")
        train_paths = [d['path'] for d in all_items]
        train_labels = [d['label'] for d in all_items]
        val_paths = [] # Initialize as empty
        val_labels = [] # Initialize as empty
        val_dataset = None # Explicitly set to None

    augmentation_pipeline = _build_augmentation_pipeline(data_cfg) # Pass data_cfg for augmentation settings
    preprocess_fn = _get_preprocess_fn(architecture)

    def load_and_preprocess(path: str, label: int, augment: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        image_string = tf.io.read_file(path)
        image = tf.image.decode_image(image_string, channels=3, expand_animations=False, dtype=tf.uint8) # Explicitly decode to uint8 first
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) # Cast to float32 early
        image.set_shape([*image_size, 3]) # Ensure shape consistency for the float32 image

        # Augmentation happens BEFORE a Keras model's preprocess_input usually
        if augment and augmentation_pipeline is not None: 
            # Augmentation pipeline expects batch dim, then removes it
            # Image is already float32 here
            img_for_aug = tf.expand_dims(image, axis=0) 
            img_aug = augmentation_pipeline(img_for_aug, training=True)
            image = tf.squeeze(img_aug, axis=0)
            image = tf.clip_by_value(image, 0.0, 255.0)
        
        # Image is already float32
        image_preprocessed = preprocess_fn(image)
        return image_preprocessed, label

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    # Apply specific per-image augmentations here if not handled by the sequential pipeline
    train_dataset = train_dataset.map(lambda p, l: load_and_preprocess(p, l, augment=data_cfg.get('augmentation', {}).get('enabled', False)), num_parallel_calls=AUTOTUNE)
    logger.info("Applying .cache() to the training dataset after mapping. This will use more RAM but speed up subsequent epochs.")
    train_dataset = train_dataset.cache() # Cache after mapping
    train_dataset = train_dataset.shuffle(buffer_size=max(1000, len(train_paths)))
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True) 

    # Create and process val_dataset only if val_paths is not empty
    if val_paths:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        val_dataset = val_dataset.map(lambda p, l: load_and_preprocess(p, l, augment=False), num_parallel_calls=AUTOTUNE) # No augmentation for validation
        logger.info("Applying .cache() to the validation dataset after mapping.")
        val_dataset = val_dataset.cache() # Cache after mapping
        val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
    # If val_paths was empty, val_dataset remains None as set above

    # Apply MixUp if enabled (after batching)
    mixup_config = data_cfg.get('augmentation', {}).get('mixup', {})
    if mixup_config.get('enabled', False):
        mixup_alpha = float(mixup_config.get('alpha', 0.2))
        if mixup_alpha > 0.0:
            logger.info(f"Applying MixUp augmentation to training data with alpha={mixup_alpha}.")
            # Need num_classes for one-hot encoding in mixup function
            if not index_to_label: # Should have been loaded earlier
                raise ValueError("Label map is required for MixUp to determine num_classes.")
            num_classes = len(index_to_label)
            train_dataset = train_dataset.map(lambda x, y: mixup(x, y, alpha=mixup_alpha, num_classes=num_classes), num_parallel_calls=AUTOTUNE)
            # Optionally apply to validation set if it exists and if desired (usually not for validation)
            # if val_dataset is not None and mixup_config.get('apply_to_validation', False):
            #     logger.info(f"Applying MixUp augmentation to validation data with alpha={mixup_alpha}.")
            #     val_dataset = val_dataset.map(lambda x, y: mixup(x, y, alpha=mixup_alpha, num_classes=num_classes), num_parallel_calls=AUTOTUNE)
        else:
            logger.info("MixUp is enabled in config but alpha is 0.0. No MixUp will be applied.")

    # Apply CutMix if enabled (after batching, and after MixUp if MixUp was applied)
    cutmix_config = data_cfg.get('augmentation', {}).get('cutmix', {})
    if cutmix_config.get('enabled', False):
        cutmix_alpha = float(cutmix_config.get('alpha', 1.0))
        if cutmix_alpha > 0.0:
            logger.info(f"Applying CutMix augmentation to training data with alpha={cutmix_alpha}.")
            if not index_to_label:
                raise ValueError("Label map is required for CutMix to determine num_classes.")
            num_classes = len(index_to_label)
            train_dataset = train_dataset.map(lambda x, y: cutmix(x, y, alpha=cutmix_alpha, num_classes=num_classes), num_parallel_calls=AUTOTUNE)
            # Optionally apply to validation set if it exists and if desired (usually not for validation)
            # if val_dataset is not None and cutmix_config.get('apply_to_validation', False):
            #     logger.info(f"Applying CutMix augmentation to validation data with alpha={cutmix_alpha}.")
            #     val_dataset = val_dataset.map(lambda x, y: cutmix(x, y, alpha=cutmix_alpha, num_classes=num_classes), num_parallel_calls=AUTOTUNE)
        else:
            logger.info("CutMix is enabled in config but alpha is 0.0. No CutMix will be applied.")

    # Prefetch for performance
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    if val_dataset is not None:
        val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
        logger.info("Validation dataset created.")
    else:
        logger.info("Validation dataset is None as there are no validation samples or split_ratio was 0.")

    # Developer mode: limit dataset size if specified (example, adapt as needed from original)
    # if config.get('dev_mode', {}).get('enabled', False):
    #     max_train_samples = config['dev_mode'].get('max_train_samples', None)
    #     max_val_samples = config['dev_mode'].get('max_val_samples', None)
    #     if max_train_samples:
    #         train_dataset = train_dataset.take(max_train_samples // batch_size + 1)
    #         logger.info(f"Dev mode: Training dataset limited to approx {max_train_samples} samples.")
    #     if val_dataset and max_val_samples:
    #         val_dataset = val_dataset.take(max_val_samples // batch_size + 1)
    #         logger.info(f"Dev mode: Validation dataset limited to approx {max_val_samples} samples.")

    return train_dataset, val_dataset, num_classes, index_to_label

# --- load_test_data and other functions will be refactored in subsequent steps --- 