import json
import os
import tensorflow as tf
from typing import Tuple, List, Dict, Optional
# import numpy as np # Not strictly used in the refactored load_classification_data
import logging
from sklearn.model_selection import train_test_split
import pathlib
# import glob # No longer needed

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

    if architecture.startswith("EfficientNetV2"):
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    elif architecture.startswith("EfficientNet"):
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif architecture.startswith("ResNet") or architecture.startswith("ConvNeXt") or architecture.startswith("MobileNet") :
        # Assuming a common pattern for these, e.g., tf.keras.applications.resnet.preprocess_input
        try:
            module_name = architecture.lower().split('v')[0] # e.g. resnet, mobilenet
            if module_name == "convnext": # tf.keras.applications.convnext.preprocess_input
                module = __import__(f"tensorflow.keras.applications.{module_name}", fromlist=['preprocess_input'])
            else: # e.g. tf.keras.applications.resnet50.preprocess_input
                module = __import__(f"tensorflow.keras.applications.{architecture.lower()}", fromlist=['preprocess_input'])
            preprocess_input = module.preprocess_input
        except ImportError:
            logger.warning(f"Could not dynamically import preprocess_input for {architecture}. Using generic rescaling.")
            preprocess_input = lambda x: x / 255.0 # Fallback
    else:
        logger.warning(f"Unknown architecture {architecture} for preprocess_input. Using generic rescaling.")
        preprocess_input = lambda x: x / 255.0 # Fallback
    
    _PREPROCESS_FN_CACHE[architecture] = preprocess_input
    return preprocess_input

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
        pipeline.add(tf.keras.layers.RandomShear(intensity=aug_config['shear_range']))
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

    logger.info(f"Built augmentation pipeline with {len(pipeline.layers)-1} augmentation layers.")
    return pipeline

def load_classification_data(config: Dict) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Dict[int, str]]:
    """
    Load and prepare classification data using a metadata.json file.
    Refactored to read from metadata_path, use label_map_path from paths config.
    Args:
        config: Dictionary from classification/config.yaml.
    Returns:
        Tuple of (train_dataset, val_dataset, index_to_label_map).
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

    for idx, item in enumerate(metadata_list):
        class_name = item.get('class_name')
        img_path = item.get('image_path')
        instance_name = item.get('instance_name')

        if idx < 3:
            logger.info(f"DEBUG: Processing item {idx}: class_name='{class_name}', img_path='{img_path}', instance_name='{instance_name}'")

        if not (class_name and img_path and instance_name):
            logger.warning(f"Skipping incomplete metadata item: {item}")
            continue
        
        class_in_map = class_name in label_to_index
        if idx < 3:
            logger.info(f"DEBUG: Item {idx}: class_name='{class_name}' in label_to_index? {class_in_map}")

        if not class_in_map:
            logger.warning(f"Class '{class_name}' in metadata (img: {img_path}) not in label_map. Skipping.")
            continue
        
        img_file_exists = pathlib.Path(img_path).is_file()
        if idx < 3:
             logger.info(f"DEBUG: Item {idx}: img_path='{img_path}' exists? {img_file_exists}")
        
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

    if not train_paths:
        raise ValueError("Training set is empty after split. Check dataset or split_ratio.")
    logger.info(f"Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

    augmentation_pipeline = _build_augmentation_pipeline(data_cfg) # Pass data_cfg for augmentation settings
    preprocess_fn = _get_preprocess_fn(architecture)

    def load_and_preprocess(path: str, label: int, augment: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        image_string = tf.io.read_file(path)
        image = tf.image.decode_image(image_string, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image.set_shape([*image_size, 3]) # Ensure shape consistency
        # Augmentation happens BEFORE a Keras model's preprocess_input usually
        if augment and augmentation_pipeline is not None:
            # Augmentation pipeline expects batch dim, then removes it
            img_for_aug = tf.expand_dims(tf.cast(image, tf.float32), axis=0)
            img_aug = augmentation_pipeline(img_for_aug, training=True)
            image = tf.squeeze(img_aug, axis=0)
            image = tf.clip_by_value(image, 0.0, 255.0) # Ensure valid pixel range post-aug
        
        image_preprocessed = preprocess_fn(tf.cast(image, tf.float32))
        return image_preprocessed, label

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    # Apply specific per-image augmentations here if not handled by the sequential pipeline
    train_dataset = train_dataset.map(lambda p, l: load_and_preprocess(p, l, augment=data_cfg.get('augmentation', {}).get('enabled', False)), num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=max(1000, len(train_paths)))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    logger.info("Training dataset created.")

    val_dataset = None
    if val_paths:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        val_dataset = val_dataset.map(lambda p, l: load_and_preprocess(p, l, augment=False), num_parallel_calls=AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size)
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

    return train_dataset, val_dataset, index_to_label

# --- load_test_data and other functions will be refactored in subsequent steps --- 