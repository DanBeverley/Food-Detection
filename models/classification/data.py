import json
import os
import tensorflow as tf
from typing import Tuple, List, Dict, Optional
import numpy as np
import logging
from sklearn.model_selection import train_test_split # For stratified splitting
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import pathlib
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent

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
        # Convert degrees to fraction of 2*pi
        factor = aug_config['rotation_range'] / 360.0
        pipeline.add(tf.keras.layers.RandomRotation(factor))
    if aug_config.get('zoom_range', 0) > 0:
        pipeline.add(tf.keras.layers.RandomZoom(height_factor=aug_config['zoom_range'], width_factor=aug_config['zoom_range']))
    if 'brightness_range' in aug_config:
        pipeline.add(tf.keras.layers.RandomBrightness(factor=max(abs(1 - aug_config['brightness_range'][0]), abs(aug_config['brightness_range'][1] - 1))))
    if aug_config.get("width_shift_range", 0) > 0:
        pipeline.add(tf.keras.layers.RandomWidth(factor=aug_config['width_shift_range']))
    if aug_config.get("height_shift_range", 0) > 0:
        # Correct argument name is 'factor'
        pipeline.add(tf.keras.layers.RandomHeight(factor=aug_config['height_shift_range']))
    # if aug_config.get("shear_range", 0) > 0:
        # RandomShear does not exist in tf.keras.layers in this version
        # pipeline.add(tf.keras.layers.RandomShear(intensity=aug_config['shear_range']))
    pipeline.add(tf.keras.layers.Resizing(config['image_size'][0], config['image_size'][1]))

    logger.info(f"Built augmentation pipeline with {len(pipeline.layers)-1} layers.")
    return pipeline

def load_classification_data(config: Dict) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[int, str]]:
    """
    Load and prepare classification data from metadata, applying augmentation and splitting.

    Args:
        config: Dictionary containing configuration parameters like metadata_path, 
                data_dir, image_size, batch_size, split_ratio, augmentation settings.

    Returns:
        Tuple of (train_dataset, val_dataset, index_to_label_map).

    Raises:
        FileNotFoundError: If metadata or images are missing.
        ValueError: If invalid configuration.
        json.JSONDecodeError: If metadata file is corrupted.
    """
    logger = logging.getLogger(__name__)
    project_root = _get_project_root()
    
    dataset_root_dir_str = config["dataset_root_dir"]
    label_map_path_str = config["label_map_path"]
    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    split_ratio = config["split_ratio"]
    num_classes_config = config.get("num_classes")

    # Handle relative paths from the project root
    if not os.path.isabs(dataset_root_dir_str):
        dataset_root_dir = project_root/dataset_root_dir_str
    else:
        dataset_root_dir = pathlib.Path(dataset_root_dir_str)
    if not os.path.isabs(label_map_path_str):
        label_map_path = project_root/label_map_path_str
    else:
        label_map_path = pathlib.Path(label_map_path_str)
    if not dataset_root_dir.exists():
        raise FileNotFoundError(f"Dataset root directory not found: {dataset_root_dir}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map file not found: {label_map_path}")

    logger.info(f"Loading label map from: {label_map_path}")
    with open(label_map_path, "r") as f:
        # JSON keys are strigs, convert to int for index
        loaded_label_map = json.load(f)
        index_to_label = {int(k):v for k, v in loaded_label_map.items()}
    label_to_index = {v:k for k, v in index_to_label.items()}
    num_classes = len(index_to_label)
    logger.info(f"Loaded {num_classes} classes from label map")
    if num_classes_config and num_classes_config != num_classes:
        logger.warning(f"Number of classes in label_map ({num_classes}) differs from config ({num_classes_config}). Using label map.")
    
    # Discover image files and instances
    all_instances_data = [] # List of {'path':str, 'label':int, 'instance_id':str}
    logger.info(f"Scanning dataset directory: {dataset_root_dir}")
    for class_names, class_idx in label_to_index.items():
        class_dir = dataset_root_dir / class_name
        if not class_dir.is_dir():
            logger.warning(f"Class directory not found: {class_dir}. Skipping")
            continue
        for instance_dir in class_dir.iterdir():
            if instance_dir.is_dir(): #e.g., Apple_1, Apple_2
                instance_id = f"{class_name}_{instance_dir.name}" # Unique ID for isntance
                original_images_dir = instance_dir / "original"
                if original_images_dir.is_dir():
                    image_files = list(original_images_dir.glob("*.jpg")) + \
                                  list(original_images_dir.glob("*.png"))
                    if not image_files:
                        logger.warning(f"No images found in {original_images_dir} for instance {instance_id}")
                        continue
                    for img_path in image_files:
                        all_instances_data.append({
                               'path': str(img_path),
                               'label': class_idx,
                               'instance_id': instance_id
                           })
    if not all_instances_data:
           raise FileNotFoundError("No image files found in the dataset structure.")
    logger.info(f"Found {len(all_instances_data)} total images across all instances.")

    # Instance-Aware Train/Validation Split 
    unique_instance_ids = sorted(list(set(item['instance_id'] for item in all_instances_data)))
    if not unique_instance_ids:
            raise ValueError("No instances found to perform train/val split.")

    logger.info(f"Performing instance-aware split for {len(unique_instance_ids)} unique instances.")
    
    # Split instance IDs
    train_instance_ids, val_instance_ids = train_test_split(
        unique_instance_ids,
        test_size=split_ratio,
        random_state=42 # for reproducibility
        # Stratification by class at instance level is complex; simple random split of instances for now
    )
    logger.info(f"Train instances: {len(train_instance_ids)}, Validation instances: {len(val_instance_ids)}")

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    for item in all_instances_data:
        if item['instance_id'] in train_instance_ids:
            train_paths.append(item['path'])
            train_labels.append(item['label'])
        elif item['instance_id'] in val_instance_ids:
            val_paths.append(item['path'])
            val_labels.append(item['label'])
    
    if not train_paths:
        raise ValueError("Training set is empty after split. Check dataset or split_ratio.")
    if not val_paths and split_ratio > 0:
            logger.warning("Validation set is empty after split. This might be intended if split_ratio is very small or dataset is tiny.")


    logger.info(f"Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

    # Build Augmentation Pipeline
    augmentation_pipeline = _build_augmentation_pipeline(config)

    # Define Image Loading and Preprocessing Function 
    def load_and_preprocess(path: str, label: int, augment: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        try:
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, image_size) # Resize first
            image.set_shape([*image_size, 3])
            image = tf.cast(image, tf.float32) # Ensure float32 [0-255]

            if augment and augmentation_pipeline is not None:
                # Augmentation pipeline expects batch dim
                img_for_aug = tf.expand_dims(image, axis=0) 
                img_aug = augmentation_pipeline(img_for_aug, training=True)
                image = tf.squeeze(img_aug, axis=0)
                # Clip values to [0, 255] after augmentation if necessary, before preprocess_input
                image = tf.clip_by_value(image, 0.0, 255.0) 

            image = preprocess_input(image) # Apply model-specific preprocessing (e.g., scaling to [-1,1] or [0,1])
            return image, label
        except Exception as e:
            logger.error(f"Error loading/preprocessing image {path}: {e}")
            # Return a placeholder or skip? For now, re-raise.
            # Consider mechanisms for skipping corrupted images if robust pipeline needed.
            raise

    # Create tf.data.Datasets 
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_dataset = train_dataset.map(lambda p, l: load_and_preprocess(p, l, augment=True), num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=max(1000, len(train_paths))) # Adjusted buffer size
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    logger.info("Training dataset created.")

    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_dataset = val_dataset.map(lambda p, l: load_and_preprocess(p, l, augment=False), num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    logger.info("Validation dataset created.")

    return train_dataset, val_dataset, index_to_label

def load_test_data(config: Dict, index_to_label_map: Optional[Dict[int, str]] = None) -> Tuple[tf.data.Dataset, Dict[int, str]]:
    """
    Load data for evaluation/inference. Requires a consistent label mapping.

    Args:
        config: Dictionary containing configuration parameters.
        index_to_label_map: Optional. If provided, use this mapping. 
                           If None, it will be recalculated (NOT RECOMMENDED for consistency).

    Returns:
        Tuple of (test_dataset, index_to_label_map used).

    Raises:
        FileNotFoundError, ValueError, json.JSONDecodeError.
    """
    metadata_path = config['metadata_path']
    data_dir = config['data_dir']
    image_size = tuple(config['image_size'])
    batch_size = config['batch_size']

    if not os.path.isabs(metadata_path):
         metadata_path = os.path.join(os.path.dirname(__file__), '..', '..', metadata_path)
    if not os.path.isabs(data_dir):
         data_dir = os.path.join(os.path.dirname(__file__), '..', '..', data_dir)

    try:
        logger.info(f"Loading test metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded test metadata for {len(metadata)} items.")

        image_paths = []
        string_labels = [] # Ground truth labels for evaluation
        for item in metadata:
            full_path = os.path.join(data_dir, item['image_path'])
            if not os.path.exists(full_path):
                 logger.warning(f"Image file not found: {full_path}. Skipping.")
                 continue
            image_paths.append(full_path)
            string_labels.append(item['label'])

        if not image_paths:
            raise FileNotFoundError("No valid image paths found for test data.")

        if index_to_label_map is None:
            logger.warning("Label map not provided for test data. Recalculating. "
                           "For consistent evaluation/inference, always use the map from training.")
            unique_labels = sorted(list(set(string_labels)))
            label_to_index = {label: index for index, label in enumerate(unique_labels)}
            index_to_label_map = {index: label for label, index in label_to_index.items()}
        else:
            logger.info("Using provided label map for test data.")
            label_to_index = {label: index for index, label in index_to_label_map.items()}

        integer_labels = [label_to_index.get(label, -1) for label in string_labels] # Use -1 for unknown test labels
        if any(lbl == -1 for lbl in integer_labels):
             logger.warning("Some test labels were not found in the provided/recalculated label map.")

        def load_and_preprocess_test(path: str, label: int) -> Tuple[tf.Tensor, tf.Tensor]:
             # Simplified version without augmentation
            try:
                image = tf.io.read_file(path)
                image = tf.image.decode_image(image, channels=3, expand_animations=False)
                image = tf.image.resize(image, image_size)
                image.set_shape([*image_size, 3])
                image = image / 255.0
                return image, label
            except Exception as e:
                 logger.error(f"Error processing image {path}: {e}")
                 raise

        AUTOTUNE = tf.data.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, integer_labels))
        dataset = dataset.map(load_and_preprocess_test, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        logger.info("Test dataset created.")

        return dataset, index_to_label_map

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Test data loading failed: {e}")
        raise