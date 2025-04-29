import json
import os
import tensorflow as tf
from typing import Tuple, List, Dict, Optional
import numpy as np
import logging
from sklearn.model_selection import train_test_split # For stratified splitting

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        pipeline.add(tf.keras.layers.RandomWidth(width_shift_range=aug_config['width_shift_range']))
    if aug_config.get("height_shift_range", 0) > 0:
        pipeline.add(tf.keras.layers.RandomHeight(height_shift_range=aug_config['height_shift_range']))
    if aug_config.get("shear_range", 0) > 0:
        pipeline.add(tf.keras.layers.RandomShear(shear_range=aug_config['shear_range']))
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
    metadata_path = config['metadata_path']
    data_dir = config['data_dir'] # Root directory containing images relative to project root
    image_size = tuple(config['image_size'])
    batch_size = config['batch_size']
    split_ratio = config['split_ratio']

    if not os.path.isabs(metadata_path):
         metadata_path = os.path.join(os.path.dirname(__file__), '..', '..', metadata_path)
    if not os.path.isabs(data_dir):
         data_dir = os.path.join(os.path.dirname(__file__), '..', '..', data_dir)

    try:
        if not 0 < split_ratio < 1:
            raise ValueError("Split ratio must be between 0 and 1.")

        logger.info(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata for {len(metadata)} items.")

        image_paths = []
        string_labels = []
        for item in metadata:
            full_path = os.path.join(data_dir, item['image_path'])
            if not os.path.exists(full_path):
                 logger.warning(f"Image file not found: {full_path}. Skipping.")
                 continue
            image_paths.append(full_path)
            string_labels.append(item['label'])

        if not image_paths:
            raise FileNotFoundError("No valid image paths found based on metadata.")

        # Create label mapping efficiently
        unique_labels = sorted(list(set(string_labels)))
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        index_to_label = {index: label for label, index in label_to_index.items()}
        num_classes = len(unique_labels)
        logger.info(f"Found {num_classes} unique classes.")
        if num_classes != config.get('num_classes'):
             logger.warning(f"Number of classes found ({num_classes}) differs from config ({config.get('num_classes')}). Using found number.")
             config['num_classes'] = num_classes # Update config if needed downstream

        # Convert string labels to integer indices
        integer_labels = [label_to_index[label] for label in string_labels]

        # Stratified split 
        logger.info(f"Splitting data with validation ratio {split_ratio} (stratified)...")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths,
            integer_labels,
            test_size=split_ratio,
            random_state=42, # for reproducibility
            stratify=integer_labels
        )
        logger.info(f"Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

        # Build augmentation pipeline (if enabled in config)
        augmentation_pipeline = _build_augmentation_pipeline(config)

        def load_and_preprocess(path: str, label: int, augment: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
            """Loads, decodes, resizes, normalizes, and optionally augments an image."""
            try:
                image = tf.io.read_file(path)
                # Use decode_image for flexibility, ensure 3 channels
                image = tf.image.decode_image(image, channels=3, expand_animations=False)
                image = tf.image.resize(image, image_size)
                image.set_shape([*image_size, 3]) # Explicitly set shape

                if augment and augmentation_pipeline is not None:
                    image = augmentation_pipeline(image, training=True)

                image = image / 255.0 # Normalize
                return image, label
            except Exception as e:
                 logger.error(f"Error processing image {path}: {e}")
                 raise

        AUTOTUNE = tf.data.AUTOTUNE

        train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        # Use a lambda to pass the 'augment=True' flag
        train_dataset = train_dataset.map(lambda p, l: load_and_preprocess(p, l, augment=True), num_parallel_calls=AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size=len(train_paths))
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        logger.info("Training dataset created.")

        # Create validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        # Use a lambda to pass the 'augment=False' flag
        val_dataset = val_dataset.map(lambda p, l: load_and_preprocess(p, l, augment=False), num_parallel_calls=AUTOTUNE)
        val_dataset = val_dataset.filter(lambda img, lbl: lbl != -1)
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
        logger.info("Validation dataset created.")

        return train_dataset, val_dataset, index_to_label

    except (FileNotFoundError, json.JSONDecodeError, ValueError, ModuleNotFoundError) as e:
        logger.error(f"Data loading failed: {e}")
        raise
    except ImportError:
        logger.error("Scikit-learn is required for stratified splitting. Please install it (`pip install scikit-learn`).")
        raise

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
        # dataset = dataset.filter(lambda img, lbl: lbl != -1) # Filter unknowns if needed
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        logger.info("Test dataset created.")

        return dataset, index_to_label_map

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Test data loading failed: {e}")
        raise