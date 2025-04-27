import json
import tensorflow as tf
from typing import Tuple, List, Dict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_classification_data(metadata_path: str, data_dir: str, image_size: Tuple[int, int] = (224, 224), batch_size: int = 32, split_ratio: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load and prepare classification data from metadata JSON, splitting into train and validation sets.

    Args:
        metadata_path: Path to JSON metadata file (e.g., from data loading scripts).
        data_dir: Root directory containing processed images (e.g., class subfolders).
        image_size: Tuple of (height, width) for image resizing.
        batch_size: Batch size for data loading.
        split_ratio: Fraction of data to use for validation (e.g., 0.2 for 20%).

    Returns:
        Tuple of (train_dataset, val_dataset) as tf.data.Dataset objects.

    Raises:
        FileNotFoundError: If metadata or images are missing.
        ValueError: If invalid metadata format or split ratio.
        json.JSONDecodeError: If metadata file is corrupted.
    """
    try:
        if not 0 < split_ratio < 1:
            raise ValueError("Split ratio must be between 0 and 1.")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        images = [item['image_path'] for item in metadata]
        labels = [item['label'] for item in metadata]
        
        # Split data
        train_images, val_images, train_labels, val_labels = np.split(np.array(list(zip(images, labels))), [int((1-split_ratio)*len(images))], axis=0)
        train_images, train_labels = train_images.tolist(), train_labels.tolist()
        val_images, val_labels = val_images.tolist(), val_labels.tolist()
        
        def load_and_preprocess_image(path: str, label: str) -> Tuple[tf.Tensor, tf.Tensor]:
            """
            Load and preprocess a single image.

            Args:
                path: Path to the image file.
                label: Label string for the image.

            Returns:
                Tuple of (image tensor, label tensor).
            """
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)  # Adjust for other formats if needed
            image = tf.image.resize(image, image_size)
            image = image / 255.0  # Normalize
            label_idx = tf.argmax(tf.constant([label == lbl for lbl in np.unique(labels)]))  # Map string to index
            return image, label_idx
        
        # Create datasets with parallelism and prefetching
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size=len(train_images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_dataset = val_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Data loading error: {e}")
        raise

def load_test_data(metadata_path: str, data_dir: str, image_size: Tuple[int, int] = (224, 224), batch_size: int = 32) -> tf.data.Dataset:
    """
    Load data for evaluation or inference from metadata JSON (no splitting).

    Args:
        metadata_path: Path to JSON metadata file.
        data_dir: Root directory containing processed images.
        image_size: Tuple of (height, width) for image resizing.
        batch_size: Batch size for data loading.

    Returns:
        Test tf.data.Dataset.

    Raises:
        FileNotFoundError: If metadata or images are missing.
        ValueError: If invalid metadata format.
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        images = [item['image_path'] for item in metadata]
        labels = [item['label'] for item in metadata]
        
        def load_and_preprocess_image(path: str, label: str) -> Tuple[tf.Tensor, tf.Tensor]:
            """
            Load and preprocess a single image for evaluation.

            Args:
                path: Path to the image file.
                label: Label string for the image.

            Returns:
                Tuple of (image tensor, label tensor).
            """
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, image_size)
            image = image / 255.0
            label_idx = tf.argmax(tf.constant([label == lbl for lbl in np.unique(labels)]))
            return image, label_idx
        
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
    
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Data loading error: {e}")
        raise