import json
import tensorflow as tf
from typing import Tuple
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_segmentation_data(metadata_path: str, data_dir: str, image_size: Tuple[int, int] = (512, 512), batch_size: int = 16, split_ratio: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load and prepare segmentation data from metadata JSON, splitting into train and validation sets.

    Args:
        metadata_path: Path to JSON metadata file (e.g., from data loading scripts, with 'image_path' and 'mask_path').
        data_dir: Root directory containing processed images and masks (e.g., subfolders for classes).
        image_size: Tuple of (height, width) for image and mask resizing.
        batch_size: Batch size for data loading.
        split_ratio: Fraction of data to use for validation (e.g., 0.2 for 20%).

    Returns:
        Tuple of (train_dataset, val_dataset) as tf.data.Dataset objects, each yielding (image, mask) pairs.

    Raises:
        FileNotFoundError: If metadata or files are missing.
        ValueError: If invalid metadata format or split ratio.
        json.JSONDecodeError: If metadata file is corrupted.
    """
    try:
        if not 0 < split_ratio < 1:
            raise ValueError("Split ratio must be between 0 and 1.")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract image and mask paths
        image_paths = [item['image_path'] for item in metadata]
        mask_paths = [item['mask_path'] for item in metadata]
        labels = [item['label'] for item in metadata]  # Optional, if needed for segmentation tasks
        
        data_pairs = list(zip(image_paths, mask_paths, labels))
        data_array = np.array(data_pairs)
        train_data, val_data = np.split(data_array, [int((1-split_ratio)*len(data_array))], axis=0)
        train_data = train_data.tolist()
        val_data = val_data.tolist()
        
        def load_and_preprocess_segmentation(img_path: str, mask_path: str, label: str) -> Tuple[tf.Tensor, tf.Tensor]:
            """
            Load and preprocess a single image-mask pair.

            Args:
                img_path: Path to the image file.
                mask_path: Path to the mask file.
                label: Label string (if used, otherwise ignore).

            Returns:
                Tuple of (image tensor, mask tensor), both normalized and resized.
            """
            image = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, image_size)
            image = image / 255.0  # Normalize image
            
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=1)  # adjustable
            mask = tf.image.resize(mask, image_size, method='nearest')  # Use nearest for masks to avoid interpolation artifacts
            mask = mask / 255.0 
            return image, mask  #NOTE: Ignore label for now, or include if multi-class segmentation requires it

        # Create datasets with parallelism and prefetching
        train_dataset = tf.data.Dataset.from_tensor_slices(( [p[0] for p in train_data], [p[1] for p in train_data] ))  # (image_path, mask_path)
        train_dataset = train_dataset.map(lambda img_path, mask_path: load_and_preprocess_segmentation(img_path, mask_path, ''), num_parallel_calls=tf.data.AUTOTUNE)  # Label not used here
        train_dataset = train_dataset.shuffle(buffer_size=len(train_data)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices(( [p[0] for p in val_data], [p[1] for p in val_data] ))
        val_dataset = val_dataset.map(lambda img_path, mask_path: load_and_preprocess_segmentation(img_path, mask_path, ''), num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Data loading error: {e}")
        raise

def load_segmentation_test_data(metadata_path: str, data_dir: str, image_size: Tuple[int, int] = (512, 512), batch_size: int = 16) -> tf.data.Dataset:
    """
    Load data for segmentation evaluation or inference from metadata JSON (no splitting).

    Args:
        metadata_path: Path to JSON metadata file.
        data_dir: Root directory containing processed images and masks.
        image_size: Tuple of (height, width) for image and mask resizing.
        batch_size: Batch size for data loading.

    Returns:
        Test tf.data.Dataset yielding (image, mask) pairs.

    Raises:
        FileNotFoundError: If metadata or files are missing.
        ValueError: If invalid metadata format.
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        image_paths = [item['image_path'] for item in metadata]
        mask_paths = [item['mask_path'] for item in metadata]
        
        def load_and_preprocess_segmentation(img_path: str, mask_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
            """
            Load and preprocess a single image-mask pair for evaluation.

            Args:
                img_path: Path to the image file.
                mask_path: Path to the mask file.

            Returns:
                Tuple of (image tensor, mask tensor).
            """
            image = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, image_size)
            image = image / 255.0
            
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=1)
            mask = tf.image.resize(mask, image_size, method='nearest')
            mask = mask / 255.0  # Or keep as integer if mask is label-based
            
            return image, mask
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(load_and_preprocess_segmentation, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
    
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Data loading error: {e}")
        raise