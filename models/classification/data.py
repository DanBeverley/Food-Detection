import json
import tensorflow as tf
from typing import Tuple, Dict, Optional, Any, List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_decode_image(path: tf.Tensor, label: tf.Tensor, image_size: tuple) -> Tuple[tf.Tensor, tf.Tensor]:
    """Loads, decodes, and resizes an image file, returning it in [0, 255] range."""
    image_data = tf.io.read_file(path)
    image = tf.image.decode_image(image_data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    return image, label

def load_classification_data(
    config: Dict[str, Any]
) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], int, int, List[str], int]:
    """Loads, splits, and prepares the classification dataset."""
    data_cfg = config.get('data', {})
    paths_cfg = data_cfg.get('paths', {})
    
    # Use Path objects for robust path handling
    metadata_path = Path(paths_cfg.get('metadata_file'))
    label_map_path = Path(paths_cfg.get('label_map_file'))
    base_image_dir = Path(data_cfg.get('base_data_dir', '.'))

    image_size = tuple(data_cfg.get('image_size', [224, 224]))
    batch_size = data_cfg.get('batch_size', 32)
    val_split = data_cfg.get('validation_split', 0.2)
    seed = data_cfg.get('random_seed', 42)

    logger.info(f"Loading metadata from: {metadata_path}")
    if not metadata_path.exists() or not label_map_path.exists():
        logger.error(f"Metadata ({metadata_path}) or label map ({label_map_path}) not found.")
        return None, None, 0, 0, [], 0

    with open(label_map_path, 'r') as f:
        index_to_label_map = {int(k): v for k, v in json.load(f).items()}
        num_classes = len(index_to_label_map)
        class_names = [index_to_label_map[i] for i in sorted(index_to_label_map.keys())]
        label_to_index_map = {v: k for k, v in index_to_label_map.items()}

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    all_paths, all_labels = [], []
    for item in metadata:
        relative_path = item.get('image_path')
        class_name = item.get('class_name')
        full_path = base_image_dir / relative_path
        if full_path.exists() and class_name in label_to_index_map:
            all_paths.append(str(full_path))
            all_labels.append(label_to_index_map[class_name])
        else:
            logger.warning(f"Skipping invalid entry: path={full_path}, class={class_name}")

    if not all_paths:
        logger.error("No valid image paths found after processing metadata.")
        return None, None, 0, 0, [], 0
    
    logger.info(f"Found {len(all_paths)} valid image samples.")

    full_dataset = tf.data.Dataset.from_tensor_slices((all_paths, all_labels))
    full_dataset = full_dataset.shuffle(buffer_size=len(all_paths), seed=seed, reshuffle_each_iteration=False)

    num_val_samples = int(len(all_paths) * val_split)
    num_train_samples = len(all_paths) - num_val_samples
    
    val_dataset_raw = full_dataset.take(num_val_samples)
    train_dataset_raw = full_dataset.skip(num_val_samples)
    logger.info(f"Dataset split: {num_train_samples} training, {num_val_samples} validation.")

    def configure_dataset(ds: tf.data.Dataset, is_training: bool) -> tf.data.Dataset:
        ds = ds.map(lambda path, label: load_and_decode_image(path, label, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        if is_training:
            ds = ds.shuffle(1024).repeat()
        ds = ds.batch(batch_size)
        ds = ds.map(lambda img, lbl: (img, tf.one_hot(tf.cast(lbl, tf.int32), depth=num_classes)), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds
        
    train_ds = configure_dataset(train_dataset_raw, is_training=True)
    val_ds = configure_dataset(val_dataset_raw, is_training=False) if num_val_samples > 0 else None

    return train_ds, val_ds, num_train_samples, num_val_samples, class_names, num_classes