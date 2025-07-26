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
    """Loads pre-split classification dataset using separate train/val metadata files."""
    data_cfg = config.get('data', {})
    paths_cfg = data_cfg.get('paths', {})
    
    # Use Path objects for robust path handling
    train_metadata_path = Path(paths_cfg.get('train_metadata_file'))
    val_metadata_path = Path(paths_cfg.get('val_metadata_file'))
    label_map_path = Path(paths_cfg.get('label_map_file'))
    base_image_dir = Path(data_cfg.get('base_data_dir', '.'))

    image_size = tuple(data_cfg.get('image_size', [224, 224]))
    batch_size = data_cfg.get('batch_size', 32)
    seed = data_cfg.get('random_seed', 42)

    logger.info(f"Loading train metadata from: {train_metadata_path}")
    logger.info(f"Loading val metadata from: {val_metadata_path}")
    
    if not train_metadata_path.exists() or not val_metadata_path.exists() or not label_map_path.exists():
        logger.error(f"Required files not found: train={train_metadata_path}, val={val_metadata_path}, labels={label_map_path}")
        return None, None, 0, 0, [], 0

    with open(label_map_path, 'r') as f:
        index_to_label_map = {int(k): v for k, v in json.load(f).items()}
        num_classes = len(index_to_label_map)
        class_names = [index_to_label_map[i] for i in sorted(index_to_label_map.keys())]
        label_to_index_map = {v: k for k, v in index_to_label_map.items()}

    def load_metadata_paths_labels(metadata_path: Path, dataset_name: str):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        paths, labels = [], []
        for item in metadata:
            relative_path = item.get('image_path')
            class_name = item.get('class_name')
            full_path = base_image_dir / relative_path
            if full_path.exists() and class_name in label_to_index_map:
                paths.append(str(full_path))
                labels.append(label_to_index_map[class_name])
            else:
                logger.warning(f"Skipping invalid {dataset_name} entry: path={full_path}, class={class_name}")
        
        logger.info(f"Found {len(paths)} valid {dataset_name} samples.")
        return paths, labels

    train_paths, train_labels = load_metadata_paths_labels(train_metadata_path, "training")
    val_paths, val_labels = load_metadata_paths_labels(val_metadata_path, "validation")

    if not train_paths:
        logger.error("No valid training image paths found.")
        return None, None, 0, 0, [], 0
    
    num_train_samples = len(train_paths)
    num_val_samples = len(val_paths)
    logger.info(f"Instance-based split: {num_train_samples} training, {num_val_samples} validation samples.")

    train_dataset_raw = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_dataset_raw = tf.data.Dataset.from_tensor_slices((val_paths, val_labels)) if val_paths else None

    def configure_dataset(ds: tf.data.Dataset, is_training: bool) -> tf.data.Dataset:
        ds = ds.map(lambda path, label: load_and_decode_image(path, label, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        if is_training:
            ds = ds.shuffle(1024).repeat()
        ds = ds.batch(batch_size)
        ds = ds.map(lambda img, lbl: (img, tf.one_hot(tf.cast(lbl, tf.int32), depth=num_classes)), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds
        
    train_ds = configure_dataset(train_dataset_raw, is_training=True)
    val_ds = configure_dataset(val_dataset_raw, is_training=False) if val_dataset_raw is not None else None

    return train_ds, val_ds, num_train_samples, num_val_samples, class_names, num_classes