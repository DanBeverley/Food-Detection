import os
import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def parse_and_sanitize_fn(example_proto, target_size, num_points):
    feature_description = {
        'rgb_image_raw': tf.io.FixedLenFeature([], tf.string),
        'mask_image_raw': tf.io.FixedLenFeature([], tf.string),
        'depth_image_raw': tf.io.FixedLenFeature([], tf.string),
        'point_cloud_raw': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    rgb = tf.image.decode_png(example['rgb_image_raw'], channels=3)
    rgb.set_shape([*target_size, 3])
    rgb = tf.cast(rgb, tf.float32)

    mask = tf.image.decode_png(example['mask_image_raw'], channels=1)
    mask.set_shape([*target_size, 1])
    mask = tf.cast(mask, tf.float32)

    depth = tf.image.decode_png(example['depth_image_raw'], channels=1, dtype=tf.uint16)
    depth.set_shape([*target_size, 1])
    depth = tf.cast(depth, tf.float32)
    depth = tf.concat([depth, depth, depth], axis=-1)

    pc = tf.io.decode_raw(example['point_cloud_raw'], tf.float32)
    pc = tf.reshape(pc, [num_points, 3])
    pc.set_shape([num_points, 3])
    
    # Sanitize point cloud only
    pc = tf.where(tf.math.is_nan(pc), 0.0, pc)
    pc = tf.where(tf.math.is_inf(pc), 0.0, pc)
    
    inputs = {
        "rgb_input": rgb,
        "depth_input": depth,
        "pc_input": pc,
    }
    return inputs, mask

def load_segmentation_data(config: Dict[str, Any]) -> Tuple[Optional[tf.data.Dataset], ...]:
    try:
        logger.info("Loading data with ultra-stable pipeline")
        data_cfg = config['data']
        paths_cfg = config['paths']
        
        target_size = tuple(data_cfg.get('image_size', (256, 256)))
        batch_size = data_cfg['batch_size']
        num_classes = data_cfg.get('num_classes', 1)
        
        tfrecord_dir = Path(paths_cfg.get('tfrecord_dir'))
        num_train_samples = data_cfg.get('num_train_samples', 0)
        num_val_samples = data_cfg.get('num_val_samples', 0)
        num_test_samples = data_cfg.get('num_test_samples', 0)
        
        pc_prep_cfg = data_cfg.get('modalities_preprocessing', {}).get('point_cloud', {})
        num_points_target = pc_prep_cfg.get('num_points', 4096)
        
        def create_dataset_from_tfrecord(tfrecord_filename, is_training):
            filepath = str(tfrecord_dir / tfrecord_filename)
            if not os.path.exists(filepath):
                logger.error(f"TFRecord file not found: {filepath}")
                return None
            
            dataset = tf.data.TFRecordDataset(filepath, num_parallel_reads=tf.data.AUTOTUNE)
            
            if is_training:
                dataset = dataset.shuffle(buffer_size=1024).repeat()

            # ONLY ONE MAP CALL - the simplest possible function
            dataset = dataset.map(
                lambda x: parse_and_sanitize_fn(x, target_size, num_points_target),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            logger.info(f"Ultra-stable pipeline created for {tfrecord_filename}")
            return dataset
        
        logger.info("Creating training dataset...")
        train_dataset = create_dataset_from_tfrecord("train.tfrecord", is_training=True)
        
        logger.info("Creating validation dataset...")
        val_dataset = create_dataset_from_tfrecord("validation.tfrecord", is_training=False)
        
        logger.info("Creating test dataset...")
        test_dataset = create_dataset_from_tfrecord("test.tfrecord", is_training=False)

        logger.info("Data loading completed")
        return train_dataset, val_dataset, test_dataset, num_train_samples, num_val_samples, num_test_samples, num_classes

    except Exception as e:
        logger.error(f"Error in load_segmentation_data: {e}", exc_info=True)
        return None, None, None, 0, 0, 0, 0