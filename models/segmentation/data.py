import os
import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
from tensorflow.keras import layers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_SEG_PREPROCESS_FN_CACHE = {}

def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def _get_segmentation_preprocess_fn(architecture: Optional[str]):
    if architecture is None:
        return None
    
    arch_key = architecture.lower()
    if arch_key in _SEG_PREPROCESS_FN_CACHE:
        return _SEG_PREPROCESS_FN_CACHE[arch_key]
    
    preprocess_input_fn = None
    try:
        if 'efficientnet' in arch_key:
            from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
            preprocess_input_fn = efficientnet_preprocess
        elif 'resnet' in arch_key:
            from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
            preprocess_input_fn = resnet_preprocess
        elif 'mobilenet' in arch_key:
            from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
            preprocess_input_fn = mobilenet_preprocess
        else:
            logger.warning(f"Unknown architecture '{architecture}' for preprocessing. Using identity function.")
            preprocess_input_fn = lambda x: x
    except ImportError as e:
        logger.error(f"Could not import preprocessing function for '{architecture}': {e}")
        preprocess_input_fn = lambda x: x
    
    _SEG_PREPROCESS_FN_CACHE[arch_key] = preprocess_input_fn
    return preprocess_input_fn


def parse_tfrecord_fn(example_proto, target_size, num_points):
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

    depth = tf.image.decode_png(example['depth_image_raw'], channels=1)
    depth.set_shape([*target_size, 1])
    depth = tf.cast(depth, tf.float32)
    depth = tf.concat([depth, depth, depth], axis=-1)

    pc = tf.io.decode_raw(example['point_cloud_raw'], tf.float32)
    pc = tf.reshape(pc, [num_points, 3])
    pc.set_shape([num_points, 3])
    
    inputs = {
        "rgb_input": rgb,
        "depth_input": depth,
        "pc_input": pc,
    }
    return inputs, mask

def load_segmentation_data(config: Dict[str, Any]) -> Tuple[Optional[tf.data.Dataset], ...]:
    try:
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

        geometric_layers = []
        if aug_cfg.get("horizontal_flip", False):
            geometric_layers.append(layers.RandomFlip("horizontal"))
        if aug_cfg.get("rotation_range", 0) > 0:
            rotation_factor = aug_cfg["rotation_range"] / 360.0
            geometric_layers.append(layers.RandomRotation(rotation_factor))

        @tf.function
        def augment_batch(inputs, mask):
            image_and_mask = tf.concat([inputs['rgb_input'], inputs['depth_input'], mask], axis=-1)
            
            for layer in geometric_layers:
                image_and_mask = layer(image_and_mask, training=True)
                
            augmented_rgb = image_and_mask[..., :3]
            augmented_depth = image_and_mask[..., 3:6]
            augmented_mask = image_and_mask[..., 6:]
            
            augmented_pc = inputs['pc_input']
            pc_dtype = augmented_pc.dtype
            
            if aug_cfg.get("horizontal_flip", False):
                batch_size = tf.shape(augmented_pc)[0]
                flip_cond = tf.random.uniform(shape=[batch_size, 1, 1]) < 0.5
                flip_multiplier = tf.where(flip_cond, tf.cast(-1.0, pc_dtype), tf.cast(1.0, pc_dtype))
                pc_flip_transform = tf.concat([flip_multiplier, tf.ones_like(flip_multiplier), tf.ones_like(flip_multiplier)], axis=-1)
                augmented_pc = augmented_pc * pc_flip_transform
                
            # if aug_cfg.get("rotation_range", 0) > 0:
            #     batch_size = tf.shape(augmented_pc)[0]
            #     rotation_degrees = aug_cfg["rotation_range"]
            #     angles = tf.random.uniform(shape=[batch_size], minval=-rotation_degrees, maxval=rotation_degrees)
            #     angles_rad = angles * np.pi / 180.0
            #     cos_angles, sin_angles = tf.cos(angles_rad), tf.sin(angles_rad)
            #     zeros, ones = tf.zeros_like(cos_angles), tf.ones_like(cos_angles)
            #     rotation_matrices = tf.stack([
            #         tf.stack([cos_angles, -sin_angles, zeros], axis=1),
            #         tf.stack([sin_angles, cos_angles, zeros], axis=1),
            #         tf.stack([zeros, zeros, ones], axis=1)
            #     ], axis=1)
            #     rotation_matrices = tf.cast(rotation_matrices, pc_dtype)
            #     augmented_pc = tf.linalg.matmul(augmented_pc, rotation_matrices, transpose_b=True)
                
            augmented_inputs = {
                'rgb_input': augmented_rgb,
                'depth_input': augmented_depth,
                'pc_input': augmented_pc
            }
            return augmented_inputs, augmented_mask

        def finalize_pre_batch(inputs, mask):
            pc = inputs['pc_input']
            pc = tf.where(tf.math.is_nan(pc), 0.0, pc)
            pc = tf.where(tf.math.is_inf(pc), 0.0, pc)
            inputs['pc_input'] = pc
            return inputs, mask

        def create_dataset_from_tfrecord(tfrecord_filename, is_training):
            filepath = str(tfrecord_dir / tfrecord_filename)
            if not os.path.exists(filepath):
                logger.error(f"TFRecord file not found: {filepath}")
                return None
            
            dataset = tf.data.TFRecordDataset(filepath, num_parallel_reads=tf.data.AUTOTUNE)
            
            if is_training:
                dataset = dataset.shuffle(buffer_size=1024).repeat()

            dataset = dataset.map(lambda x: parse_tfrecord_fn(x, target_size, num_points_target), num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(finalize_pre_batch, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            
            if is_training and aug_cfg.get('enabled', False):
                dataset = dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
            
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            return dataset
        
        logger.info("Creating training dataset...")
        train_dataset = create_dataset_from_tfrecord("train.tfrecord", is_training=True)
        
        logger.info("Creating validation dataset...")
        val_dataset = create_dataset_from_tfrecord("validation.tfrecord", is_training=False)
        
        logger.info("Creating test dataset...")
        test_dataset = create_dataset_from_tfrecord("test.tfrecord", is_training=False)

        logger.info("--- Finished load_segmentation_data (Full-Featured) ---")
        return train_dataset, val_dataset, test_dataset, num_train_samples, num_val_samples, num_test_samples, num_classes

    except Exception as e:
        logger.error(f"An unexpected error occurred in load_segmentation_data: {e}", exc_info=True)
        return None, None, None, 0, 0, 0, 0