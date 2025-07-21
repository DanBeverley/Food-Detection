import os
import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Optional, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

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

    depth = tf.image.decode_png(example['depth_image_raw'], channels=1, dtype=tf.uint16)
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

@tf.function
def process_and_augment_batch(inputs, mask, do_flip, do_color_aug):
    # Sanitize point cloud
    pc = inputs['pc_input']
    pc = tf.where(tf.math.is_nan(pc), 0.0, pc)
    pc = tf.where(tf.math.is_inf(pc), 0.0, pc)
    inputs['pc_input'] = pc
    
    # Scale images to [0, 1] for color augmentation
    rgb_scaled_for_aug = inputs['rgb_input'] / 255.0
    depth_scaled_for_aug = inputs['depth_input'] / 255.0
    
    # Combine for synchronized geometric augmentations
    image_and_mask = tf.concat([rgb_scaled_for_aug, depth_scaled_for_aug, mask], axis=-1)
    
    # Conditional augmentation using tf.cond
    def flip_fn():
        flipped_imgs = tf.image.flip_left_right(image_and_mask)
        flipped_pc = inputs['pc_input'] * tf.constant([-1.0, 1.0, 1.0])
        return flipped_imgs, flipped_pc
    
    def no_flip_fn():
        return image_and_mask, inputs['pc_input']
    
    # Apply flip conditionally
    image_and_mask, final_pc = tf.cond(
        tf.logical_and(do_flip, tf.random.uniform(()) > 0.5),
        flip_fn,
        no_flip_fn
    )
    
    # Split back into components
    aug_rgb = image_and_mask[..., :3]
    aug_depth = image_and_mask[..., 3:6]
    aug_mask = image_and_mask[..., 6:]
    
    # Color augmentations using tf.cond on [0, 1] data
    def color_fn():
        bright = tf.image.random_brightness(aug_rgb, 0.1)
        contrast = tf.image.random_contrast(bright, 0.9, 1.1)
        return contrast
    
    def no_color_fn():
        return aug_rgb
    
    aug_rgb = tf.cond(do_color_aug, color_fn, no_color_fn)
    
    # Un-scale data back to [0, 255] for model preprocessing
    final_rgb = aug_rgb * 255.0
    final_depth = aug_depth * 255.0
    
    final_inputs = {'rgb_input': final_rgb, 'depth_input': final_depth, 'pc_input': final_pc}
    return final_inputs, aug_mask

def load_segmentation_data(config: Dict[str, Any]) -> Tuple[Optional[tf.data.Dataset], ...]:
    try:
        logger.info("Loading data with augmentation pipeline")
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
        
        # Get augmentation flags from config
        aug_cfg = data_cfg.get('augmentation', {})
        DO_FLIP = aug_cfg.get('horizontal_flip', False)
        DO_COLOR = aug_cfg.get('brightness', False) or aug_cfg.get('contrast', False)
        
        def create_dataset_from_tfrecord(tfrecord_filename, is_training):
            filepath = str(tfrecord_dir / tfrecord_filename)
            if not os.path.exists(filepath):
                logger.error(f"TFRecord file not found: {filepath}")
                return None
            
            dataset = tf.data.TFRecordDataset(filepath, num_parallel_reads=tf.data.AUTOTUNE)
            
            if is_training:
                dataset = dataset.shuffle(buffer_size=1024).repeat()

            dataset = dataset.map(
                lambda x: parse_tfrecord_fn(x, target_size, num_points_target),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            dataset = dataset.batch(batch_size, drop_remainder=True)
            
            if is_training:
                dataset = dataset.map(
                    lambda inputs, mask: process_and_augment_batch(inputs, mask, DO_FLIP, DO_COLOR),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            else:
                dataset = dataset.map(
                    lambda inputs, mask: process_and_augment_batch(inputs, mask, False, False),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            logger.info(f"Pipeline created for {tfrecord_filename}")
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