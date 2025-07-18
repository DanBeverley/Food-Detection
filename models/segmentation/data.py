import os
import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Optional, List, Any
from sklearn.model_selection import train_test_split
import pathlib
import json
from tensorflow.keras import layers
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_SEG_PREPROCESS_FN_CACHE = {}

def _get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent.parent

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

class SegmentationAugmentation(tf.keras.Model):
    def __init__(self, aug_config, **kwargs):
        super().__init__(**kwargs)
        self.aug_config = aug_config

        # Use Keras Layers for Image Augmentations
        self.geometric_layers = []
        if self.aug_config.get("horizontal_flip", False):
            self.geometric_layers.append(layers.RandomFlip("horizontal"))
        if self.aug_config.get("rotation_range", 0) > 0:
            rotation_factor = self.aug_config["rotation_range"] / 360.0
            self.geometric_layers.append(layers.RandomRotation(rotation_factor))

        self.color_layers = []
        if self.aug_config.get("brightness_range", [1.0, 1.0]) != [1.0, 1.0]:
            factor = max(abs(1.0 - self.aug_config["brightness_range"][0]), abs(self.aug_config["brightness_range"][1] - 1.0))
            self.color_layers.append(layers.RandomBrightness(factor=factor, value_range=(0.0, 255.0)))
        if self.aug_config.get("contrast_factor", 0) > 0:
            self.color_layers.append(layers.RandomContrast(factor=self.aug_config["contrast_factor"]))

    def call(self, inputs):
        # This call now happens AFTER batching, so inputs are batched
        inputs_dict, mask = inputs
        
        # Augment Images and Mask Together
        image_and_mask = tf.concat([inputs_dict['rgb_input'], inputs_dict['depth_input'], mask], axis=-1)
        
        for layer in self.geometric_layers:
            image_and_mask = layer(image_and_mask, training=True)
            
        # De-concatenate
        augmented_rgb = image_and_mask[..., :3]
        augmented_depth = image_and_mask[..., 3:6]
        augmented_mask = image_and_mask[..., 6:]
        
        # Augment Point Cloud Separately (vectorized for entire batch)
        augmented_pc = inputs_dict['pc_input']

        # Random Flip (vectorized)
        if self.aug_config.get("horizontal_flip", False):
            batch_size = tf.shape(augmented_pc)[0]
            flip_cond = tf.random.uniform(shape=[batch_size, 1, 1]) < 0.5
            flip_multiplier = tf.where(flip_cond, -1.0, 1.0)
            pc_flip_transform = tf.concat([flip_multiplier, tf.ones_like(flip_multiplier), tf.ones_like(flip_multiplier)], axis=-1)
            augmented_pc = augmented_pc * pc_flip_transform

        # Random Rotation (vectorized)
        if self.aug_config.get("rotation_range", 0) > 0:
            batch_size = tf.shape(augmented_pc)[0]
            rotation_degrees = self.aug_config["rotation_range"]
            angles = tf.random.uniform(shape=[batch_size], minval=-rotation_degrees, maxval=rotation_degrees)
            angles_rad = angles * np.pi / 180.0
            
            # Create rotation matrices for the entire batch
            cos_angles = tf.cos(angles_rad)
            sin_angles = tf.sin(angles_rad)
            zeros = tf.zeros_like(cos_angles)
            ones = tf.ones_like(cos_angles)
            
            # Stack rotation matrices [batch_size, 3, 3]
            rotation_matrices = tf.stack([
                tf.stack([cos_angles, -sin_angles, zeros], axis=1),
                tf.stack([sin_angles, cos_angles, zeros], axis=1),
                tf.stack([zeros, zeros, ones], axis=1)
            ], axis=1)
            
            # Apply batch matrix multiplication
            augmented_pc = tf.linalg.matmul(augmented_pc, rotation_matrices, transpose_b=True)

        # Apply Color Augmentations
        for layer in self.color_layers:
            augmented_rgb = layer(augmented_rgb, training=True)
            
        augmented_inputs = {
            'rgb_input': augmented_rgb,
            'depth_input': augmented_depth,
            'pc_input': augmented_pc
        }
        return augmented_inputs, augmented_mask

def parse_tfrecord_fn(example_proto, target_size, num_points):
    """Parses a single tf.train.Example from a TFRecord file into tensors."""
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

    def decode_depth():
        d = tf.image.decode_png(example['depth_image_raw'], channels=1)
        d.set_shape([*target_size, 1])
        return tf.cast(d, tf.float32)
    
    depth = tf.cond(
        tf.strings.length(example['depth_image_raw']) > 0, 
        decode_depth, 
        lambda: tf.zeros([*target_size, 1], dtype=tf.float32)
    )
    
    # Replicate depth channels to match model input
    depth = tf.concat([depth, depth, depth], axis=-1)

    pc = tf.io.decode_raw(example['point_cloud_raw'], tf.float32)
    pc = tf.reshape(pc, [num_points, 3])
    pc.set_shape([num_points, 3])
    
    # Return a TUPLE that the model expects: (x, y) where x can be a dictionary
    inputs = {
        "rgb_input": rgb,
        "depth_input": depth,
        "pc_input": pc,
    }
    return inputs, mask

def load_segmentation_data(config: Dict[str, Any]) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset], int, int, int, int]:
    try:
        logger.info("--- [DEBUG MODE] Starting Simplified load_segmentation_data ---")
        
        data_cfg = config['data']
        paths_cfg = config['paths']
        
        target_size = tuple(data_cfg.get('image_size', (256, 256)))
        batch_size = data_cfg['batch_size']
        
        tfrecord_dir = Path(paths_cfg.get('tfrecord_dir', paths_cfg['metadata_dir'] + "/tfrecords"))
        logger.info(f"TFRecord directory set to: {tfrecord_dir}")

        num_train_samples = data_cfg.get('num_train_samples', 0)
        num_val_samples = data_cfg.get('num_val_samples', 0)
        num_test_samples = data_cfg.get('num_test_samples', 0)
        num_classes = data_cfg.get('num_classes', 1)
        
        pc_prep_cfg = data_cfg.get('modalities_preprocessing', {}).get('point_cloud', {})
        num_points_target = pc_prep_cfg.get('num_points', 4096)

        augmentation_pipeline = SegmentationAugmentation(aug_cfg)
        preprocess_fn = _get_segmentation_preprocess_fn(model_cfg.get('backbone'))
        
        def finalize_pre_batch(inputs, mask):
            inputs['rgb_input'] = (inputs['rgb_input'] / 127.5) - 1.0
            inputs['depth_input'] = (inputs['depth_input'] / 127.5) - 1.0
            
            return inputs, mask

        def augment_batch(inputs, mask):
            return augmentation_pipeline(inputs, mask)

        def create_dataset_from_tfrecord(tfrecord_filename, is_training):
            filepath = str(tfrecord_dir / tfrecord_filename)
            if not os.path.exists(filepath):
                logger.error(f"FATAL: TFRecord file not found: {filepath}")
                return None
            
            dataset = tf.data.TFRecordDataset(
                filepath, 
                num_parallel_reads=tf.data.AUTOTUNE
            )
            
            if is_training:
                # IMPORTANT: repeat() is essential for multi-epoch training
                dataset = dataset.shuffle(buffer_size=1024).repeat()

            dataset = dataset.map(
                lambda x: parse_tfrecord_fn(x, target_size, num_points_target),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            dataset = dataset.map(finalize_pre_batch, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            
            # if is_training and aug_cfg.get('enabled', False):
            #     dataset = dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
            
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            logger.info(f"[{tfrecord_filename}] Test pipeline created (Preprocessing ONLY).")
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