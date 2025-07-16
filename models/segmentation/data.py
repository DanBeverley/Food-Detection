import os
import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Optional, List, Any
from sklearn.model_selection import train_test_split
import pathlib
import json
from tensorflow.keras import layers

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
        
        self.geometric_layers = []
        if self.aug_config.get("horizontal_flip", False):
            self.geometric_layers.append(layers.RandomFlip("horizontal"))
        if self.aug_config.get("rotation_range", 0) > 0:
            self.geometric_layers.append(layers.RandomRotation(self.aug_config["rotation_range"] / 360.0))
        if self.aug_config.get("zoom_range", 0) > 0:
            zoom = self.aug_config["zoom_range"]
            self.geometric_layers.append(layers.RandomZoom(height_factor=(-zoom, zoom)))

        self.color_layers = []
        if self.aug_config.get("brightness_range", [1.0, 1.0]) != [1.0, 1.0]:
            factor = max(abs(1.0 - self.aug_config["brightness_range"][0]), abs(self.aug_config["brightness_range"][1] - 1.0))
            self.color_layers.append(layers.RandomBrightness(factor=factor))
        if self.aug_config.get("contrast_factor", 0) > 0:
            self.color_layers.append(layers.RandomContrast(factor=self.aug_config["contrast_factor"]))

    def call(self, inputs):
        inputs_dict, mask = inputs
        
        image_and_mask = layers.concatenate([inputs_dict['rgb_input'], inputs_dict['depth_input'], mask], axis=-1)
        
        for layer in self.geometric_layers:
            image_and_mask = layer(image_and_mask)

        augmented_rgb = image_and_mask[..., :3]
        augmented_depth = image_and_mask[..., 3:6]
        augmented_mask = image_and_mask[..., 6:]
        
        for layer in self.color_layers:
            augmented_rgb = layer(augmented_rgb)
        
        augmented_inputs = {
            'rgb_input': augmented_rgb,
            'depth_input': augmented_depth,
            'pc_input': inputs_dict['pc_input']
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
    
    mask = tf.image.decode_png(example['mask_image_raw'], channels=1)
    mask.set_shape([*target_size, 1])
    
    def decode_depth():
        d = tf.image.decode_png(example['depth_image_raw'], channels=1)
        d.set_shape([*target_size, 1])
        return d
    
    depth = tf.cond(
        tf.strings.length(example['depth_image_raw']) > 0, 
        decode_depth, 
        lambda: tf.zeros([*target_size, 1], dtype=tf.uint8)
    )
    
    pc = tf.io.decode_raw(example['point_cloud_raw'], tf.float32)
    pc = tf.reshape(pc, [num_points, 3])
    pc.set_shape([num_points, 3])
    
    return {
        "rgb_input": tf.cast(rgb, tf.float32),
        "depth_input": tf.cast(depth, tf.float32),
        "pc_input": pc,
        "mask": tf.cast(mask, tf.float32)
    }

def load_segmentation_data(config: Dict[str, Any]) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset], int, int, int, int]:
    try:
        data_cfg = config['data']
        paths_cfg = config['paths']
        model_cfg = config['model']
        aug_cfg = data_cfg.get('augmentation', {})
        
        target_size = tuple(data_cfg.get('image_size', (256, 256)))
        batch_size = data_cfg['batch_size']
        num_classes = data_cfg['num_classes']
        
        project_root = _get_project_root()
        tfrecord_dir = project_root / paths_cfg['metadata_dir'] / "tfrecords"
        metadata_path = project_root / paths_cfg['metadata_dir'] / paths_cfg['metadata_filename']
        
        # Load metadata to get sample counts
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
        
        valid_entries = [e for e in all_metadata if e.get('image_path') and e.get('mask_path')]
        train_meta, temp_meta = train_test_split(valid_entries, test_size=0.3, random_state=42)
        val_meta, test_meta = train_test_split(temp_meta, test_size=0.5, random_state=42)
        
        num_train_samples = len(train_meta)
        num_val_samples = len(val_meta)
        num_test_samples = len(test_meta)
        
        augmentation_pipeline = SegmentationAugmentation(aug_cfg)
        preprocess_fn = _get_segmentation_preprocess_fn(model_cfg.get('backbone'))
        
        pc_prep_cfg = data_cfg.get('modalities_preprocessing', {}).get('point_cloud', {})
        num_points_target = pc_prep_cfg.get('num_points', 4096)

        def create_dataset(tfrecord_filename, is_training):
            filepath = str(tfrecord_dir / tfrecord_filename)
            if not os.path.exists(filepath):
                logger.error(f"TFRecord file not found: {filepath}")
                return None
            
            dataset = tf.data.TFRecordDataset(filepath, num_parallel_reads=tf.data.AUTOTUNE)
            
            if is_training:
                dataset = dataset.shuffle(buffer_size=2048)
            
            dataset = dataset.map(
                lambda x: parse_tfrecord_fn(x, target_size, num_points_target),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            def augment_and_finalize(sample):
                inputs = {k: v for k, v in sample.items() if k != 'mask'}
                mask = sample['mask']
                
                depth_3_channel = tf.concat([inputs['depth_input'], inputs['depth_input'], inputs['depth_input']], axis=-1)
                inputs['depth_input'] = depth_3_channel
                
                if is_training and aug_cfg.get('enabled', False):
                    inputs, mask = augmentation_pipeline((inputs, mask))
                
                if preprocess_fn:
                    inputs['rgb_input'] = preprocess_fn(inputs['rgb_input'])
                    inputs['depth_input'] = preprocess_fn(inputs['depth_input'])
                
                return inputs, tf.squeeze(mask, axis=-1)

            dataset = dataset.map(augment_and_finalize, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset

        train_dataset = create_dataset("train.tfrecord", is_training=True)
        val_dataset = create_dataset("validation.tfrecord", is_training=False)
        test_dataset = create_dataset("test.tfrecord", is_training=False)

        return train_dataset, val_dataset, test_dataset, num_train_samples, num_val_samples, num_test_samples, num_classes

    except Exception as e:
        logger.error(f"An unexpected error occurred in load_segmentation_data: {e}", exc_info=True)
        return None, None, None, 0, 0, 0, 0