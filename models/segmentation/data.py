import os
import yaml 
import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Optional, List, Any, Callable
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
        
        # Concatenate for geometric transformations
        image_and_mask = layers.concatenate([inputs_dict['rgb_input'], inputs_dict['depth_input'], mask], axis=-1)
        
        # Apply geometric layers
        for layer in self.geometric_layers:
            image_and_mask = layer(image_and_mask)

        # Split back
        augmented_rgb = image_and_mask[..., :3]
        augmented_depth = image_and_mask[..., 3:6]
        augmented_mask = image_and_mask[..., 6:]
        
        # Apply color layers to RGB only
        for layer in self.color_layers:
            augmented_rgb = layer(augmented_rgb)
        
        augmented_inputs = {
            'rgb_input': augmented_rgb,
            'depth_input': augmented_depth,
            'pc_input': inputs_dict['pc_input']
        }
        return augmented_inputs, augmented_mask

def data_generator(metadata_list: List[Dict], data_cfg: Dict):
    """
    A pure Python generator that loads data using the absolute paths
    provided in the metadata list.
    """
    use_depth = data_cfg.get('use_depth_map', False)
    use_pc = data_cfg.get('use_point_cloud', False)
    pc_prep_cfg = data_cfg.get('modalities_preprocessing', {}).get('point_cloud', {})
    num_points_target = pc_prep_cfg.get('num_points', 4096)

    while True:
        np.random.shuffle(metadata_list)
        for item_data in metadata_list:
            try:
                full_rgb_path = item_data.get('image_path')
                full_mask_path = item_data.get('mask_path')
                
                if not (full_rgb_path and os.path.exists(full_rgb_path) and 
                        full_mask_path and os.path.exists(full_mask_path)):
                    continue

                from PIL import Image

                with open(full_rgb_path, 'rb') as f:
                    rgb_img = Image.open(f).convert('RGB')
                    rgb_np = np.array(rgb_img, dtype=np.float32)

                with open(full_mask_path, 'rb') as f:
                    mask_img = Image.open(f).convert('L')
                    mask_np = np.array(mask_img, dtype=np.float32)
                    mask_np = np.expand_dims(mask_np, axis=-1)

                depth_np = np.zeros_like(mask_np, dtype=np.float32)
                full_depth_path = item_data.get('depth_map_path')
                if use_depth and full_depth_path and os.path.exists(full_depth_path):
                    with open(full_depth_path, 'rb') as f:
                        depth_img = Image.open(f).convert('L')
                        depth_np = np.array(depth_img, dtype=np.float32)
                        depth_np = np.expand_dims(depth_np, axis=-1)
                
                pc_np = np.zeros((num_points_target, 3), dtype=np.float32)
                full_pc_path = item_data.get('point_cloud_path')
                if use_pc and full_pc_path and os.path.exists(full_pc_path):
                    pc_np = np.load(full_pc_path).astype(np.float32)
                
                yield {
                    "rgb_input": rgb_np,
                    "depth_input": depth_np,
                    "pc_input": pc_np,
                    "mask": mask_np
                }
            except Exception as e:
                logger.warning(f"Skipping sample {item_data.get('image_path')} due to generator error: {e}")
                continue

def load_segmentation_data(config: Dict[str, Any]) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset], int, int, int, int]:
    try:
        data_cfg = config['data']
        paths_cfg = config['paths']
        model_cfg = config['model'] 
        training_cfg = config.get('training', {})
        aug_cfg = data_cfg.get('augmentation', {})

        target_size = tuple(data_cfg.get('image_size', (256, 256))) 
        batch_size = data_cfg['batch_size']
        num_classes = data_cfg['num_classes']
        split_ratios = data_cfg['split_ratios']
        random_seed = data_cfg.get('random_seed', 42)

        debug_settings = config.get('debug', {})
        debug_mode = debug_settings.get('enabled', False) or training_cfg.get('debug_mode', False)
        debug_max_samples = debug_settings.get('max_samples_per_split') or data_cfg.get('debug_max_samples')

        project_root = _get_project_root()
        metadata_dir = project_root / paths_cfg['metadata_dir']
        metadata_path = metadata_dir / paths_cfg['metadata_filename']

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)

        if debug_mode and debug_max_samples:
            all_metadata = all_metadata[:debug_max_samples]
            logger.info(f"Debug mode: Limited dataset to {len(all_metadata)} samples")

        # Split data
        train_meta, temp_meta = train_test_split(
            all_metadata, 
            test_size=(1 - split_ratios['train']), 
            random_state=random_seed,
            shuffle=True
        )
        
        val_size = split_ratios['val'] / (split_ratios['val'] + split_ratios['test'])
        val_meta, test_meta = train_test_split(
            temp_meta,
            test_size=(1 - val_size),
            random_state=random_seed,
            shuffle=True
        )

        num_train_samples = len(train_meta)
        num_val_samples = len(val_meta)
        num_test_samples = len(test_meta)

        logger.info(f"Dataset split: Train {num_train_samples}, Val {num_val_samples}, Test {num_test_samples}")

        # Create augmentation pipeline and preprocessing
        augmentation_pipeline = SegmentationAugmentation(aug_cfg)
        preprocess_fn = _get_segmentation_preprocess_fn(model_cfg.get('backbone'))
        
        pc_prep_cfg = data_cfg.get('modalities_preprocessing', {}).get('point_cloud', {})
        num_points_target = pc_prep_cfg.get('num_points', 4096)

        def create_dataset(metadata_subset, is_training):
            if not metadata_subset:
                return None

            # Define the shapes and types of the NumPy arrays yielded by the generator
            output_signature = {
                "rgb_input": tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                "depth_input": tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                "pc_input": tf.TensorSpec(shape=(num_points_target, 3), dtype=tf.float32),
                "mask": tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            }

            # Create the dataset from the pure Python generator
            dataset = tf.data.Dataset.from_generator(
                lambda: data_generator(metadata_subset, data_cfg),
                output_signature=output_signature
            )

            # This map function contains ONLY pure TensorFlow operations
            def tf_map_fn(sample):
                rgb = tf.image.resize(sample['rgb_input'], target_size)
                depth = tf.image.resize(sample['depth_input'], target_size)
                mask = tf.image.resize(sample['mask'], target_size, method='nearest')

                # Convert depth to 3-channel for the model backbone
                depth_3_channel = tf.concat([depth, depth, depth], axis=-1)
                
                inputs = {
                    "rgb_input": rgb,
                    "depth_input": depth_3_channel,
                    "pc_input": sample['pc_input']
                }

                # Apply augmentations if this is the training set
                if is_training and aug_cfg.get('enabled', False):
                    inputs, mask = augmentation_pipeline((inputs, mask))
                
                # Apply backbone-specific preprocessing
                if preprocess_fn:
                    inputs['rgb_input'] = preprocess_fn(inputs['rgb_input'])
                    inputs['depth_input'] = preprocess_fn(inputs['depth_input'])

                # Final squeeze of the mask for the loss function
                return inputs, tf.squeeze(mask, axis=-1)

            if is_training:
                dataset = dataset.shuffle(buffer_size=min(len(metadata_subset), 1024))
            
            dataset = dataset.map(tf_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset

        train_dataset = create_dataset(train_meta, is_training=True)
        val_dataset = create_dataset(val_meta, is_training=False)
        test_dataset = create_dataset(test_meta, is_training=False)

        return train_dataset, val_dataset, test_dataset, num_train_samples, num_val_samples, num_test_samples, num_classes

    except KeyError as e:
        logger.error(f"Configuration key error: {e}. Please check your segmentation config.yaml.")
        return None, None, None, 0, 0, 0, 0
    except Exception as e:
        logger.error(f"An unexpected error occurred in load_segmentation_data: {e}", exc_info=True)
        return None, None, None, 0, 0, 0, 0