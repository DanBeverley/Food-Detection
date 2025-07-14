import os
import yaml 
import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Optional, List, Any, Callable
from sklearn.model_selection import train_test_split
import pathlib
import json
import traceback 
import sys
import math 
 
from tensorflow.keras import layers 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_SEG_PREPROCESS_FN_CACHE = {}

def _get_project_root() -> pathlib.Path:
    """Find the project root directory."""
    return pathlib.Path(__file__).resolve().parent.parent.parent

def _get_segmentation_preprocess_fn(architecture: Optional[str]):
    """Dynamically imports and returns the correct preprocess_input function for segmentation model backbones."""
    global _SEG_PREPROCESS_FN_CACHE
    if not architecture or architecture.lower() == 'none' or architecture.lower() == 'unet': 
        logger.info(f"No specific backbone ('{architecture}') requiring Keras preprocess_input. Using generic scaling (image/127.5 - 1.0).")
        return lambda x: (x / 127.5) - 1.0 

    if architecture in _SEG_PREPROCESS_FN_CACHE:
        return _SEG_PREPROCESS_FN_CACHE[architecture]

    preprocess_input_fn = None
    try:
        if architecture.startswith("EfficientNetV2"):
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as pi
            preprocess_input_fn = pi
        elif architecture.startswith("EfficientNet"):
            from tensorflow.keras.applications.efficientnet import preprocess_input as pi
            preprocess_input_fn = pi
        elif architecture.startswith("ResNet"):
            module_name = f"tensorflow.keras.applications.{architecture.lower().split('v')[0]}" # e.g., resnet, resnet50
            base_module = __import__(module_name, fromlist=['preprocess_input'])
            preprocess_input_fn = base_module.preprocess_input
        elif architecture.startswith("MobileNetV2"):
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pi
            preprocess_input_fn = pi
        elif architecture.startswith("MobileNet"):
            from tensorflow.keras.applications.mobilenet import preprocess_input as pi
            preprocess_input_fn = pi
        else:
            logger.warning(f"Unsupported backbone '{architecture}' for specific preprocess_input. Using generic scaling (image/127.5 - 1.0).")
            preprocess_input_fn = lambda x: (x / 127.5) - 1.0 
    except ImportError:
        logger.error(f"Could not import preprocess_input for backbone {architecture}. Using generic scaling.", exc_info=True)
        preprocess_input_fn = lambda x: (x / 127.5) - 1.0 
    
    _SEG_PREPROCESS_FN_CACHE[architecture] = preprocess_input_fn
    return preprocess_input_fn

# Removed _load_and_preprocess_point_cloud_py_seg function as point clouds are now preprocessed offline





def segmentation_data_generator(metadata_list, data_cfg, paths_cfg):
    """Pure Python generator that loads and yields single samples as NumPy arrays."""
    project_root = _get_project_root()
    metadata_dir = project_root / paths_cfg['metadata_dir']
    
    use_depth = data_cfg.get('use_depth_map', False)
    depth_map_dir_name = data_cfg.get('depth_map_dir_name', 'depth')
    
    use_pc = data_cfg.get('use_point_cloud', False)
    pc_prep_cfg = data_cfg.get('modalities_preprocessing', {}).get('point_cloud', {})
    num_points_target = pc_prep_cfg.get('num_points', 4096)

    while True:
        np.random.shuffle(metadata_list)
        
        for item_data in metadata_list:
            try:
                rgb_rel_path = item_data.get('image_path')
                mask_rel_path = item_data.get('mask_path')
                if not rgb_rel_path or not mask_rel_path:
                    continue

                full_rgb_path = str(metadata_dir / rgb_rel_path)
                full_mask_path = str(metadata_dir / mask_rel_path)

                if not os.path.exists(full_rgb_path) or not os.path.exists(full_mask_path):
                    continue

                # Load RGB
                image_string = tf.io.read_file(full_rgb_path)
                image_decoded = tf.image.decode_image(image_string, channels=3, expand_animations=False).numpy()

                # Load Mask
                mask_string = tf.io.read_file(full_mask_path)
                mask_decoded = tf.image.decode_image(mask_string, channels=1, expand_animations=False).numpy()
                
                # Load Depth - derive path from RGB path
                depth_decoded = np.zeros_like(mask_decoded, dtype=np.uint8)
                if use_depth:
                    rgb_parent_dir = pathlib.Path(full_rgb_path).parent
                    depth_img_name = pathlib.Path(full_rgb_path).name
                    potential_depth_path = rgb_parent_dir / depth_map_dir_name / depth_img_name
                    
                    depth_path_str = ""
                    if potential_depth_path.exists():
                        depth_path_str = str(potential_depth_path)
                    else:
                        for ext in ['.png', '.jpg', '.jpeg', '.tif']:
                            potential_depth_path_ext = potential_depth_path.with_suffix(ext)
                            if potential_depth_path_ext.exists():
                                depth_path_str = str(potential_depth_path_ext)
                                break

                    if depth_path_str:
                        depth_string = tf.io.read_file(depth_path_str)
                        depth_decoded = tf.image.decode_image(depth_string, channels=1).numpy()
                
                # Load Point Cloud - from preprocessed .npy file
                pc_data = np.zeros((num_points_target, 3), dtype=np.float32)
                if use_pc:
                    pc_rel_path = item_data.get('point_cloud_path')
                    if pc_rel_path:
                        try:
                            full_pc_path = metadata_dir / pc_rel_path
                            if full_pc_path.exists() and full_pc_path.suffix == '.npy':
                                pc_data = np.load(str(full_pc_path)).astype(np.float32)
                                # Ensure correct shape
                                if pc_data.shape != (num_points_target, 3):
                                    logger.warning(f"Point cloud shape mismatch for {full_pc_path}: expected ({num_points_target}, 3), got {pc_data.shape}")
                                    pc_data = np.zeros((num_points_target, 3), dtype=np.float32)
                        except Exception as e:
                            logger.warning(f"Error loading preprocessed point cloud {pc_rel_path}: {e}")
                            pc_data = np.zeros((num_points_target, 3), dtype=np.float32)
                
                yield {
                    "rgb_input": image_decoded.astype(np.float32),
                    "depth_input": depth_decoded.astype(np.float32),
                    "pc_input": pc_data,
                    "mask": mask_decoded.astype(np.float32)
                }

            except Exception as e:
                logger.warning(f"Skipping sample {item_data.get('image_path')} due to generator error: {e}")
                continue

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
        if self.aug_config.get("contrast_range", [1.0, 1.0]) != [1.0, 1.0]:
            contrast_factor = self.aug_config.get("contrast_factor", 0.2)
            self.color_layers.append(layers.RandomContrast(factor=contrast_factor))

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


def load_segmentation_data(config: Dict[str, Any]) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset], int, int, int, int]:
    try:
        data_cfg = config['data']
        paths_cfg = config['paths']
        model_cfg = config['model'] 
        training_cfg = config.get('training', {})
        aug_cfg = data_cfg.get('augmentation', {})

        target_size_py = tuple(data_cfg.get('image_size', (256, 256))) 
        target_size_tensor = tf.constant(target_size_py, dtype=tf.int32)   
        batch_size = data_cfg['batch_size']
        num_classes = data_cfg['num_classes']
        num_classes_tensor = tf.constant(num_classes, dtype=tf.int32)
        split_ratios = data_cfg['split_ratios']
        random_seed = data_cfg.get('random_seed', 42)

        # Debug settings from config
        is_debug_mode_active = training_cfg.get('runtime_is_debug_mode', False)
        debug_max_samples = data_cfg.get('debug_max_samples', None)

        use_depth = data_cfg.get('use_depth_map', False)
        depth_map_dir_name = data_cfg.get('depth_map_dir_name', 'depth')
        depth_prep_cfg = data_cfg.get('modalities_preprocessing', {}).get('depth_map', {})

        use_pc = data_cfg.get('use_point_cloud', False)
        pc_root_dir = data_cfg.get('point_cloud_root_dir', '')
        pc_sampling_rate_dir = data_cfg.get('point_cloud_sampling_rate_dir', '')
        pc_suffix = data_cfg.get('point_cloud_suffix', '')
        pc_prep_cfg = data_cfg.get('modalities_preprocessing', {}).get('point_cloud', {})

        project_root = _get_project_root()
        metadata_dir = project_root / paths_cfg['metadata_dir']
        metadata_file = metadata_dir / paths_cfg['metadata_filename']

        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return None, None, None, 0, 0, 0, 0

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Directly use the loaded list if it's not empty
        if not isinstance(metadata, list) or not metadata:
            logger.error(f"Metadata file {metadata_file} does not contain a valid list of items or is empty.")
            return None, None, None, 0, 0, 0, 0

        # Debug Sampling Logic
        if is_debug_mode_active and debug_max_samples is not None and isinstance(debug_max_samples, int):
            if 0 < debug_max_samples < len(metadata):
                logger.info(f"SEGMENTATION DEBUG MODE: Limiting to {debug_max_samples} samples out of {len(metadata)} total.")
                np.random.seed(random_seed) 
                np.random.shuffle(metadata)
                metadata = metadata[:debug_max_samples]
            else:
                logger.warning(f"SEGMENTATION DEBUG MODE: debug_max_samples ({debug_max_samples}) is invalid or not smaller than total samples ({len(metadata)}). Using full dataset for debug run.")
        elif is_debug_mode_active:
            logger.info("SEGMENTATION DEBUG MODE: runtime_is_debug_mode is True, but debug_max_samples not set or invalid in data_config. Using full dataset for debug run.")

        train_meta, temp_meta = train_test_split(metadata, test_size=1-split_ratios['train'], random_state=random_seed)
        val_prop_in_remainder = split_ratios['val'] / (split_ratios['val'] + split_ratios['test'])
        val_meta, test_meta = train_test_split(temp_meta, train_size=val_prop_in_remainder, random_state=random_seed)

        num_train_samples = len(train_meta)
        num_val_samples = len(val_meta)
        num_test_samples = len(test_meta)

        logger.info(f"Dataset split: Train {num_train_samples}, Val {num_val_samples}, Test {num_test_samples}")

        if num_train_samples == 0:
            logger.error("No training samples after splitting. Check dataset size, debug settings, and split ratios.")
            return None, None, None, num_train_samples, num_val_samples, num_test_samples, num_classes

        augmentation_pipeline = SegmentationAugmentation(aug_cfg)
        preprocess_fn = _get_segmentation_preprocess_fn(model_cfg.get('backbone', 'efficientnetb0'))
        
        num_points_target = pc_prep_cfg.get('num_points', 4096)

        def create_dataset(metadata_subset, is_training):
            if not metadata_subset:
                return None

            output_signature = {
                "rgb_input": tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                "depth_input": tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                "pc_input": tf.TensorSpec(shape=(num_points_target, 3), dtype=tf.float32),
                "mask": tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            }

            dataset = tf.data.Dataset.from_generator(
                lambda: segmentation_data_generator(metadata_subset, data_cfg, paths_cfg),
                output_signature=output_signature
            )

            def map_fn(sample):
                rgb = tf.image.resize(sample['rgb_input'], target_size_py)
                depth = tf.image.resize(sample['depth_input'], target_size_py)
                depth_3_channel = tf.concat([depth, depth, depth], axis=-1)
                mask = tf.image.resize(sample['mask'], target_size_py, method='nearest')
                
                inputs = {
                    "rgb_input": rgb,
                    "depth_input": depth_3_channel,
                    "pc_input": sample['pc_input']
                }

                if is_training and augmentation_pipeline is not None:
                    inputs, final_mask_with_channel = augmentation_pipeline((inputs, mask))
                else:
                    final_mask_with_channel = mask
                
                if preprocess_fn:
                    inputs['rgb_input'] = preprocess_fn(inputs['rgb_input'])
                    inputs['depth_input'] = preprocess_fn(inputs['depth_input'])

                final_mask = tf.squeeze(final_mask_with_channel, axis=-1)
                return inputs, final_mask

            if is_training:
                dataset = dataset.shuffle(buffer_size=min(len(metadata_subset), 1000))
            
            dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
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