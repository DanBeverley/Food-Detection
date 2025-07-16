import os
import yaml 
import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Optional, List, Any
from sklearn.model_selection import train_test_split
import pathlib
import json
from PIL import Image

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

def data_generator(metadata_list: List[Dict], config: Dict, is_training: bool):
    """
    A pure Python generator that does ALL preprocessing and yields final NumPy arrays.
    """
    data_cfg = config['data']
    model_cfg = config.get('model', {})
    aug_cfg = data_cfg.get('augmentation', {})
    target_size = tuple(data_cfg.get('image_size', (256, 256)))
    
    do_flip = is_training and aug_cfg.get('enabled', False) and aug_cfg.get("horizontal_flip", False)
    
    preprocess_fn = _get_segmentation_preprocess_fn(model_cfg.get('backbone'))
    
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

                rgb_np = np.array(Image.open(full_rgb_path).convert('RGB').resize(target_size), dtype=np.float32)
                mask_np = np.array(Image.open(full_mask_path).convert('L').resize(target_size, Image.NEAREST), dtype=np.float32)
                
                depth_np = np.zeros(target_size, dtype=np.float32)
                full_depth_path = item_data.get('depth_map_path')
                if full_depth_path and os.path.exists(full_depth_path):
                    depth_np = np.array(Image.open(full_depth_path).convert('L').resize(target_size), dtype=np.float32)
                
                pc_np = np.zeros((num_points_target, 3), dtype=np.float32)
                full_pc_path = item_data.get('point_cloud_path')
                if full_pc_path and os.path.exists(full_pc_path):
                    pc_np = np.load(full_pc_path)

                if do_flip and np.random.rand() > 0.5:
                    rgb_np = np.fliplr(rgb_np)
                    depth_np = np.fliplr(depth_np)
                    mask_np = np.fliplr(mask_np)

                depth_3_channel_np = np.stack([depth_np, depth_np, depth_np], axis=-1)
                
                if preprocess_fn:
                    rgb_np = preprocess_fn(rgb_np)
                    depth_3_channel_np = preprocess_fn(depth_3_channel_np)
                
                yield (
                    {
                        "rgb_input": rgb_np,
                        "depth_input": depth_3_channel_np,
                        "pc_input": pc_np
                    },
                    mask_np
                )

            except Exception as e:
                logger.warning(f"Generator skipping sample {item_data.get('image_path')}: {e}")
                continue

def load_segmentation_data(config: Dict[str, Any]) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset], int, int, int, int]:
    try:
        data_cfg = config['data']
        paths_cfg = config['paths']
        batch_size = data_cfg['batch_size']
        num_classes = data_cfg['num_classes']
        split_ratios = data_cfg['split_ratios']
        random_seed = data_cfg.get('random_seed', 42)
        target_size = tuple(data_cfg.get('image_size', (256, 256)))

        project_root = _get_project_root()
        metadata_path = project_root / paths_cfg['metadata_dir'] / paths_cfg['metadata_filename']
        
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)

        train_meta, temp_meta = train_test_split(all_metadata, test_size=(1-split_ratios['train']), random_state=random_seed)
        val_prop_in_remainder = split_ratios['val'] / (split_ratios['val'] + split_ratios['test'])
        val_meta, test_meta = train_test_split(temp_meta, test_size=(1-val_prop_in_remainder), random_state=random_seed)

        num_train_samples = len(train_meta)
        num_val_samples = len(val_meta)
        num_test_samples = len(test_meta)
        logger.info(f"Dataset split: Train {num_train_samples}, Val {num_val_samples}, Test {num_test_samples}")

        pc_prep_cfg = data_cfg.get('modalities_preprocessing', {}).get('point_cloud', {})
        num_points_target = pc_prep_cfg.get('num_points', 4096)

        def create_dataset(metadata_subset, is_training):
            if not metadata_subset:
                return None

            output_signature = (
                {
                    "rgb_input": tf.TensorSpec(shape=(*target_size, 3), dtype=tf.float32),
                    "depth_input": tf.TensorSpec(shape=(*target_size, 3), dtype=tf.float32),
                    "pc_input": tf.TensorSpec(shape=(num_points_target, 3), dtype=tf.float32)
                },
                tf.TensorSpec(shape=target_size, dtype=tf.float32)
            )

            dataset = tf.data.Dataset.from_generator(
                lambda: data_generator(metadata_subset, config, is_training),
                output_signature=output_signature
            )

            if is_training:
                dataset = dataset.shuffle(buffer_size=min(len(metadata_subset), 1024))
            
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset

        train_dataset = create_dataset(train_meta, is_training=True)
        val_dataset = create_dataset(val_meta, is_training=False)
        test_dataset = create_dataset(test_meta, is_training=False)

        return train_dataset, val_dataset, test_dataset, num_train_samples, num_val_samples, num_test_samples, num_classes

    except Exception as e:
        logger.error(f"An unexpected error occurred in load_segmentation_data: {e}", exc_info=True)
        return None, None, None, 0, 0, 0, 0