import os
import yaml 
import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Optional, List, Any
from sklearn.model_selection import train_test_split
import pathlib
import json
import traceback 

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
            base_module = __import__(f"tensorflow.keras.applications.{architecture.lower()}", fromlist=['preprocess_input'])
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


@tf.function
def apply_augmentations(image: tf.Tensor, mask: tf.Tensor, aug_config: Dict) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies augmentations consistently to image and mask. TF Addons might be needed for rotate/transform."""
    if not aug_config.get('enabled', False):
        return image, mask

    if aug_config.get('horizontal_flip', False) and tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    brightness_max_delta = aug_config.get('brightness_max_delta', 0.0) 
    if brightness_max_delta > 0:
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        image = tf.clip_by_value(image, 0.0, 255.0) 

    return image, mask


def load_and_preprocess_segmentation(
    image_path_tensor: tf.Tensor, 
    mask_path_tensor: tf.Tensor, 
    target_size: Tuple[int, int], 
    num_classes: int, 
    preprocess_input_fn: Any,
    augment: bool = False,
    aug_config: Optional[Dict] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Loads, decodes, resizes, and preprocesses an image and its mask."""
    try:
        image_path = image_path_tensor.numpy().decode('utf-8')
        mask_path = mask_path_tensor.numpy().decode('utf-8')

        img_str = tf.io.read_file(image_path)
        image = tf.image.decode_image(img_str, channels=3, expand_animations=False)
        image = tf.image.resize(image, target_size)
        image.set_shape([*target_size, 3])
        image_for_aug = tf.cast(image, tf.float32) 

        mask_str = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask_str, channels=1, expand_animations=False) 
        mask = tf.image.resize(mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask.set_shape([*target_size, 1])
        mask = tf.cast(mask, tf.float32)
        if tf.reduce_max(mask) > (num_classes -1): 
             mask = mask / 255.0 

        if augment and aug_config:
            image_for_aug, mask = apply_augmentations(image_for_aug, mask, aug_config)

        image_preprocessed = preprocess_input_fn(image_for_aug) 
        return image_preprocessed, mask

    except Exception as e:
        img_p = image_path_tensor.numpy().decode('utf-8') if hasattr(image_path_tensor, 'numpy') else 'unknown_img_path'
        msk_p = mask_path_tensor.numpy().decode('utf-8') if hasattr(mask_path_tensor, 'numpy') else 'unknown_mask_path'
        logger.error(f"Error in load_and_preprocess_segmentation for image '{img_p}' or mask '{msk_p}': {e}", exc_info=True)
        print(f"--- TRACEBACK FROM load_and_preprocess_segmentation ({img_p}) ---")
        print(traceback.format_exc())
        print("-----------------------------------------------------------------")
        raise 


def load_segmentation_data(config: Dict[str, Any]) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset]]:
    """
    Loads segmentation data (image-mask pairs) using metadata.json.
    Implements instance-aware train/val/test splitting.
    Args:
        config: Dictionary from models/segmentation/config.yaml.
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    project_root = _get_project_root()
    try:
        data_cfg = config['data']
        model_cfg = config['model']
        metadata_path_str = data_cfg['metadata_path']
        image_size = tuple(data_cfg['image_size'])
        batch_size = data_cfg['batch_size']
        val_split_ratio = data_cfg.get('validation_split_ratio', 0.0)
        test_split_ratio = data_cfg.get('test_split_ratio', 0.0)
        random_seed = data_cfg.get('random_seed', 42)
        num_classes_data = data_cfg.get('num_classes', 2) 
        num_classes_model = model_cfg.get('num_classes', 2) 
        if num_classes_data != num_classes_model:
            logger.warning(f"num_classes mismatch: data.num_classes={num_classes_data}, model.num_classes={num_classes_model}. Using model.num_classes={num_classes_model}.")
        num_classes = num_classes_model
        backbone = model_cfg.get('backbone', 'None')
        aug_settings = data_cfg.get('augmentation', {'enabled': False})
    except KeyError as e:
        raise ValueError(f"Configuration error in segmentation config: missing key {e}")

    metadata_file = project_root / metadata_path_str
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Segmentation metadata JSON file not found: {metadata_file}")

    if not (0 <= val_split_ratio < 1 and 0 <= test_split_ratio < 1 and (val_split_ratio + test_split_ratio) < 1):
        raise ValueError("Invalid split ratios. Must be [0, 1) and val_split + test_split < 1.")

    logger.info(f"Loading segmentation data from metadata: {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata_list = json.load(f)
    
    all_pairs_data = []
    for item in metadata_list:
        img_path = item.get('image_path')
        mask_path = item.get('mask_path')
        class_name = item.get('class_name', 'unknown_class') 
        instance_name = item.get('instance_name', 'unknown_instance') 

        if not (img_path and mask_path):
            logger.warning(f"Skipping metadata item with missing image or mask path: {item}")
            continue
        if not (pathlib.Path(img_path).is_file() and pathlib.Path(mask_path).is_file()):
            logger.warning(f"Skipping pair due to missing file: img='{img_path}', mask='{mask_path}'")
            continue
        all_pairs_data.append({
            'image_path': str(img_path),
            'mask_path': str(mask_path),
            'instance_id': f"{class_name}_{instance_name}"
        })

    if not all_pairs_data:
        raise ValueError(f"No valid image-mask pairs loaded from metadata: {metadata_file}. Check paths and file existence.")
    logger.info(f"Found {len(all_pairs_data)} total valid image-mask pairs from metadata.")

    train_items, val_items, test_items = [], [], []
    if val_split_ratio == 0.0 and test_split_ratio == 0.0:
        logger.info("No validation or test split. Using all data for training.")
        train_items = all_pairs_data
    else:
        unique_instance_ids = sorted(list(set(d['instance_id'] for d in all_pairs_data)))
        if len(unique_instance_ids) < 2:
             logger.warning(f"Only {len(unique_instance_ids)} unique instance(s). Split might not be diverse. Consider dataset structure.")
        
        remaining_instances = unique_instance_ids
        test_instance_ids = []
        if test_split_ratio > 0 and len(unique_instance_ids) > 0:
            if len(unique_instance_ids) == 1 and test_split_ratio > 0:
                 logger.warning("Only 1 unique instance, cannot create a test set via instance split. Test set will be empty.")
            else:
                remaining_instances, test_instance_ids = train_test_split(
                    unique_instance_ids, test_size=test_split_ratio, random_state=random_seed, shuffle=True)
        
        train_instance_ids = remaining_instances
        val_instance_ids = []
        if val_split_ratio > 0 and len(remaining_instances) > 0:
            effective_val_ratio = val_split_ratio / (1.0 - test_split_ratio) 
            if effective_val_ratio >= 1.0 and len(remaining_instances) > 1: 
                effective_val_ratio = 0.5 if len(remaining_instances) > 1 else 0.0 
            elif len(remaining_instances) == 1 and effective_val_ratio > 0:
                 logger.warning("Only 1 unique instance remaining after test split, cannot create validation set. Val set will be empty.")
                 effective_val_ratio = 0.0

            if effective_val_ratio > 0 and effective_val_ratio < 1.0:
                 train_instance_ids, val_instance_ids = train_test_split(
                    remaining_instances, test_size=effective_val_ratio, random_state=random_seed, shuffle=True)
            elif effective_val_ratio == 0.0:
                 train_instance_ids = remaining_instances 
            else: 
                 logger.warning(f"Effective validation ratio {effective_val_ratio} is too high. Assigning remaining to train.")
                 train_instance_ids = remaining_instances

        for item in all_pairs_data:
            if item['instance_id'] in train_instance_ids:
                train_items.append(item)
            elif item['instance_id'] in val_instance_ids:
                val_items.append(item)
            elif item['instance_id'] in test_instance_ids:
                test_items.append(item)

    logger.info(f"Data split: Train={len(train_items)}, Validation={len(val_items)}, Test={len(test_items)} items.")
    if not train_items and (val_split_ratio > 0 or test_split_ratio > 0):
        logger.warning("Training set is empty after instance-aware split. This can happen with small datasets or few instances.")
    elif not train_items:
         raise ValueError("Training set is empty. Check dataset and metadata.")

    preprocess_input_for_backbone = _get_segmentation_preprocess_fn(backbone)
    AUTOTUNE = tf.data.AUTOTUNE

    def create_dataset(items: List[Dict], augment: bool) -> Optional[tf.data.Dataset]:
        if not items:
            return None
        img_paths = [item['image_path'] for item in items]
        msk_paths = [item['mask_path'] for item in items]
        
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
        dataset = dataset.map(lambda img_p, msk_p: 
            tf.py_function(func=load_and_preprocess_segmentation, 
                           inp=[img_p, msk_p, image_size, num_classes, preprocess_input_for_backbone, augment, aug_settings if augment else None],
                           Tout=(tf.float32, tf.float32)),
            num_parallel_calls=AUTOTUNE)
        
        def _set_shape(image, mask):
            image.set_shape([*image_size, 3])
            mask.set_shape([*image_size, 1]) 
            return image, mask
        dataset = dataset.map(_set_shape, num_parallel_calls=AUTOTUNE)

        if augment:
            dataset = dataset.shuffle(buffer_size=max(100, len(items)))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    train_dataset = create_dataset(train_items, augment=True)
    val_dataset = create_dataset(val_items, augment=False)
    test_dataset = create_dataset(test_items, augment=False)

    logger.info("Segmentation datasets created.")
    return train_dataset, val_dataset, test_dataset