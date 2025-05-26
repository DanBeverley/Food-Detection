import json
import os
import tensorflow as tf
from typing import Tuple, List, Dict, Optional
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import pathlib
from tqdm import tqdm
import random # Import for random sampling
from collections import Counter
import traceback
from pathlib import Path

# Dynamic import for preprocess_input
_PREPROCESS_FN_CACHE = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent.parent

def compute_class_weights(y_true: List[int], num_classes: int) -> Optional[Dict[int, float]]:
    if not y_true:
        logger.warning("Cannot compute class weights: y_true is empty.")
        return None
    counts = Counter(y_true)
    total_samples = len(y_true)
    class_weights = {}
    for class_idx in range(num_classes):
        if counts[class_idx] > 0:
            weight = total_samples / (num_classes * counts[class_idx])
            class_weights[class_idx] = weight
    if not class_weights:
        logger.warning("Class weights dictionary is empty after computation. Check input y_true and num_classes.")
        return None
    return class_weights

def _get_preprocess_fn(architecture: str):
    global _PREPROCESS_FN_CACHE
    if architecture in _PREPROCESS_FN_CACHE:
        return _PREPROCESS_FN_CACHE[architecture]
    preprocess_input_fn = None
    if architecture == "MobileNet":
        from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_input_fn
    elif architecture == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_fn
    elif architecture == "MobileNetV3Small" or architecture == "MobileNetV3Large":
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_input_fn
    elif architecture.startswith("EfficientNetV2"):
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_input_fn
    elif architecture.startswith("EfficientNet"):
        from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_fn
    elif architecture.startswith("ResNet") and "V2" not in architecture:
        from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_fn
    elif architecture.startswith("ResNet") and "V2" in architecture:
        from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_fn
    elif architecture.startswith("ConvNeXt"):
        from tensorflow.keras.applications.convnext import preprocess_input as preprocess_input_fn
    else:
        logger.warning(f"Preprocessing function not explicitly defined or imported for {architecture}. Using generic scaling (image / 255.0). This may be suboptimal.")
        preprocess_input_fn = lambda x: x / 255.0
    _PREPROCESS_FN_CACHE[architecture] = preprocess_input_fn
    return preprocess_input_fn

def _build_augmentation_pipeline(data_conf: Dict) -> Optional[tf.keras.Sequential]: # Corrected to take data_conf
    aug_config = data_conf.get('augmentation', {})
    image_s = tuple(data_conf.get('image_size', [224, 224]))

    if not aug_config.get('enabled', False):
        return None
    
    pipeline = tf.keras.Sequential(name="augmentation_pipeline")
    pipeline.add(tf.keras.layers.Input(shape=(*image_s, 3))) # Use image_s
    
    # Basic augmentations
    if aug_config.get('horizontal_flip', False):
        pipeline.add(tf.keras.layers.RandomFlip("horizontal"))
    if aug_config.get('rotation_range', 0) > 0:
        factor = aug_config['rotation_range'] / 360.0
        pipeline.add(tf.keras.layers.RandomRotation(factor))
    if aug_config.get('zoom_range', 0) > 0:
        zoom_factor = aug_config['zoom_range']
        pipeline.add(tf.keras.layers.RandomZoom(height_factor=(-zoom_factor, zoom_factor), width_factor=(-zoom_factor, zoom_factor)))
    if 'brightness_range' in aug_config and aug_config['brightness_range']:
        br_range = aug_config['brightness_range']
        if isinstance(br_range, list) and len(br_range) == 2:
            factor = max(abs(1.0 - br_range[0]), abs(br_range[1] - 1.0))
        else:
            factor = br_range 
        pipeline.add(tf.keras.layers.RandomBrightness(factor=factor))
    if aug_config.get("width_shift_range", 0) > 0:
        pipeline.add(tf.keras.layers.RandomTranslation(height_factor=0, width_factor=aug_config['width_shift_range']))
    if aug_config.get("height_shift_range", 0) > 0:
        pipeline.add(tf.keras.layers.RandomTranslation(height_factor=aug_config['height_shift_range'], width_factor=0))
    
    # Enhanced augmentations for overfitting prevention
    if 'contrast_range' in aug_config and aug_config['contrast_range']:
        contrast_range = aug_config['contrast_range']
        if isinstance(contrast_range, list) and len(contrast_range) == 2:
            factor = max(abs(1.0 - contrast_range[0]), abs(contrast_range[1] - 1.0))
            pipeline.add(tf.keras.layers.RandomContrast(factor=factor))
    
    # Random erasing (cutout) for regularization
    random_erasing_config = aug_config.get('random_erasing', {})
    if random_erasing_config.get('enabled', False):
        pipeline.add(RandomErasing(
            probability=random_erasing_config.get('probability', 0.25),
            area_ratio_range=tuple(random_erasing_config.get('area_ratio_range', [0.02, 0.33])),
            aspect_ratio_range=tuple(random_erasing_config.get('aspect_ratio_range', [0.3, 3.3]))
        ))
        logger.info("Random erasing layer added to augmentation pipeline")
    
    # Gaussian noise for robustness
    if aug_config.get('gaussian_noise_std', 0) > 0:
        pipeline.add(GaussianNoise(stddev=aug_config['gaussian_noise_std']))
        logger.info(f"Gaussian noise layer added with std={aug_config['gaussian_noise_std']}")
    
    pipeline.add(tf.keras.layers.Resizing(image_s[0], image_s[1])) # Use image_s
    return pipeline

@tf.function
def mixup(batch_images: tf.Tensor, batch_labels: tf.Tensor, alpha: float, num_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(batch_images)[0]
    # Ensure labels are one-hot.
    if len(tf.shape(batch_labels)) == 1: # Sparse labels (batch_size,)
        labels_one_hot = tf.one_hot(tf.cast(batch_labels, dtype=tf.int32), depth=num_classes)
    else: # Already one-hot (batch_size, num_classes)
        labels_one_hot = batch_labels

    if alpha > 0.0:
        beta_dist = tf.compat.v1.distributions.Beta(alpha, alpha)
        lambda_val = beta_dist.sample(batch_size) # Sample per image in batch
        lambda_img = tf.reshape(lambda_val, [batch_size, 1, 1, 1])
        lambda_lbl = tf.reshape(lambda_val, [batch_size, 1])
    else: # No mixing if alpha is 0
        return batch_images, labels_one_hot

    shuffled_indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(batch_images, shuffled_indices)
    shuffled_labels_one_hot = tf.gather(labels_one_hot, shuffled_indices)

    mixed_images = lambda_img * batch_images + (1.0 - lambda_img) * shuffled_images
    mixed_labels = lambda_lbl * labels_one_hot + (1.0 - lambda_lbl) * shuffled_labels_one_hot
    return mixed_images, mixed_labels

@tf.function
def cutmix(batch_images: tf.Tensor, batch_labels: tf.Tensor, alpha: float, num_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(batch_images)[0]
    image_h = tf.shape(batch_images)[1]
    image_w = tf.shape(batch_images)[2]

    if len(tf.shape(batch_labels)) == 1:
        labels_one_hot = tf.one_hot(tf.cast(batch_labels, dtype=tf.int32), depth=num_classes)
    else:
        labels_one_hot = batch_labels

    if alpha > 0.0:
        beta_dist = tf.compat.v1.distributions.Beta(alpha, alpha)
        lambda_val = beta_dist.sample(1)[0] 
    else:
        return batch_images, labels_one_hot
    
    shuffled_indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(batch_images, shuffled_indices)
    shuffled_labels_one_hot = tf.gather(labels_one_hot, shuffled_indices)

    cut_ratio = tf.sqrt(1.0 - lambda_val) 
    cut_h = tf.cast(cut_ratio * tf.cast(image_h, dtype=tf.float32), dtype=tf.int32)
    cut_w = tf.cast(cut_ratio * tf.cast(image_w, dtype=tf.float32), dtype=tf.int32)

    # Ensure cut_h and cut_w are at least 1 to avoid issues with tf.zeros
    cut_h = tf.maximum(cut_h, 1)
    cut_w = tf.maximum(cut_w, 1)
    
    # Ensure bbx1, bby1 calculations do not result in negative ranges for uniform
    range_w = image_w - cut_w
    range_h = image_h - cut_h

    bbx1 = tf.random.uniform([], minval=0, maxval=tf.maximum(1, range_w), dtype=tf.int32) if range_w > 0 else 0
    bby1 = tf.random.uniform([], minval=0, maxval=tf.maximum(1, range_h), dtype=tf.int32) if range_h > 0 else 0
    
    bbx2 = bbx1 + cut_w
    bby2 = bby1 + cut_h

    # Create mask
    # The mask needs to be of shape (height, width, 1) to broadcast correctly with (h, w, c) images
    mask_center_shape = [cut_h, cut_w, 1] # For broadcasting with channels
    padding = [[bby1, image_h - bby2], [bbx1, image_w - bbx2], [0, 0]] # Pad for channels dim too
    
    mask_center = tf.zeros(mask_center_shape, dtype=tf.float32)
    mask = tf.pad(mask_center, padding, "CONSTANT", constant_values=1.0) # Pad with 1s

    # Apply mask to images
    # Mask has 1s where original image should be, 0s where patch from shuffled image should be
    cutmix_images = batch_images * mask + shuffled_images * (1.0 - mask)
    mixed_labels = lambda_val * labels_one_hot + (1.0 - lambda_val) * shuffled_labels_one_hot
    return cutmix_images, mixed_labels

# Custom augmentation layers for overfitting prevention
class RandomErasing(tf.keras.layers.Layer):
    """Random Erasing augmentation layer for regularization."""
    
    def __init__(self, probability=0.25, area_ratio_range=(0.02, 0.33), aspect_ratio_range=(0.3, 3.3), **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.area_ratio_range = area_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        # Simple random erasing implementation
        def apply_erasing():
            shape = tf.shape(inputs)
            height = shape[0]
            width = shape[1]
            channels = shape[2]
            
            # Fixed erasing size for simplicity (10% of image area)
            erase_area = tf.cast(height * width, tf.float32) * 0.1
            erase_size = tf.cast(tf.sqrt(erase_area), tf.int32)
            erase_size = tf.minimum(erase_size, tf.minimum(height, width) // 2)
            erase_size = tf.maximum(erase_size, 1)
            
            # Random position
            max_y = tf.maximum(height - erase_size, 0)
            max_x = tf.maximum(width - erase_size, 0)
            y = tf.random.uniform([], 0, max_y + 1, dtype=tf.int32)
            x = tf.random.uniform([], 0, max_x + 1, dtype=tf.int32)
            
            # Create a mask
            mask = tf.ones_like(inputs)
            # Create zero patch
            zero_patch = tf.zeros([erase_size, erase_size, channels])
            
            # Apply the patch using slicing
            erased = tf.concat([
                inputs[:y],
                tf.concat([
                    inputs[y:y+erase_size, :x],
                    zero_patch,
                    inputs[y:y+erase_size, x+erase_size:]
                ], axis=1),
                inputs[y+erase_size:]
            ], axis=0)
            
            return erased
        
        # Random decision to apply erasing
        should_erase = tf.random.uniform([]) < self.probability
        return tf.cond(should_erase, apply_erasing, lambda: inputs)

class GaussianNoise(tf.keras.layers.Layer):
    """Gaussian noise layer for regularization."""
    
    def __init__(self, stddev=0.02, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        noise = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=self.stddev)
        return inputs + noise

def load_classification_data(
    config: Dict,
    max_samples_to_load: Optional[int] = None
) -> Tuple[
    Optional[tf.data.Dataset],
    Optional[tf.data.Dataset],
    Optional[tf.data.Dataset],
    int, # num_train_samples
    int, # num_val_samples
    int, # num_test_samples
    List[str], # class_names (ordered by index)
    int, # num_classes
]:
    project_root = _get_project_root()
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    model_config = config.get('model', {})

    metadata_path = project_root / data_config.get('paths', {}).get('metadata_file', 'data/classification/metadata.json')
    label_map_path = project_root / data_config.get('paths', {}).get('label_map_file', 'data/classification/label_map.json')
    base_image_dir = project_root 

    image_size = tuple(data_config.get('image_size', [224, 224]))
    batch_size = data_config.get('batch_size', 32)
    val_split = data_config.get('validation_split', 0.2)
    test_split = data_config.get('test_split', 0.1) 

    logger.info(f"Loading classification metadata from: {metadata_path}")
    logger.info(f"Using batch_size: {batch_size} from data configuration")
    
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return None, None, None, 0, 0, 0, [], 0
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata from {metadata_path}: {e}")
        return None, None, None, 0, 0, 0, [], 0

    logger.info(f"Loading label map from: {label_map_path}")
    if not label_map_path.exists():
        logger.error(f"Label map file not found: {label_map_path}")
        return None, None, None, 0, 0, 0, [], 0 
    try:
        with open(label_map_path, 'r') as f:
            index_to_label_map_str_keys = json.load(f)
            index_to_label_map = {int(k): v for k, v in index_to_label_map_str_keys.items()}
            num_classes = len(index_to_label_map)
            class_names = [index_to_label_map[i] for i in sorted(index_to_label_map.keys())]
            if not class_names or num_classes == 0:
                logger.error(f"Label map loaded from {label_map_path} is empty or invalid.")
                return None, None, None, 0, 0, 0, [], 0
            logger.info(f"Successfully loaded label map. Number of classes: {num_classes}")
    except Exception as e:
        logger.error(f"Error loading or processing label map from {label_map_path}: {e}")
        return None, None, None, 0, 0, 0, [], 0

    all_image_paths = []
    all_labels = [] 
    label_to_index_map = {v: k for k, v in index_to_label_map.items()}

    logger.info("Collecting image paths and labels from metadata...")
    # Handle both list format and dict format for metadata
    if isinstance(metadata, list):
        metadata_items = metadata
    else:
        metadata_items = metadata.get('images', [])
    
    for item in tqdm(metadata_items, desc="Processing metadata entries"):
        # Handle different field name formats
        relative_path = item.get('path') or item.get('image_path')
        label_name = item.get('label') or item.get('class_name')
        
        if not relative_path or label_name is None:
            logger.warning(f"Skipping item due to missing path or label: {item}")
            continue
        
        # Handle absolute paths from the dataset
        if relative_path.startswith('E:'):
            # Use the absolute path directly
            full_image_path = Path(relative_path)
        else:
            # Handle relative paths
            if relative_path.startswith(str(base_image_dir)):
                relative_path = str(Path(relative_path).relative_to(base_image_dir))
            full_image_path = base_image_dir / relative_path
        
        if not full_image_path.exists():
            logger.warning(f"Image file not found: {full_image_path}. Skipping.")
            continue

        if label_name not in label_to_index_map:
            logger.error(f"Label '{label_name}' from metadata not found in label_map {label_map_path}. Skipping image {full_image_path}.")
            continue
        
        all_image_paths.append(str(full_image_path))
        all_labels.append(label_to_index_map[label_name])

    if not all_image_paths:
        logger.error("No valid image paths found after processing metadata.")
        return None, None, None, 0, 0, 0, class_names, num_classes
    
    logger.info(f"Collected {len(all_image_paths)} total image paths with labels.")

    if max_samples_to_load is not None and max_samples_to_load > 0:
        logger.info(f"Debug mode: max_samples_to_load specified as {max_samples_to_load}.")
        if len(all_image_paths) > max_samples_to_load:
            logger.info(f"Shuffling and truncating dataset from {len(all_image_paths)} to {max_samples_to_load} samples.")
            combined = list(zip(all_image_paths, all_labels))
            random.shuffle(combined)
            all_image_paths_shuffled, all_labels_shuffled = zip(*combined)
            all_image_paths = list(all_image_paths_shuffled[:max_samples_to_load])
            all_labels = list(all_labels_shuffled[:max_samples_to_load])
            logger.info(f"Dataset truncated to {len(all_image_paths)} samples for debug mode.")
        else:
            logger.info(f"max_samples_to_load ({max_samples_to_load}) is >= total samples ({len(all_image_paths)}). Using all collected samples.")

    if not all_image_paths: 
        logger.error("No image paths remaining after potential debug mode truncation.")
        return None, None, None, 0, 0, 0, class_names, num_classes

    all_image_paths_np = np.array(all_image_paths)
    all_labels_np = np.array(all_labels)

    if val_split + test_split >= 1.0:
        logger.error(f"Validation split ({val_split}) + Test split ({test_split}) must be less than 1.0.")
        return None, None, None, 0, 0, 0, class_names, num_classes

    can_stratify_initial = True
    if num_classes > 1:
        label_counts = Counter(all_labels_np)
        if any(count < 2 for count in label_counts.values()):
            logger.warning("Cannot stratify initial split: at least one class has fewer than 2 samples. Proceeding without stratification.")
            can_stratify_initial = False
    else: 
        can_stratify_initial = False
        logger.info("Single class dataset or num_classes <= 1, proceeding without stratification.")

    if (1.0 - val_split - test_split) <= 0: # Ensure train split is positive
        logger.error(f"The combined validation ({val_split}) and test ({test_split}) splits leave no data for training.")
        return None, None, None, 0, 0, 0, class_names, num_classes

    if test_split > 0:
        if len(all_image_paths_np) < 2 : # Need at least 2 samples to split
            logger.warning(f"Too few samples ({len(all_image_paths_np)}) to create a test set. Test set will be empty.")
            train_val_paths, train_val_labels = all_image_paths_np, all_labels_np
            test_paths, test_labels = np.array([]), np.array([])
        else:
            stratify_param_tv = all_labels_np if can_stratify_initial else None
            train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
                all_image_paths_np, all_labels_np, 
                test_size=test_split, 
                random_state=data_config.get('random_seed', 42),
                stratify=stratify_param_tv
            )
    else:
        train_val_paths, train_val_labels = all_image_paths_np, all_labels_np
        test_paths, test_labels = np.array([]), np.array([]) 

    if val_split > 0 and len(train_val_paths) > 0:
        # Adjust val_split fraction relative to the remaining train_val_paths
        adjusted_val_split = val_split / (1.0 - test_split) if (1.0 - test_split) > 0 else val_split

        if len(train_val_paths) < 2:
            logger.warning(f"Too few samples in train_val_paths ({len(train_val_paths)}) for validation set. Validation set will be empty.")
            train_paths, train_labels = train_val_paths, train_val_labels
            val_paths, val_labels = np.array([]), np.array([])
        else:
            can_stratify_val = True
            if num_classes > 1:
                tv_label_counts = Counter(train_val_labels)
                if any(count < 2 for count in tv_label_counts.values()):
                    logger.warning("Cannot stratify train/validation split. Proceeding without stratification for this split.")
                    can_stratify_val = False
            else:
                can_stratify_val = False
            
            stratify_param_val = train_val_labels if can_stratify_val else None
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_val_paths, train_val_labels, 
                test_size=adjusted_val_split, 
                random_state=data_config.get('random_seed', 42), 
                stratify=stratify_param_val
            )
    elif len(train_val_paths) > 0:
        train_paths, train_labels = train_val_paths, train_val_labels
        val_paths, val_labels = np.array([]), np.array([])
    else: 
        train_paths, train_labels = np.array([]), np.array([])
        val_paths, val_labels = np.array([]), np.array([])

    num_train_samples = len(train_paths)
    num_val_samples = len(val_paths)
    num_test_samples = len(test_paths)

    logger.info(f"Dataset split sizes: Train={num_train_samples}, Validation={num_val_samples}, Test={num_test_samples}")

    if num_train_samples == 0:
        logger.warning("Training set is empty after splits. Cannot proceed with training.")
        # Still return class_names and num_classes as they are derived from label_map
        return None, None, None, 0, 0, num_test_samples, class_names, num_classes

    # Class weights computation is now done in train.py if needed, using train_labels

    architecture = model_config.get('architecture', 'MobileNetV2')
    preprocess_input_fn = _get_preprocess_fn(architecture)
    augmentation_pipeline = _build_augmentation_pipeline(data_config) # Pass data_config

    def load_and_preprocess_image(path: str, label: int) -> Tuple[tf.Tensor, int]:
        try:
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, image_size) 
            return img, label 
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}. Traceback: {traceback.format_exc()}")
            dummy_img = tf.zeros([*image_size, 3], dtype=tf.float32)
            dummy_label = tf.constant(0, dtype=tf.int32) 
            return dummy_img, dummy_label

    def configure_dataset(paths_np: np.ndarray, labels_np: np.ndarray, shuffle_ds: bool, augment_ds: bool, is_training_set_flag: bool) -> Optional[tf.data.Dataset]:
        if len(paths_np) == 0:
            return None
        try:
            dataset = tf.data.Dataset.from_tensor_slices((list(paths_np), list(labels_np)))
            dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            
            def map_fn_wrapper(image, label):
                image = tf.cast(image, tf.float32)
                if augmentation_pipeline is not None and augment_ds and is_training_set_flag:
                    # Add batch dimension for augmentation pipeline
                    image = tf.expand_dims(image, 0)
                    image = augmentation_pipeline(image, training=True) # Pass training=True
                    # Remove batch dimension
                    image = tf.squeeze(image, 0)
                if preprocess_input_fn:
                    image = preprocess_input_fn(image)
                return image, label

            dataset = dataset.map(map_fn_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

            if shuffle_ds:
                buffer_size = data_config.get('shuffle_buffer_size', 1000)
                dataset_size = len(paths_np)
                actual_buffer_size = min(buffer_size, dataset_size)
                if actual_buffer_size > 0 : # Ensure buffer_size is positive
                    dataset = dataset.shuffle(actual_buffer_size, seed=data_config.get('random_seed', None))
            
            dataset = dataset.batch(batch_size)
            
            # Convert integer labels to one-hot encoding
            def convert_to_one_hot(images, labels):
                labels_one_hot = tf.one_hot(labels, depth=num_classes)
                return images, labels_one_hot
            
            dataset = dataset.map(convert_to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

            if is_training_set_flag: # Apply MixUp/CutMix only to training set after batching
                # Read from both old and new config locations for backward compatibility
                mixup_alpha = training_config.get('mixup_alpha', 0.0)
                cutmix_alpha = training_config.get('cutmix_alpha', 0.0)
                
                # Check advanced_augmentation section for new config
                advanced_aug_config = data_config.get('advanced_augmentation', {})
                if advanced_aug_config.get('mixup', {}).get('enabled', False):
                    mixup_alpha = advanced_aug_config['mixup'].get('alpha', mixup_alpha)
                if advanced_aug_config.get('cutmix', {}).get('enabled', False):
                    cutmix_alpha = advanced_aug_config['cutmix'].get('alpha', cutmix_alpha)
                
                if mixup_alpha > 0.0:
                    logger.info(f"Applying MixUp with alpha={mixup_alpha} to the training dataset.")
                    dataset = dataset.map(
                        lambda images, lbls: mixup(images, lbls, mixup_alpha, num_classes),
                        num_parallel_calls=tf.data.AUTOTUNE
                    )
                if cutmix_alpha > 0.0:
                    logger.info(f"Applying CutMix with alpha={cutmix_alpha} to the training dataset.")
                    dataset = dataset.map(
                        lambda images, lbls: cutmix(images, lbls, cutmix_alpha, num_classes),
                        num_parallel_calls=tf.data.AUTOTUNE
                    )
            
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            return dataset
        except Exception as e:
            logger.error(f"Error configuring dataset: {e}. Traceback: {traceback.format_exc()}")
            return None

    augment_train = data_config.get('augmentation', {}).get('augment_training_data', True) and data_config.get('augmentation', {}).get('enabled', False)
    augment_val = data_config.get('augmentation', {}).get('augment_validation_data', False) and data_config.get('augmentation', {}).get('enabled', False)

    train_dataset = configure_dataset(train_paths, train_labels, shuffle_ds=True, augment_ds=augment_train, is_training_set_flag=True)
    val_dataset = configure_dataset(val_paths, val_labels, shuffle_ds=data_config.get('shuffle_validation_data', True), augment_ds=augment_val, is_training_set_flag=False)
    test_dataset = configure_dataset(test_paths, test_labels, shuffle_ds=False, augment_ds=False, is_training_set_flag=False)

    if train_dataset: logger.info(f"Training dataset: {tf.data.experimental.cardinality(train_dataset).numpy()} batches.")
    if val_dataset: logger.info(f"Validation dataset: {tf.data.experimental.cardinality(val_dataset).numpy()} batches.")
    if test_dataset: logger.info(f"Test dataset: {tf.data.experimental.cardinality(test_dataset).numpy()} batches.")

    return (
        train_dataset, val_dataset, test_dataset, 
        num_train_samples, num_val_samples, num_test_samples, 
        class_names, num_classes
    )

if __name__ == '__main__':
    logger.info("Testing load_classification_data function...")
    dummy_config_classification = {
        'data': {
            'paths': {
                'metadata_file': 'data/classification/metadata_small_test.json', 
                'label_map_file': 'data/classification/label_map_small_test.json'
            },
            'image_size': [64, 64], 
            'validation_split': 0.2,
            'test_split': 0.1,
            'random_seed': 42,
            'shuffle_buffer_size': 10, 
            'shuffle_validation_data': True,
            'augmentation': {
                'enabled': True,
                'augment_training_data': True,
                'augment_validation_data': False, 
                'horizontal_flip': True,
                'rotation_range': 10, 
                'zoom_range': 0.1,
                'brightness_range': [0.8, 1.2]
            }
        },
        'training': {
            'batch_size': 4, 
            'use_class_weights': True, 
            'mixup_alpha': 0.2, 
            'cutmix_alpha': 0.1 
        },
        'model': {
            'architecture': 'MobileNetV2' 
        }
    }

    project_r = _get_project_root()
    os.makedirs(project_r / 'data/classification', exist_ok=True)
    dummy_image_base_dir = project_r / "dummy_images_clf_test" 
    os.makedirs(dummy_image_base_dir / "class_a", exist_ok=True)
    os.makedirs(dummy_image_base_dir / "class_b", exist_ok=True)
    os.makedirs(dummy_image_base_dir / "class_c", exist_ok=True)

    dummy_label_map = {0: "class_a", 1: "class_b", 2: "class_c"}
    with open(project_r / dummy_config_classification['data']['paths']['label_map_file'], 'w') as f:
        json.dump(dummy_label_map, f)

    dummy_metadata_content = {"images": []}
    images_created_count = 0
    try:
        from PIL import Image
        for i in range(15):
            for class_idx, class_label_str_val in dummy_label_map.items():
                img_rel_path = f"dummy_images_clf_test/{class_label_str_val}/img_{i}_{class_label_str_val}.png"
                img_abs_path = project_r / img_rel_path
                
                dummy_pil_img = Image.new('RGB', (20, 20), color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
                dummy_pil_img.save(img_abs_path)
                dummy_metadata_content["images"].append({"path": img_rel_path, "label": class_label_str_val})
                images_created_count +=1
    except ImportError: 
        logger.warning("Pillow not installed. Cannot create dummy image files for testing. Test might fail at image loading.")
    except Exception as e_img: 
        logger.error(f"Error creating dummy image {img_abs_path}: {e_img}")

    if images_created_count == 0 : 
        logger.warning("No dummy images were created. Test might not be meaningful.")
    else: 
        logger.info(f"Created {images_created_count} dummy images for testing in {dummy_image_base_dir}.")

    with open(project_r / dummy_config_classification['data']['paths']['metadata_file'], 'w') as f:
        json.dump(dummy_metadata_content, f)
    
    logger.info(f"Created dummy metadata: {project_r / dummy_config_classification['data']['paths']['metadata_file']}")
    logger.info(f"Created dummy label map: {project_r / dummy_config_classification['data']['paths']['label_map_file']}")

    logger.info("\n--- Testing with max_samples_to_load = 7 ---")
    datasets_debug = load_classification_data(dummy_config_classification, max_samples_to_load=7)
    if datasets_debug and datasets_debug[0] is not None: 
        train_ds_debug, val_ds_debug, test_ds_debug, num_tr_debug, num_v_debug, num_t_debug, cn_debug, nc_debug = datasets_debug
        logger.info(f"Debug - Num train: {num_tr_debug}, Num val: {num_v_debug}, Num test: {num_t_debug}. Total actual: {num_tr_debug+num_v_debug+num_t_debug} (expected <=7). Classes: {nc_debug}, Names: {cn_debug}")
        if train_ds_debug:
            for img_batch, lbl_batch in train_ds_debug.take(1): 
                logger.info(f"Debug train batch - Images shape: {img_batch.shape}, Labels shape: {lbl_batch.shape}")
                pass
            logger.info("Successfully iterated over one batch of debug training data.")
        if val_ds_debug:
             for img_batch, lbl_batch in val_ds_debug.take(1): 
                logger.info(f"Debug val batch - Images shape: {img_batch.shape}, Labels shape: {lbl_batch.shape}")
                pass
    else: 
        logger.error("load_classification_data with max_samples_to_load failed or returned None for train_dataset.")

    logger.info("\n--- Testing with full dummy dataset ({} images) ---".format(images_created_count))
    datasets_full = load_classification_data(dummy_config_classification)
    if datasets_full and datasets_full[0] is not None:
        train_ds_full, val_ds_full, test_ds_full, n_tr_f, n_v_f, n_t_f, cn_f, nc_f = datasets_full
        logger.info(f"Full - Num train: {n_tr_f}, Num val: {n_v_f}, Num test: {n_t_f}. Total: {n_tr_f+n_v_f+n_t_f}. Classes: {nc_f}, Names: {cn_f}")
        if train_ds_full:
            for img_batch, lbl_batch in train_ds_full.take(1): 
                logger.info(f"Full train batch - Images shape: {img_batch.shape}, Labels shape: {lbl_batch.shape}")
                pass
            logger.info("Successfully iterated over one batch of full training data.")
        if val_ds_full:
            for img_batch, lbl_batch in val_ds_full.take(1): 
                logger.info(f"Full val batch - Images shape: {img_batch.shape}, Labels shape: {lbl_batch.shape}")
                pass
    else: 
        logger.error("load_classification_data with full dataset failed or returned None for train_dataset.")
    
    logger.info("Test finished. Consider cleaning up dummy files and directories if created by Pillow, especially in 'dummy_images_clf_test'.")
