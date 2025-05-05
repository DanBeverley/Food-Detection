import os
import yaml
import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Optional, List
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
import pathlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_project_root() -> pathlib.Path:
    """Find the project root directory."""
    return pathlib.Path(__file__).parent.parent.parent

def find_image_mask_pairs(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    """Find corresponding image and mask files in directories."""
    image_files = {pathlib.Path(f).stem: os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if os.path.isfile(os.path.join(image_dir, f))}
    mask_files = {pathlib.Path(f).stem: os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                  if os.path.isfile(os.path.join(mask_dir, f))}

    pairs = []
    for stem, img_path in image_files.items():
        if stem in mask_files:
            pairs.append((img_path, mask_files[stem]))
        else:
            logger.warning(f"No corresponding mask found for image: {img_path}")

    if not pairs:
        raise FileNotFoundError(f"No image-mask pairs found in {image_dir} and {mask_dir}")
    logger.info(f"Found {len(pairs)} image-mask pairs.")
    return pairs

def split_data(image_mask_pairs: List[Tuple[str, str]], val_ratio: float, test_ratio: float, random_state: int = 42):
    """Splits data into train, validation, and test sets."""
    num_total = len(image_mask_pairs)
    num_test = int(num_total * test_ratio)
    num_val = int(num_total * val_ratio)
    num_train = num_total - num_val - num_test

    if num_train <= 0 or num_val < 0 or num_test < 0:
        raise ValueError("Invalid split ratios result in non-positive set sizes.")

    logger.info(f"Splitting data: Train={num_train}, Val={num_val}, Test={num_test}")

    # First split off test set
    if num_test > 0:
        train_val_pairs, test_pairs = train_test_split(
            image_mask_pairs, test_size=num_test, random_state=random_state, shuffle=True
        )
    else:
        train_val_pairs = image_mask_pairs
        test_pairs = []

    # Split remaining into train and validation
    if num_val > 0 and len(train_val_pairs) > num_val:
         # Calculate val proportion relative to the remaining train_val set
        relative_val_ratio = num_val / len(train_val_pairs)
        train_pairs, val_pairs = train_test_split(
            train_val_pairs, test_size=relative_val_ratio, random_state=random_state, shuffle=True
        )
    elif num_val > 0:
         # Handle case where only validation split is requested, no test
        train_pairs = []
        val_pairs = train_val_pairs
    else: # No validation or test split needed
        train_pairs = train_val_pairs
        val_pairs = []

    logger.info(f"Split complete: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs

def load_and_preprocess(image_path: tf.Tensor, mask_path: tf.Tensor, target_size: Tuple[int, int], num_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Loads and preprocesses a single image-mask pair."""
    image_path_str = tf.compat.as_str_any(image_path.numpy())
    mask_path_str = tf.compat.as_str_any(mask_path.numpy())

    # Load Image
    image = tf.io.read_file(image_path_str)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, target_size)
    # Ensure float32 in [0, 255] range for preprocess_input
    image = tf.cast(image, tf.float32)
    # Apply EfficientNet preprocessing
    image = preprocess_input(image) 

    # Load Mask
    mask = tf.io.read_file(mask_path_str)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Handle mask values (ensure float32 [0, 1] for common losses like Dice/BCE with sigmoid)
    if tf.reduce_max(mask) > 1:
        mask = mask / 255.0
    mask = tf.cast(mask, tf.float32)

    return image, mask

@tf.function
def apply_augmentations(image: tf.Tensor, mask: tf.Tensor, config: Dict) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies augmentations consistently to image and mask."""
    augment_config = config.get('data', {}).get('augmentation', {})
    if not augment_config.get('enabled', False):
        return image, mask

    seed = tf.random.experimental.stateless_split(tf.random.Generator.from_seed(123).normal([2]), num=1)[0]

    # Horizontal Flip
    if augment_config.get('horizontal_flip', False) and tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Rotation
    rotation_range = augment_config.get('rotation_range', 0)
    if rotation_range > 0:
        degrees = rotation_range * tf.random.uniform((), minval=-1, maxval=1)
        radians = degrees * (np.pi / 180.0)
        image = tf.contrib.image.rotate(image, radians, interpolation='BILINEAR')
        mask = tf.contrib.image.rotate(mask, radians, interpolation='NEAREST') # Use nearest for mask

    # Width/Height Shift
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    width_shift_range = augment_config.get('width_shift_range', 0.0)
    height_shift_range = augment_config.get('height_shift_range', 0.0)
    if width_shift_range > 0.0 or height_shift_range > 0.0:
        tx = width_shift_range * tf.cast(width, tf.float32) * tf.random.uniform((), minval=-1, maxval=1)
        ty = height_shift_range * tf.cast(height, tf.float32) * tf.random.uniform((), minval=-1, maxval=1)
        transform = tf.convert_to_tensor([1, 0, tx, 0, 1, ty, 0, 0], dtype=tf.float32)
        image = tf.contrib.image.transform(image, transform, interpolation='BILINEAR')
        mask = tf.contrib.image.transform(mask, transform, interpolation='NEAREST')

    # Zoom
    zoom_range = augment_config.get('zoom_range', 0.0)
    if zoom_range > 0.0:
        zx = zy = tf.random.uniform((), minval=1.0 - zoom_range, maxval=1.0 + zoom_range)
        # TODO: Implement zoom augmentation carefully (tf.image.crop_and_resize might be needed)
        # This requires calculating crop boxes which is more involved.
        # Skipping zoom for now, can be added later if critical.
        pass

    # Brightness/Contrast (apply only to image)
    brightness_delta = augment_config.get('brightness_delta', 0.0)
    if brightness_delta > 0:
         image = tf.image.stateless_random_brightness(image, max_delta=brightness_delta, seed=seed)
    contrast_range = augment_config.get('contrast_range', [])
    if len(contrast_range) == 2:
        image = tf.image.stateless_random_contrast(image, lower=contrast_range[0], upper=contrast_range[1], seed=seed)

    return image, mask

def load_segmentation_datasets(config_path: str) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset]]:
    """Loads train, validation, and test datasets based on the config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from: {config_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        return None, None, None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        return None, None, None

    data_config = config.get('data', {})
    aug_config = data_config.get('augmentation', {})
    project_root = _get_project_root()

    # Get paths and parameters
    image_dir_rel = data_config.get('image_dir', 'data/segmentation/images/')
    mask_dir_rel = data_config.get('mask_dir', 'data/segmentation/masks/')
    image_dir = os.path.join(project_root, image_dir_rel)
    mask_dir = os.path.join(project_root, mask_dir_rel)
    image_size = tuple(data_config.get('image_size', [256, 256]))
    batch_size = data_config.get('batch_size', 16)
    val_split = data_config.get('validation_split_ratio', 0.15)
    test_split = data_config.get('test_split_ratio', 0.15)
    num_classes = data_config.get('num_classes', 2)
    apply_augmentation = aug_config.get('enabled', False)

    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        logger.error(f"Image ({image_dir}) or Mask ({mask_dir}) directory not found.")
        return None, None, None

    # Find and Split Data
    try:
        all_pairs = find_image_mask_pairs(image_dir, mask_dir)
    except FileNotFoundError as e:
        logger.error(e)
        return None, None, None

    if not all_pairs:
        return None, None, None

    img_paths, mask_paths = zip(*all_pairs)

    if test_split > 0:
        img_train_val, img_test, mask_train_val, mask_test = train_test_split(
            img_paths, mask_paths, test_size=test_split, random_state=42, shuffle=True)
    else:
        img_train_val, mask_train_val = img_paths, mask_paths
        img_test, mask_test = [], []

    if val_split > 0 and img_train_val:
        val_split_adjusted = val_split / (1.0 - test_split) if (1.0 - test_split) > 0 else 0
        if val_split_adjusted >= 1.0:
             img_train, mask_train = [], []
             img_val, mask_val = img_train_val, mask_train_val
        else:
            img_train, img_val, mask_train, mask_val = train_test_split(
                img_train_val, mask_train_val, test_size=val_split_adjusted, random_state=42, shuffle=True)
    else:
        img_train, mask_train = img_train_val, mask_train_val
        img_val, mask_val = [], []

    logger.info(f"Dataset split: Train={len(img_train)}, Validation={len(img_val)}, Test={len(img_test)}")

    # Create tf.data.Datasets
    def create_dataset(img_paths_list, mask_paths_list, is_training):
        if not img_paths_list:
            return None
        dataset = tf.data.Dataset.from_tensor_slices((list(img_paths_list), list(mask_paths_list)))
        
        dataset = dataset.map(lambda img_p, mask_p: tf.py_function(
            func=load_and_preprocess,
            inp=[img_p, mask_p, image_size, num_classes],
            Tout=[tf.float32, tf.float32]
        ), num_parallel_calls=tf.data.AUTOTUNE)
        
        def set_shape(image, mask):
            image.set_shape([image_size[0], image_size[1], 3])
            mask.set_shape([image_size[0], image_size[1], 1]) # Assuming 1 channel mask
            return image, mask
        dataset = dataset.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)

        if is_training:
            dataset = dataset.cache()
            dataset = dataset.shuffle(buffer_size=len(img_paths_list))
            if apply_augmentation:
                dataset = dataset.map(lambda img, mask: apply_augmentations(img, mask, aug_config), num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    train_ds = create_dataset(img_train, mask_train, is_training=True)
    val_ds = create_dataset(img_val, mask_val, is_training=False)
    test_ds = create_dataset(img_test, mask_test, is_training=False)

    return train_ds, val_ds, test_ds