import os
import yaml
import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Optional, List, Any
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
import pathlib
import glob
import json

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

def load_and_preprocess(image_path_tensor, mask_path_tensor, target_size: Tuple[int, int], num_classes: int):
    """Loads and preprocesses a single image and mask."""
    try: # Add outer try block
        # Decode tensor paths to strings
        image_path = image_path_tensor.numpy().decode('utf-8')
        mask_path = mask_path_tensor.numpy().decode('utf-8')

        # Load Image
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, target_size)
            # Ensure float32 in [0, 255] range for preprocess_input
            image = tf.cast(image, tf.float32)
            # Apply EfficientNet preprocessing
            image = preprocess_input(image) 
        except Exception as img_e:
            logger.error(f"Error loading/processing image {image_path}: {img_e}")
            # Return dummy tensors on error to avoid crashing the whole pipeline?
            # This might hide errors but allow processing other files.
            # Or raise the exception:
            raise # Re-raise the exception to be caught by the outer block

        # Load Mask
        try:
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
            mask = tf.image.resize(mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            # Handle mask values (ensure float32 [0, 1] for common losses like Dice/BCE with sigmoid)
            if tf.reduce_max(mask) > 1:
                mask = mask / 255.0
            mask = tf.cast(mask, tf.float32)

        except Exception as mask_e:
            logger.error(f"Error loading/processing mask {mask_path}: {mask_e}")
            raise # Re-raise the exception

        return image, mask

    except BaseException as e: # Catch any exception during preprocessing
        # Log the error with traceback
        image_path_str = image_path_tensor.numpy().decode('utf-8') if hasattr(image_path_tensor, 'numpy') else 'unknown'
        mask_path_str = mask_path_tensor.numpy().decode('utf-8') if hasattr(mask_path_tensor, 'numpy') else 'unknown'
        logger.error(f"Unhandled error processing image '{image_path_str}' or mask '{mask_path_str}': {e}", exc_info=True)
        print(f"--- TRACEBACK FROM load_and_preprocess ({image_path_str}) ---")
        print(traceback.format_exc())
        print("----------------------------------------------------------")
        # Re-raise the exception to ensure the dataset pipeline fails clearly
        raise e

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

    try:
        train_ds = create_dataset(img_train, mask_train, is_training=True)
        val_ds = create_dataset(img_val, mask_val, is_training=False)
        test_ds = create_dataset(img_test, mask_test, is_training=False)
        
        # Force execution of the pipeline for one element to catch errors early
        if train_ds:
            logger.info("Forcing dataset iteration to check for loading errors...")
            _ = next(iter(train_ds))
            logger.info("Dataset iteration check passed.")
            
    except BaseException as e: # Catch BaseException and log details
        logger.error(f"Error creating or iterating dataset: {e}", exc_info=True) # Modified log message
        print("--- TRACEBACK FROM create_dataset/iteration ---") # Modified print message
        print(traceback.format_exc())
        print("-------------------------------------------")
        return None, None, None # Return None as before

    return train_ds, val_ds, test_ds

def _build_segmentation_augmentation_pipeline(config: Dict[str, Any]) -> Optional[tf.keras.Sequential]:
    """Builds a data augmentation pipeline for segmentation (image and mask)."""
    if not config.get('augmentation', {}).get('enabled', False):
        return None

    aug_config = config['augmentation']
    pipeline_ops = [] # Store ops to apply to both image and mask

    # Geometric transformations need to be applied consistently
    if aug_config.get('horizontal_flip', False):
        pipeline_ops.append(tf.keras.layers.RandomFlip("horizontal"))
    
    if aug_config.get('rotation_range', 0) > 0:
        factor = aug_config['rotation_range'] / 360.0
        pipeline_ops.append(tf.keras.layers.RandomRotation(factor, fill_mode='nearest'))
    
    # Corrected RandomZoom logic:
    # RandomZoom's height_factor/width_factor can be:
    # - A float `x` (e.g. 0.1), resulting in zoom range [1-x, 1+x] (i.e., factors [-x, x])
    # - A tuple `(low, high)` (e.g. (-0.1, 0.2)), resulting in zoom range [1+low, 1+high]
    # The config's 'zoom_range' (e.g. 0.1 or [-0.1, 0.1]) can be passed directly.
    if 'zoom_range' in aug_config and aug_config['zoom_range'] != 0: # Allow 0 or absence for no zoom
        zoom_value = aug_config['zoom_range'] # This can be a float like 0.1 or a tuple like (-0.1, 0.1)
        pipeline_ops.append(tf.keras.layers.RandomZoom(
            height_factor=zoom_value, 
            width_factor=zoom_value, 
            fill_mode='nearest'
        ))

    if aug_config.get("width_shift_range", 0) > 0 or aug_config.get("height_shift_range", 0) > 0:
        width_factor = aug_config.get("width_shift_range", 0)
        height_factor = aug_config.get("height_shift_range", 0)
        pipeline_ops.append(tf.keras.layers.RandomTranslation(height_factor=height_factor, width_factor=width_factor, fill_mode='nearest'))

    # Photometric augmentations (applied only to image typically, or carefully to mask if needed)
    # For simplicity, this example focuses on geometric ones for image-mask pairs.
    # If adding brightness etc., apply only to image tensor in load_and_preprocess_segmentation.

    if not pipeline_ops:
        return None

    # Create a model that applies these ops. Input will be a stacked image and mask.
    # We'll apply it by passing image and mask separately and concatenating.
    logger.info(f"Built segmentation augmentation pipeline with {len(pipeline_ops)} operations.")
    return pipeline_ops # Return list of layers to be applied

def load_and_preprocess_segmentation(
    image_path: tf.Tensor,
    mask_path: tf.Tensor,
    image_size: Tuple[int, int],
    augmentation_ops: Optional[List[tf.keras.layers.Layer]] = None,
    augment: bool = False
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Loads, decodes, resizes, and preprocesses an image and its mask."""
    try:
        # Load Image
        image_string = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image_string, channels=3) # Assuming JPG
        image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]

        # Load Mask
        mask_string = tf.io.read_file(mask_path)
        mask = tf.image.decode_jpeg(mask_string, channels=1) # Assuming JPG, load as grayscale
        mask = tf.image.resize(mask, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # Use nearest for masks
        
        # Cast mask to float32 before comparison with float threshold
        mask = tf.cast(mask, tf.float32) # Now tf.float32, values in [0.0, 255.0]

        # Convert mask to binary: pixel > threshold becomes 1 (foreground), else 0 (background)
        # Common threshold is 128 for 8-bit images, or 0.5 if normalized.
        # Assuming masks are simple (e.g., white object on black background or vice-versa)
        threshold = 127.5 # For 0-255 range pixel values (compares with float32 mask)
        mask = tf.cast(mask > threshold, tf.float32) # Binary mask [0.0, 1.0]
        mask.set_shape([*image_size, 1])

        if augment and augmentation_ops:
            # Apply augmentations consistently. Stack image and mask, augment, then unstack.
            # Some augmentations might need separate handling if they affect color vs. geometry.
            # For geometric ops, this is a common approach.
            
            # For layers like RandomBrightness, apply only to image.
            # This simplified example applies all ops to a concatenated tensor then splits.
            # More robust: separate image-only augs from geometric (image+mask) augs.

            # Ensure image is in [0,1] for augmentation if ops expect that.
            # Mask is already [0,1] binary.
            stacked_data = tf.concat([image, mask], axis=-1) # H, W, C_img+C_mask
            
            for op in augmentation_ops:
                # Augmentation layers expect batch dimension
                stacked_data_batch = tf.expand_dims(stacked_data, axis=0)
                augmented_batch = op(stacked_data_batch, training=True)
                stacked_data = tf.squeeze(augmented_batch, axis=0)
            
            image, mask = tf.split(stacked_data, [3, 1], axis=-1) # Split back
            
            # Ensure mask remains binary after geometric transformations (e.g. rotation might introduce non-binary values)
            mask = tf.round(mask) # or mask = tf.cast(mask > 0.5, tf.float32)
            image = tf.clip_by_value(image, 0.0, 1.0) # Ensure image stays in [0,1]

        image.set_shape([*image_size, 3])
        mask.set_shape([*image_size, 1])
        
        return image, mask
    except Exception as e:
        tf.print(f"Error in load_and_preprocess_segmentation for image {image_path}, mask {mask_path}: {e}")
        # Return dummy data or raise error. For now, re-raise to catch issues.
        raise

def load_segmentation_data(config: Dict[str, Any]) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset]]:
    """
    Loads segmentation data (image-mask pairs) from the specified directory structure.

    Args:
        config: Dictionary containing parameters:
            - dataset_root_dir: Path to the segmentation dataset (e.g., 'E:/.../RGBD_videos').
            - image_size: Tuple (height, width) for resizing.
            - batch_size: Integer batch size.
            - split_ratio: Float ratio for validation set (e.g., 0.2).
            - augmentation: Dictionary for augmentation settings.
            - random_seed: (Optional) Integer for reproducible splits.

    Returns:
        Tuple of (train_dataset, val_dataset). Datasets yield (image, mask) pairs.
    """
    project_root = _get_project_root()

    dataset_root_dir_str = config['dataset_root_dir']
    image_size = tuple(config['image_size'])
    batch_size = config['batch_size']
    split_ratio = config.get('split_ratio', 0.2) # Default to 0.2 if not provided
    random_seed = config.get('random_seed', 42)

    if not os.path.isabs(dataset_root_dir_str):
        dataset_root_dir = project_root / dataset_root_dir_str
    else:
        dataset_root_dir = pathlib.Path(dataset_root_dir_str)

    if not dataset_root_dir.exists():
        raise FileNotFoundError(f"Segmentation dataset root directory not found: {dataset_root_dir}")
    if not 0 <= split_ratio < 1:
        raise ValueError("Split ratio must be between 0 (no validation) and 1 (exclusive).")

    all_image_mask_pairs = [] # List of {'image_path': str, 'mask_path': str, 'instance_id': str}
    logger.info(f"Scanning segmentation dataset directory: {dataset_root_dir}")

    class_dirs = [d for d in dataset_root_dir.iterdir() if d.is_dir()]
    for class_dir in class_dirs:
        class_name = class_dir.name
        instance_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
        for instance_dir in instance_dirs:
            instance_id = f"{class_name}_{instance_dir.name}"
            original_dir = instance_dir / "original"
            masks_dir = instance_dir / "masks"

            if original_dir.is_dir() and masks_dir.is_dir():
                # Get all image files (jpg, png) from original
                original_files = list(original_dir.glob('*.jpg')) + list(original_dir.glob('*.png'))
                
                for img_path in original_files:
                    # Construct corresponding mask path (assuming same filename, could be jpg or png)
                    mask_path_jpg = masks_dir / (img_path.stem + '.jpg')
                    mask_path_png = masks_dir / (img_path.stem + '.png')
                    
                    mask_path = None
                    if mask_path_jpg.exists():
                        mask_path = mask_path_jpg
                    elif mask_path_png.exists():
                        mask_path = mask_path_png
                    
                    if mask_path:
                        all_image_mask_pairs.append({
                            'image_path': str(img_path),
                            'mask_path': str(mask_path),
                            'instance_id': instance_id
                        })
                    else:
                        logger.debug(f"Mask not found for image {img_path.name} in instance {instance_id}")
            else:
                logger.debug(f"'original' or 'masks' folder missing in {instance_dir}")

    if not all_image_mask_pairs:
        raise FileNotFoundError(f"No image-mask pairs found in the dataset structure at {dataset_root_dir}.")
    logger.info(f"Found {len(all_image_mask_pairs)} total image-mask pairs across all instances.")

    unique_instance_ids = sorted(list(set(item['instance_id'] for item in all_image_mask_pairs)))
    if not unique_instance_ids:
        raise ValueError("No instances found to perform train/val split.")

    train_instance_ids, val_instance_ids = [], []
    if split_ratio > 0:
        logger.info(f"Performing instance-aware split for {len(unique_instance_ids)} unique instances with ratio {split_ratio}.")
        train_instance_ids, val_instance_ids = train_test_split(
            unique_instance_ids,
            test_size=split_ratio,
            random_state=random_seed
        )
        logger.info(f"Train instances: {len(train_instance_ids)}, Validation instances: {len(val_instance_ids)}")
    else:
        train_instance_ids = unique_instance_ids # All data for training if split_ratio is 0
        logger.info(f"All {len(unique_instance_ids)} instances will be used for training (split_ratio=0).")

    train_img_paths, train_mask_paths = [], []
    val_img_paths, val_mask_paths = [], []

    for item in all_image_mask_pairs:
        if item['instance_id'] in train_instance_ids:
            train_img_paths.append(item['image_path'])
            train_mask_paths.append(item['mask_path'])
        elif item['instance_id'] in val_instance_ids:
            val_img_paths.append(item['image_path'])
            val_mask_paths.append(item['mask_path'])
    
    if not train_img_paths:
        raise ValueError("Training set is empty after split. Check dataset structure or split_ratio.")
    if split_ratio > 0 and not val_img_paths:
        logger.warning("Validation set is empty after split. This might be intended if split_ratio is very small, the dataset is tiny, or all instances went to train.")
    
    logger.info(f"Train image-mask pairs: {len(train_img_paths)}, Validation image-mask pairs: {len(val_img_paths)}")

    augmentation_ops = _build_segmentation_augmentation_pipeline(config)
    AUTOTUNE = tf.data.AUTOTUNE

    train_paths_ds = tf.data.Dataset.from_tensor_slices((list(train_img_paths), list(train_mask_paths)))
    logger.info(f"Train image-mask pairs: {len(train_img_paths)}, Validation image-mask pairs: {len(val_img_paths)}")

    # Build augmentation pipeline if enabled
    augmentation_pipeline = _build_segmentation_augmentation_pipeline(config.get('augmentation', {}))
    if augmentation_pipeline:
        logger.info(f"Built segmentation augmentation pipeline with {len(augmentation_pipeline.layers)} operations.")
    else:
        logger.info("Segmentation augmentation is disabled or no operations configured.")

    # Prepare training dataset
    train_dataset = (train_paths_ds
                     .shuffle(buffer_size=len(train_img_paths), seed=config.get('random_seed', None))
                     .map(lambda img_path, mask_path: load_and_preprocess_segmentation(img_path, mask_path, tuple(config['image_size']), augmentation_pipeline, augment=True), num_parallel_calls=AUTOTUNE)
                     .batch(config['batch_size'])
                     .prefetch(AUTOTUNE))
    logger.info("Training dataset for segmentation created.")

    val_dataset = None
    if val_img_paths:
        val_paths_ds = tf.data.Dataset.from_tensor_slices((list(val_img_paths), list(val_mask_paths)))
        val_dataset = (val_paths_ds
                       .map(lambda img_path, mask_path: load_and_preprocess_segmentation(img_path, mask_path, tuple(config['image_size']), augment=False), num_parallel_calls=AUTOTUNE) # No augmentation for validation
                       .batch(config['batch_size'])
                       .prefetch(AUTOTUNE))
        logger.info("Validation dataset for segmentation created.")

    # Limit dataset size for development/testing if specified
    max_train_samples = config.get('dev_max_train_samples')
    if max_train_samples and isinstance(max_train_samples, int) and max_train_samples > 0:
        train_dataset = train_dataset.take(max_train_samples // config['batch_size'] + (1 if (max_train_samples % config['batch_size']) > 0 else 0) )
        # The above ensures we take enough batches to cover max_train_samples
        logger.info(f"Limiting training dataset to approximately {max_train_samples} samples ({max_train_samples // config['batch_size'] + (1 if (max_train_samples % config['batch_size']) > 0 else 0)} batches).")

    max_val_samples = config.get('dev_max_val_samples')
    if val_dataset and max_val_samples and isinstance(max_val_samples, int) and max_val_samples > 0:
        val_dataset = val_dataset.take(max_val_samples // config['batch_size'] + (1 if (max_val_samples % config['batch_size']) > 0 else 0) )
        logger.info(f"Limiting validation dataset to approximately {max_val_samples} samples ({max_val_samples // config['batch_size'] + (1 if (max_val_samples % config['batch_size']) > 0 else 0)} batches).")

    return train_dataset, val_dataset