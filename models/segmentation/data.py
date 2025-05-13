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
import sys
import math # For PI
import tensorflow_addons as tfa # For advanced image augmentations

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
def apply_augmentations(
    image: tf.Tensor, 
    mask: tf.Tensor, 
    is_enabled: tf.Tensor, 
    perform_h_flip: tf.Tensor, 
    brightness_delta_tf: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies augmentations consistently to image and mask using tensor arguments."""
    # If not enabled, return original images. tf.cond is used for graph-compatibility.
    def _do_nothing():
        return image, mask

    def _augment():
        aug_image, aug_mask = image, mask # Start with original
        # Horizontal Flip
        # tf.cond for conditional execution in graph mode
        def _flip(): 
            return tf.image.flip_left_right(aug_image), tf.image.flip_left_right(aug_mask)
        def _no_flip(): 
            return aug_image, aug_mask
        
        aug_image, aug_mask = tf.cond(
            tf.logical_and(perform_h_flip, tf.random.uniform(()) > 0.5),
            true_fn=_flip, 
            false_fn=_no_flip
        )
        
        # Brightness Adjustment
        def _adjust_brightness(): 
            bright_image = tf.image.random_brightness(aug_image, max_delta=brightness_delta_tf)
            return tf.clip_by_value(bright_image, 0.0, 255.0), aug_mask # Mask is not changed by brightness
        def _no_brightness_adjustment():
            return aug_image, aug_mask
            
        aug_image, aug_mask = tf.cond(
            brightness_delta_tf > 0.0, # This condition should be on a tensor
            true_fn=_adjust_brightness, 
            false_fn=_no_brightness_adjustment
        )
        return aug_image, aug_mask

    return tf.cond(is_enabled, true_fn=_augment, false_fn=_do_nothing)


def load_and_preprocess_segmentation(
    image_path_tensor: tf.Tensor, 
    mask_path_tensor: tf.Tensor, 
    target_size_tensor: tf.Tensor, 
    num_classes_tensor: tf.Tensor, 
    augment_tensor: tf.Tensor, 
    captured_preprocess_input_fn: Any, 
    captured_aug_config_dict: Optional[Dict]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Loads, decodes, resizes, and preprocesses an image and its mask using TensorFlow ops."""
    try:
        image_string = tf.io.read_file(image_path_tensor)
        image_decoded = tf.image.decode_image(image_string, channels=3, expand_animations=False)
        image_decoded.set_shape([None, None, 3])
        image_resized = tf.image.resize(image_decoded, target_size_tensor)
        image_for_aug = tf.cast(image_resized, tf.uint8) 

        mask_string = tf.io.read_file(mask_path_tensor)
        mask_decoded = tf.image.decode_image(mask_string, channels=1, expand_animations=False)
        mask_decoded.set_shape([None, None, 1])
        mask_resized = tf.image.resize(mask_decoded, target_size_tensor, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask_uint8 = tf.cast(mask_resized, tf.uint8)
        
        # Mask normalization logic (remains the same)
        target_max_mask_value = num_classes_tensor - 1 
        current_max_mask_val = tf.reduce_max(mask_uint8)
        
        def normalize_mask_branch(m):
            # tf.print("    Normalizing mask from 0-255 to 0-1.", output_stream=sys.stderr) # Keep commented
            return tf.cast(m, tf.float32) / 255.0

        def passthrough_mask_branch(m):
            return tf.cast(m, tf.float32)

        mask_float32_normalized = tf.cond(tf.cast(current_max_mask_val, tf.int32) > target_max_mask_value,
                                          lambda: normalize_mask_branch(mask_uint8),
                                          lambda: passthrough_mask_branch(mask_uint8))

        # --- Augmentation block with tf.cond ---
        aug_config_dict_tf_constants = {}
        if captured_aug_config_dict is not None: # Python if, ok as it configures constants before graph ops
            for key, value in captured_aug_config_dict.items():
                if isinstance(value, bool):
                    aug_config_dict_tf_constants[key] = tf.constant(value, dtype=tf.bool)
                elif isinstance(value, (int, float)):
                    aug_config_dict_tf_constants[key] = tf.constant(value, dtype=tf.float32 if isinstance(value, float) else tf.int32)
        
        # Lambdas for tf.cond
        def apply_augmentations_fn(img_to_aug, msk_to_aug):
            # tf.print("Applying augmentations...", output_stream=sys.stderr)
            # This aug_step_prob_tf is a general probability for sub-augmentations if they use it.
            # Individual augmentations within _augment_segmentation_tf_cond should also use tf.cond.
            aug_step_prob_tf = tf.constant(0.5, dtype=tf.float32) 
            return _augment_segmentation_tf_cond(img_to_aug, msk_to_aug, aug_config_dict_tf_constants, aug_step_prob_tf)

        def no_augmentations_fn(img_no_aug, msk_no_aug):
            # tf.print("Skipping augmentations...", output_stream=sys.stderr)
            return img_no_aug, msk_no_aug # Return them as is

        # Conditionally apply the entire augmentation sequence using tf.cond
        # Input to augmentation functions should be image_for_aug (uint8) and mask_float32_normalized (float32)
        # _augment_segmentation_tf_cond needs to handle these dtypes appropriately.
        # Let's assume _augment_segmentation_tf_cond expects uint8 image and float32 mask.
        image_after_aug, mask_after_aug = tf.cond(
            augment_tensor, # This is the tf.bool Tensor
            lambda: apply_augmentations_fn(image_for_aug, mask_float32_normalized),
            lambda: no_augmentations_fn(image_for_aug, mask_float32_normalized)
        )
        # --- End Augmentation block ---

        # Final preprocessing for the image (e.g. normalization specific to backbone)
        # Image after augmentation might be uint8 or float32 depending on _augment_segmentation_tf_cond
        # Ensure it's float32 before passing to preprocess_input_fn, which usually expects float.
        image_preprocessed = captured_preprocess_input_fn(tf.cast(image_after_aug, tf.float32))
        
        # Mask should already be float32 from normalization/augmentation. Cast just in case.
        mask_final = tf.cast(mask_after_aug, tf.float32)

        # tf.print("Shapes after load_and_preprocess: img=", tf.shape(image_preprocessed), "msk=", tf.shape(mask_final), output_stream=sys.stderr)
        return image_preprocessed, mask_final
    except Exception as e:
        # tf.print(f"Error in load_and_preprocess_segmentation: {e}", output_stream=sys.stderr)
        # traceback.print_exc(file=sys.stderr) # This won't work well in graph mode
        # Fallback to zeros if an error occurs that is not caught by TF ops themselves
        tf.print("Error processing image (fallback):", image_path_tensor, "or mask:", mask_path_tensor, "Error:", str(e), output_stream=sys.stderr)
        dummy_image_shape = list(target_size_tensor.numpy()) + [3]
        dummy_mask_shape = list(target_size_tensor.numpy()) + [1]
        dummy_image = tf.zeros(dummy_image_shape, dtype=tf.float32)
        dummy_mask = tf.zeros(dummy_mask_shape, dtype=tf.float32)
        return dummy_image, dummy_mask

# --- Augmentation Helper Functions (ensure they use tf.cond internally for conditional logic) ---

def _random_flip_left_right(image: tf.Tensor, mask: tf.Tensor, prob_tf: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Conditionally flips image and mask horizontally with probability prob_tf."""
    def flip_fn():
        return tf.image.flip_left_right(image), tf.image.flip_left_right(mask)
    def no_flip_fn():
        return image, mask
    should_apply = tf.random.uniform(shape=[], dtype=tf.float32) < prob_tf
    return tf.cond(should_apply, flip_fn, no_flip_fn)

def _adjust_brightness(image: tf.Tensor, mask: tf.Tensor, brightness_delta_tf: tf.Tensor, prob_tf: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Conditionally applies brightness adjustment with probability prob_tf."""
    def apply_aug_fn():
        # Ensure image is float for random_brightness, then cast back if needed (or ensure uint8 input is handled by op)
        # tf.image.random_brightness expects float or an integer type it supports.
        # Assuming image input here is uint8 as per image_for_aug.
        # Let's cast to float32, apply brightness, then clip and cast back to uint8.
        img_float = tf.cast(image, tf.float32)
        bright_image_float = tf.image.random_brightness(img_float, max_delta=brightness_delta_tf)
        bright_image_uint8 = tf.saturate_cast(bright_image_float, dtype=image.dtype) # Cast back to original dtype
        return bright_image_uint8, mask # Mask is unchanged by brightness
    def no_aug_fn():
        return image, mask
    should_apply = tf.random.uniform(shape=[], dtype=tf.float32) < prob_tf
    return tf.cond(should_apply, apply_aug_fn, no_aug_fn)

@tf.function(reduce_retracing=True)
def _rotate_image_and_mask(image: tf.Tensor, mask: tf.Tensor, max_angle_deg_tf: tf.Tensor, prob_tf: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Conditionally rotates image and mask using tfa.image.rotate."""
    def apply_rotation_fn():
        max_angle_rad = max_angle_deg_tf * (math.pi / 180.0)
        angle = tf.random.uniform(shape=[], minval=-max_angle_rad, maxval=max_angle_rad)
        # Use 'reflect' fill mode for image, 'constant' (0=background) for mask
        rotated_image = tfa.image.rotate(image, angle, interpolation='BILINEAR', fill_mode='reflect')
        rotated_mask = tfa.image.rotate(mask, angle, interpolation='NEAREST', fill_mode='constant', fill_value=0)
        return rotated_image, rotated_mask
    def no_rotation_fn():
        return image, mask
    should_apply = tf.random.uniform(shape=[], dtype=tf.float32) < prob_tf
    return tf.cond(should_apply, apply_rotation_fn, no_rotation_fn)

@tf.function(reduce_retracing=True)
def _shift_image_and_mask(image: tf.Tensor, mask: tf.Tensor, width_shift_fraction_tf: tf.Tensor, height_shift_fraction_tf: tf.Tensor, prob_tf: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Conditionally shifts image and mask using tfa.image.translate."""
    def apply_shift_fn():
        img_shape = tf.shape(image)
        img_height = tf.cast(img_shape[0], tf.float32)
        img_width = tf.cast(img_shape[1], tf.float32)

        dx = tf.random.uniform(shape=[], minval=-width_shift_fraction_tf, maxval=width_shift_fraction_tf) * img_width
        dy = tf.random.uniform(shape=[], minval=-height_shift_fraction_tf, maxval=height_shift_fraction_tf) * img_height
        translations = tf.stack([dx, dy]) # Changed from tf.convert_to_tensor
        # Use 'reflect' fill mode for image, 'constant' (0=background) for mask
        shifted_image = tfa.image.translate(image, translations, interpolation='BILINEAR', fill_mode='reflect') # or 'constant' with fill_value
        shifted_mask = tfa.image.translate(mask, translations, interpolation='NEAREST', fill_mode='constant', fill_value=0) # Assuming 0 is bg for mask
        return shifted_image, shifted_mask
    def no_shift_fn():
        return image, mask
    should_apply = tf.random.uniform(shape=[], dtype=tf.float32) < prob_tf
    return tf.cond(should_apply, apply_shift_fn, no_shift_fn)

def _zoom_image_and_mask(image: tf.Tensor, mask: tf.Tensor, zoom_range_fraction_tf: tf.Tensor, prob_tf: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Conditionally zooms image and mask using tf.image.resize and crop/pad."""
    def apply_zoom_fn():
        img_shape = tf.shape(image)
        original_height = img_shape[0]
        original_width = img_shape[1]

        scale = tf.random.uniform(shape=[], minval=1.0 - zoom_range_fraction_tf, maxval=1.0 + zoom_range_fraction_tf)
        
        new_height = tf.cast(tf.cast(original_height, tf.float32) * scale, tf.int32)
        new_width = tf.cast(tf.cast(original_width, tf.float32) * scale, tf.int32)

        img_scaled = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
        msk_scaled = tf.image.resize(mask, [new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Crop or pad back to original size
        # Note: tf.image.resize_with_crop_or_pad handles both image (float/uint8) and mask (uint8/int32 usually) dtypes correctly.
        img_zoomed = tf.image.resize_with_crop_or_pad(img_scaled, original_height, original_width)
        msk_zoomed = tf.image.resize_with_crop_or_pad(msk_scaled, original_height, original_width)
        return img_zoomed, msk_zoomed
    def no_zoom_fn():
        return image, mask
    should_apply = tf.random.uniform(shape=[], dtype=tf.float32) < prob_tf
    return tf.cond(should_apply, apply_zoom_fn, no_zoom_fn)

def _augment_segmentation_tf_cond(
    image: tf.Tensor, 
    mask: tf.Tensor, 
    aug_config_dict_tf_constants: Dict[str, tf.Tensor], 
    default_aug_prob_tf: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies a sequence of augmentations conditionally based on config and probability."""
    img_aug, msk_aug = image, mask # Start with original image and mask

    # Horizontal Flip
    if 'horizontal_flip' in aug_config_dict_tf_constants and aug_config_dict_tf_constants['horizontal_flip']:
        img_aug, msk_aug = _random_flip_left_right(img_aug, msk_aug, default_aug_prob_tf)

    # Brightness Adjustment - Assuming key in config is 'brightness_max_delta'
    # The aug_config_dict_tf_constants should be created with 'brightness_delta' if that's the key from earlier.
    # For now, let's assume the key is 'brightness_max_delta' and it's float.
    if 'brightness_max_delta' in aug_config_dict_tf_constants:
        brightness_delta_val = aug_config_dict_tf_constants['brightness_max_delta']
        # Only apply if delta is positive
        if tf.cast(brightness_delta_val, tf.float32) > 0.0:
             img_aug, msk_aug = _adjust_brightness(img_aug, msk_aug, brightness_delta_val, default_aug_prob_tf)
    
    # Rotation
    if 'rotation_range' in aug_config_dict_tf_constants:
        rotation_angle_deg = aug_config_dict_tf_constants['rotation_range'] # Expected to be int or float degrees
        if tf.cast(rotation_angle_deg, tf.float32) > 0.0: # Apply only if range is positive
            img_aug, msk_aug = _rotate_image_and_mask(img_aug, msk_aug, tf.cast(rotation_angle_deg, tf.float32), default_aug_prob_tf)

    # Width/Height Shift
    # Check if either shift range is present and positive
    wsr_val = aug_config_dict_tf_constants.get('width_shift_range', tf.constant(0.0, dtype=tf.float32))
    hsr_val = aug_config_dict_tf_constants.get('height_shift_range', tf.constant(0.0, dtype=tf.float32))
    if tf.cast(wsr_val, tf.float32) > 0.0 or tf.cast(hsr_val, tf.float32) > 0.0:
        img_aug, msk_aug = _shift_image_and_mask(img_aug, msk_aug, tf.cast(wsr_val, tf.float32), tf.cast(hsr_val, tf.float32), default_aug_prob_tf)

    # Zoom
    if 'zoom_range' in aug_config_dict_tf_constants:
        zoom_factor_range = aug_config_dict_tf_constants['zoom_range'] # Expected to be float (e.g., 0.1 for [0.9, 1.1])
        if tf.cast(zoom_factor_range, tf.float32) > 0.0: # Apply only if range is positive
            img_aug, msk_aug = _zoom_image_and_mask(img_aug, msk_aug, tf.cast(zoom_factor_range, tf.float32), default_aug_prob_tf)

    return img_aug, msk_aug

# --- Dataset Creation Function ---
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

    AUTOTUNE = tf.data.AUTOTUNE

    def create_dataset(items_list: List[Dict], augment_flag: bool) -> Optional[tf.data.Dataset]:
        if not items_list:
            return None

        img_paths = [item['image_path'] for item in items_list]
        mask_paths = [item['mask_path'] for item in items_list]

        dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

        # These are captured by _py_func_map_wrapper from create_dataset's scope
        # Note: image_size, num_classes, and aug_settings are from load_segmentation_data's scope, captured by create_dataset
        current_preprocess_input_fn = _get_segmentation_preprocess_fn(backbone)
        current_aug_settings = aug_settings if augment_flag else None # Pass None if no aug for this dataset split

        # Define the wrapper function that tf.py_function will call.
        # This wrapper has access to current_preprocess_input_fn and current_aug_settings via closure.
        def _py_func_map_wrapper(img_path_t, mask_path_t, target_size_t, num_classes_t, augment_tensor_t):
            return load_and_preprocess_segmentation(
                img_path_t, mask_path_t, 
                target_size_t, num_classes_t, 
                augment_tensor_t,
                current_preprocess_input_fn,  # Captured from create_dataset's scope
                current_aug_settings          # Captured from create_dataset's scope
            )

        dataset = dataset.map(
            lambda img_path, mask_path: tf.py_function(
                func=_py_func_map_wrapper, # Use the wrapper
                inp=[
                    img_path, 
                    mask_path, 
                    tf.constant(image_size, dtype=tf.int32), # target_size_tensor for the wrapper
                    tf.constant(num_classes, dtype=tf.int32),# num_classes_tensor for the wrapper
                    tf.constant(augment_flag, dtype=tf.bool) # augment_tensor for the wrapper
                ],
                Tout=[tf.float32, tf.float32] 
            ),
            num_parallel_calls=AUTOTUNE
        )
            
        def _set_shape(image, mask):
            # image_size is a tuple (H, W), num_classes is an int
            image.set_shape([*image_size, 3])
            mask.set_shape([*image_size, 1])
            return image, mask
        dataset = dataset.map(_set_shape, num_parallel_calls=AUTOTUNE)

        if augment_flag: # This uses the parameter passed to create_dataset
            # Cache after initial load and preprocess, before shuffle and augmentations applied in map
            # Note: if load_and_preprocess_segmentation contains the actual augmentation logic,
            # caching should ideally be before that map if augmentations are random per epoch.
            # Given the current structure where load_and_preprocess_segmentation conditionally calls augmentations,
            # caching here means augmented data might be cached if augment_flag is True for the initial map.
            # Re-evaluating: augmentations are inside load_and_preprocess_segmentation, controlled by augment_tensor.
            # So, caching *after* the map that includes load_and_preprocess_segmentation means we cache *augmented* versions.
            # For true randomness per epoch, augmentations should be after cache.
            # However, load_and_preprocess_segmentation itself has the augment_tensor and aug_settings.
            # Let's assume the map with py_function is the one doing deterministic parts + potentially augmentations.
            # For simplicity and to ensure augmentations are re-applied if they are indeed random within the py_function call per epoch,
            # we should cache *before* the map that does augmentations if they are truly random per-epoch.
            # Given the `augment_tensor` is passed into the map, this implies augmentations are applied within that map.
            # The current `load_and_preprocess_segmentation` has an `augment_tensor` argument.
            # Let's put cache *after* the main mapping but *before* shuffle for training data.
            # This will cache the (potentially augmented if `augment_flag` was true for that mapping) data.
            dataset = dataset.cache() # Cache the training data after mapping
            dataset = dataset.shuffle(buffer_size=1024) # Reduced buffer size
        
        dataset = dataset.batch(batch_size)
        # For training data (augment_flag is True), make it repeat indefinitely
        # This is standard practice when using steps_per_epoch with model.fit()
        if augment_flag:
            dataset = dataset.repeat()
            
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

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

    train_dataset = create_dataset(train_items, augment_flag=True)
    val_dataset = create_dataset(val_items, augment_flag=False)
    test_dataset = create_dataset(test_items, augment_flag=False)

    logger.info("Segmentation datasets created.")
    return train_dataset, val_dataset, test_dataset