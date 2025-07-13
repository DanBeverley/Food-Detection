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
numpy_compat_attrs = {
    'fastCopyAndTranspose': np.transpose,
    'set_numeric_ops': lambda **kwargs: None,
    '_using_numpy2_behavior': lambda: False,
    'get_handler_name': lambda obj: 'default',
    'get_handler_version': lambda obj: (1, 0)
}
for attr, value in numpy_compat_attrs.items():
    if not hasattr(np.core.multiarray, attr):
        setattr(np.core.multiarray, attr, value)

# Patch numpy._type_aliases._bits_of for compatibility
try:
    import numpy.core._type_aliases as ta
    original_bits_of = ta._bits_of
    def _bits_of_patched(obj):
        try:
            return original_bits_of(obj)
        except AttributeError:
            return np.dtype(obj).itemsize * 8
    ta._bits_of = _bits_of_patched
except (ImportError, AttributeError):
    pass
import trimesh 
from tensorflow.keras import layers # Import Keras layers

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
            # Corrected import for ResNet variants
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

# Removed old apply_augmentations function as it's no longer used.

def _load_and_preprocess_point_cloud_py_seg(pc_path_bytes: bytes, num_points_target: int, normalization_method_str: str) -> np.ndarray:
    """Loads a point cloud, samples/pads to num_points_target, and normalizes.
    Args:
        pc_path_bytes: Path to the point cloud file, as bytes.
        num_points_target: Target number of points.
        normalization_method_str: String specifying normalization ('unit_sphere', 'unit_cube', 'centered_only', or 'none').
    Returns:
        A NumPy array of shape (num_points_target, 3) dtype=np.float32.
    """
    try:
        pc_path = pc_path_bytes.decode('utf-8')
        if not pc_path or not os.path.exists(pc_path):
            # logger.debug(f"Point cloud path is empty or does not exist: {pc_path}. Returning zeros.") # Reduce verbosity
            return np.zeros((num_points_target, 3), dtype=np.float32)

        mesh_or_points = trimesh.load(pc_path, process=False) 

        if isinstance(mesh_or_points, trimesh.Trimesh):
            points = mesh_or_points.vertices
        elif isinstance(mesh_or_points, trimesh.points.PointCloud):
            points = mesh_or_points.vertices
        else:
            # logger.warning(f"Loaded object from {pc_path} is not a Trimesh or PointCloud. Type: {type(mesh_or_points)}. Returning zeros.")
            return np.zeros((num_points_target, 3), dtype=np.float32)

        if points.shape[0] == 0:
            # logger.debug(f"No points found in {pc_path}. Returning zeros.")
            return np.zeros((num_points_target, 3), dtype=np.float32)

        points = points.astype(np.float32)

        current_num_points = points.shape[0]
        if current_num_points > num_points_target:
            indices = np.random.choice(current_num_points, num_points_target, replace=False)
            points = points[indices]
        elif current_num_points < num_points_target:
            if current_num_points == 0: # Should be caught above, but defensive
                return np.zeros((num_points_target, 3), dtype=np.float32)
            padding_indices = np.random.choice(current_num_points, num_points_target - current_num_points, replace=True)
            points = np.vstack((points, points[padding_indices]))
        
        if normalization_method_str != 'none':
            points_mean = np.mean(points, axis=0)
            points_centered = points - points_mean

            if normalization_method_str == 'unit_sphere':
                max_dist = np.max(np.linalg.norm(points_centered, axis=1))
                points_normalized = points_centered / (max_dist + 1e-6) # Add epsilon
            elif normalization_method_str == 'unit_cube':
                max_abs_coord = np.max(np.abs(points_centered))
                points_normalized = points_centered / (max_abs_coord + 1e-6) # Add epsilon
            elif normalization_method_str == 'centered_only':
                points_normalized = points_centered
            else: 
                points_normalized = points_centered # Fallback
            points = points_normalized
        
        return points.astype(np.float32)

    except Exception as e:
        path_str = pc_path_bytes.decode('utf-8', errors='ignore') 
        logger.error(f"Error processing point cloud {path_str}: {e}")
        return np.zeros((num_points_target, 3), dtype=np.float32)





class SegmentationAugmentation(tf.keras.Model):
    def __init__(self, aug_config, **kwargs):
        super().__init__(**kwargs)
        self.aug_config = aug_config
        
        # Create all layers ONCE in the constructor
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
        augmented_mask = tf.squeeze(image_and_mask[..., 6:], axis=-1)
        
        # Apply color layers to RGB only
        for layer in self.color_layers:
            augmented_rgb = layer(augmented_rgb)
        
        augmented_inputs = {
            'rgb_input': augmented_rgb,
            'depth_input': augmented_depth,
            'pc_input': inputs_dict['pc_input']
        }
        return augmented_inputs, augmented_mask

def load_and_preprocess_segmentation(
    input_paths_tuple: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    target_size_py: tuple,
    target_size_tensor: tf.Tensor, 
    num_classes_tensor: tf.Tensor, 
    augment_tensor: tf.Tensor,
    captured_preprocess_input_fn: Optional[Callable],
    augmentation_pipeline: Optional[SegmentationAugmentation],
    use_depth_map_tensor: tf.Tensor,
    depth_prep_cfg_dict: Dict, 
    use_point_cloud_tensor: tf.Tensor,
    pc_prep_cfg_dict: Dict
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    # Unpack the tensor elements. In graph mode, input_paths_tuple is a 1D tensor.
    image_path_tensor = input_paths_tuple[0]
    depth_path_tensor = input_paths_tuple[1]
    pc_path_tensor = input_paths_tuple[2]
    mask_path_tensor = input_paths_tuple[3]

    try:
        # --- RGB Image Processing ---
        image_string = tf.io.read_file(image_path_tensor)
        image_decoded = tf.image.decode_image(image_string, channels=3, expand_animations=False)
        image_decoded.set_shape([None, None, 3])
        image_resized = tf.image.resize(image_decoded, target_size_tensor)
        image_float_for_aug = tf.cast(image_resized, tf.float32) # Start with float for consistency

        # --- Mask Processing ---
        mask_string = tf.io.read_file(mask_path_tensor)
        mask_decoded = tf.image.decode_image(mask_string, channels=1, expand_animations=False)
        mask_decoded.set_shape([None, None, 1])
        mask_resized = tf.image.resize(mask_decoded, target_size_tensor, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask_uint8 = tf.cast(mask_resized, tf.uint8)
        
        target_max_mask_value = num_classes_tensor - 1 
        current_max_mask_val = tf.reduce_max(mask_uint8)
        
        def normalize_mask_branch(m):
            return tf.cast(m, tf.float32) / 255.0
        def passthrough_mask_branch(m):
            return tf.cast(m, tf.float32)

        mask_float32_normalized = tf.cond(tf.cast(current_max_mask_val, tf.int32) > target_max_mask_value,
                                          lambda: normalize_mask_branch(mask_uint8),
                                          lambda: passthrough_mask_branch(mask_uint8))
        
        # --- Depth Map Processing ---
        def load_and_process_depth_fn():
            depth_string = tf.io.read_file(depth_path_tensor)
            try:
                depth_image_decoded = tf.image.decode_png(depth_string, channels=1, dtype=tf.uint8) # Assuming 8-bit depth for now
            except tf.errors.InvalidArgumentError:
                depth_image_decoded = tf.image.decode_jpeg(depth_string, channels=1)
            depth_image_resized = tf.image.resize(depth_image_decoded, target_size_tensor, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            depth_image_float = tf.cast(depth_image_resized, tf.float32)
            norm_method = depth_prep_cfg_dict.get('normalization', 'min_max_local')
            if norm_method == 'min_max_local':
                min_val, max_val = tf.reduce_min(depth_image_float), tf.reduce_max(depth_image_float)
                denominator = max_val - min_val
                depth_normalized = tf.cond(denominator < 1e-6, lambda: tf.zeros_like(depth_image_float), lambda: (depth_image_float - min_val) / denominator)
            elif norm_method == 'fixed_range':
                fixed_min = float(depth_prep_cfg_dict.get('fixed_min_val', 0.0))
                fixed_max = float(depth_prep_cfg_dict.get('fixed_max_val', 255.0))
                denominator = fixed_max - fixed_min
                if denominator < 1e-6: denominator = 1.0 
                depth_normalized = (depth_image_float - fixed_min) / denominator
                depth_normalized = tf.clip_by_value(depth_normalized, 0.0, 1.0)
            else: 
                depth_normalized = depth_image_float / 255.0 
            # Convert single channel to 3-channel for backbone compatibility
            depth_3_channel = tf.concat([depth_normalized, depth_normalized, depth_normalized], axis=-1)
            depth_3_channel.set_shape([target_size_py[0], target_size_py[1], 3]) 
            return depth_3_channel

        def zeros_for_depth_fn():
            return tf.zeros([target_size_py[0], target_size_py[1], 3], dtype=tf.float32) # 3-channel zeros
        
        depth_input_tensor_val = tf.cond(tf.logical_and(use_depth_map_tensor, tf.strings.length(depth_path_tensor) > 0),
                                    load_and_process_depth_fn,
                                    zeros_for_depth_fn)

        # --- Point Cloud Processing ---
        def load_and_process_pc_fn():
            num_points = int(pc_prep_cfg_dict.get('num_points', 4096))
            norm_method_str = pc_prep_cfg_dict.get('normalization', 'unit_sphere')
            pc_data = tf.py_function(_load_and_preprocess_point_cloud_py_seg, 
                                     inp=[pc_path_tensor, num_points, norm_method_str], 
                                     Tout=tf.float32)
            pc_data.set_shape([num_points, 3]) 
            return pc_data

        def zeros_for_pc_fn():
            num_points = int(pc_prep_cfg_dict.get('num_points', 4096))
            return tf.zeros([num_points, 3], dtype=tf.float32)

        pc_input_tensor_val = tf.cond(tf.logical_and(use_point_cloud_tensor, tf.strings.length(pc_path_tensor) > 0),
                                 load_and_process_pc_fn,
                                 zeros_for_pc_fn)
        
        inputs_for_processing = {
            'rgb_input': image_float_for_aug,
            'depth_input': depth_input_tensor_val,
            'pc_input': pc_input_tensor_val
        }

        def augment_fn():
            if augmentation_pipeline is not None:
                return augmentation_pipeline((inputs_for_processing, mask_float32_normalized))
            else:
                squeezed_mask = tf.squeeze(mask_float32_normalized, axis=-1)
                return inputs_for_processing, squeezed_mask
        
        def no_augment_fn():
            squeezed_mask = tf.squeeze(mask_float32_normalized, axis=-1)
            return inputs_for_processing, squeezed_mask

        final_processed_inputs, final_mask = tf.cond(
            augment_tensor,
            true_fn=augment_fn,
            false_fn=no_augment_fn
        )
        
        if captured_preprocess_input_fn:
            final_processed_inputs['rgb_input'] = captured_preprocess_input_fn(final_processed_inputs['rgb_input'])
            final_processed_inputs['depth_input'] = captured_preprocess_input_fn(final_processed_inputs['depth_input'])
        
        return final_processed_inputs, final_mask

    except Exception as e: 
        # Log the error with a simpler message. tf.strings.as_string(e) is problematic.
        error_message = "Error processing segmentation sample. Check logs for details."
        tf.print("Error in load_and_preprocess_segmentation (fallback): paths=", 
                 image_path_tensor, depth_path_tensor, pc_path_tensor, mask_path_tensor, 
                 "ErrorMessage:", error_message, 
                 output_stream=sys.stderr)
        
        logger.error(f"Exception in TF graph load_and_preprocess_segmentation for {image_path_tensor}, {mask_path_tensor}: {e}", exc_info=True)
        # Fallback: return zero tensors
        dummy_image_shape = [target_size_py[0], target_size_py[1], 3]
        dummy_depth_shape = [target_size_py[0], target_size_py[1], 3]  # Updated to 3-channel
        dummy_mask_shape  = [target_size_py[0], target_size_py[1], 1]
        dummy_pc_num_points = int(pc_prep_cfg_dict.get('num_points', 4096) if pc_prep_cfg_dict else 4096)
        dummy_pc_shape = [dummy_pc_num_points, 3]

        dummy_inputs_dict = {
            'rgb_input': tf.zeros(dummy_image_shape, dtype=tf.float32),
            'depth_input': tf.zeros(dummy_depth_shape, dtype=tf.float32),
            'pc_input': tf.zeros(dummy_pc_shape, dtype=tf.float32),
        }
        dummy_mask_tensor = tf.zeros(dummy_mask_shape, dtype=tf.float32)
        # Ensure dummy mask has correct shape [H, W] instead of [H, W, 1]
        dummy_mask_tensor = tf.squeeze(dummy_mask_tensor, axis=-1)
        return dummy_inputs_dict, dummy_mask_tensor


def load_segmentation_data(config: Dict[str, Any]) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset], int, int, int, int]:
    try:
        data_cfg = config['data']
        paths_cfg = config['paths']
        model_cfg = config['model'] # Used for backbone specific preprocessing
        training_cfg = config.get('training', {}) # Get training config for debug mode check
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

        # --- Debug Sampling Logic --- 
        if is_debug_mode_active and debug_max_samples is not None and isinstance(debug_max_samples, int):
            if 0 < debug_max_samples < len(metadata):
                logger.info(f"SEGMENTATION DEBUG MODE: Limiting to {debug_max_samples} samples out of {len(metadata)} total.")
                np.random.seed(random_seed) # Ensure consistent shuffle for debug
                np.random.shuffle(metadata)
                metadata = metadata[:debug_max_samples]
            else:
                logger.warning(f"SEGMENTATION DEBUG MODE: debug_max_samples ({debug_max_samples}) is invalid or not smaller than total samples ({len(metadata)}). Using full dataset for debug run.")
        elif is_debug_mode_active:
            logger.info("SEGMENTATION DEBUG MODE: runtime_is_debug_mode is True, but debug_max_samples not set or invalid in data_config. Using full dataset for debug run.")

        # Prepare file paths and labels
        all_rgb_paths, all_depth_paths, all_pc_paths, all_mask_paths, all_labels = [], [], [], [], []
        # Iterate directly over the list of items (dictionaries)
        for item_data in metadata:
            if not isinstance(item_data, dict):
                logger.warning(f"Skipping item due to unexpected data format: {item_data}")
                continue
            
            # Construct full RGB path relative to metadata file's directory
            rgb_rel_path = item_data.get('image_path') # Corrected key from 'rgb_image_path'
            if not rgb_rel_path:
                logger.warning(f"Skipping item due to missing 'image_path'.") # Corrected key in warning
                continue
            full_rgb_path = str(metadata_dir / rgb_rel_path)

            # Construct full mask path relative to metadata file's directory
            mask_rel_path = item_data.get('mask_path')
            if not mask_rel_path:
                logger.warning(f"Skipping item due to missing 'mask_path'.")
                continue
            full_mask_path = str(metadata_dir / mask_rel_path)

            if not os.path.exists(full_rgb_path):
                logger.warning(f"RGB image file not found, skipping: {full_rgb_path}")
                continue
            if not os.path.exists(full_mask_path):
                logger.warning(f"Mask file not found, skipping: {full_mask_path}")
                continue

            all_rgb_paths.append(full_rgb_path)
            all_mask_paths.append(full_mask_path)

            # Depth path (optional)
            depth_path_str = ""
            if use_depth:
                # Assuming depth maps are in a subfolder relative to the RGB image's folder
                rgb_parent_dir = pathlib.Path(full_rgb_path).parent
                depth_img_name = pathlib.Path(full_rgb_path).name # Use same name as RGB
                potential_depth_path = rgb_parent_dir / depth_map_dir_name / depth_img_name
                if potential_depth_path.exists():
                    depth_path_str = str(potential_depth_path)
                else:
                    # Try with common extensions if exact match fails
                    for ext in ['.png', '.jpg', '.jpeg', '.tif']:
                        potential_depth_path_ext = potential_depth_path.with_suffix(ext)
                        if potential_depth_path_ext.exists():
                            depth_path_str = str(potential_depth_path_ext)
                            break
                    if not depth_path_str:
                         logger.debug(f"Depth map not found for {full_rgb_path} at {potential_depth_path} or with common extensions. Will use zeros.")
            all_depth_paths.append(depth_path_str)

            # Point cloud path (optional)
            pc_path_str = ""
            if use_pc and pc_root_dir and pc_sampling_rate_dir and pc_suffix:
                # Example structure: pc_root_dir / food_category / image_id / sampling_rate_dir / image_id + suffix
                # This needs to align with how point clouds are actually stored.
                # Assuming image_id can be derived from rgb_rel_path (e.g., 'food_category/image_id.jpg')
                try:
                    rel_path_parts = pathlib.Path(rgb_rel_path).parts
                    if len(rel_path_parts) >= 2:
                        food_category_pc = rel_path_parts[-2] # e.g., 'salad_10'
                        image_id_pc = pathlib.Path(rgb_rel_path).stem # e.g., 'rgb_1100'
                        
                        potential_pc_path = pathlib.Path(pc_root_dir) / food_category_pc / image_id_pc / pc_sampling_rate_dir / (image_id_pc + pc_suffix)
                        
                        if potential_pc_path.exists():
                            pc_path_str = str(potential_pc_path)
                        else:
                            logger.debug(f"Point cloud not found for {full_rgb_path} at {potential_pc_path}. Will use zeros.")
                    else:
                        logger.debug(f"Could not determine food_category/image_id for PC from {rgb_rel_path}. Will use zeros.")
                except Exception as e_pc_path:
                    logger.warning(f"Error constructing PC path for {full_rgb_path}: {e_pc_path}. Will use zeros.")
            all_pc_paths.append(pc_path_str)

            # Store a dummy label (0) for each sample, as segmentation typically learns pixel-wise classes from masks.
            all_labels.append(0) 

        if not all_rgb_paths:
            logger.error("No valid data items found after checking paths.")
            return None, None, None, 0, 0, 0, 0

        logger.info(f"Loaded {len(all_rgb_paths)} items for segmentation processing.")

        # Create path tuples for tf.data.Dataset
        # (rgb_path, depth_path, pc_path, mask_path)
        path_tuples = list(zip(all_rgb_paths, all_depth_paths, all_pc_paths, all_mask_paths))
        labels_np = np.array(all_labels, dtype=np.int32) # Use numpy array for scikit-learn compatibility

        indices = list(range(len(path_tuples)))
        train_indices, val_test_indices = train_test_split(indices, train_size=split_ratios['train'], random_state=random_seed, stratify=labels_np if len(set(all_labels)) > 1 else None)
        
        # Calculate the proportion of val set within the val_test_indices subset
        val_prop_in_remainder = split_ratios['val'] / (split_ratios['val'] + split_ratios['test'])
        val_indices, test_indices = train_test_split(val_test_indices, train_size=val_prop_in_remainder, random_state=random_seed, stratify=labels_np[val_test_indices] if len(set(all_labels)) > 1 else None)

        train_paths = [path_tuples[i] for i in train_indices]
        val_paths = [path_tuples[i] for i in val_indices]
        test_paths = [path_tuples[i] for i in test_indices]

        num_train_samples = len(train_paths)
        num_val_samples = len(val_paths)
        num_test_samples = len(test_paths)

        logger.info(f"Dataset split: Train {num_train_samples}, Val {num_val_samples}, Test {num_test_samples}")

        if num_train_samples == 0:
            logger.error("No training samples after splitting. Check dataset size, debug settings, and split ratios.")
            return None, None, None, num_train_samples, num_val_samples, num_test_samples, num_classes

        captured_aug_config = aug_cfg.copy()
        model_arch = model_cfg.get('backbone', 'UNet')
        captured_preprocess_input_fn_seg = _get_segmentation_preprocess_fn(model_arch)

        augmentation_pipeline = SegmentationAugmentation(captured_aug_config)

        use_depth_map_tensor = tf.constant(use_depth, dtype=tf.bool)
        use_point_cloud_tensor = tf.constant(use_pc, dtype=tf.bool)

        def _map_fn(paths_tuple, label_dummy, augment_flag):
            return load_and_preprocess_segmentation(
                paths_tuple, 
                target_size_py, 
                target_size_tensor, 
                num_classes_tensor, 
                augment_flag,
                captured_preprocess_input_fn_seg, 
                augmentation_pipeline,
                use_depth_map_tensor,
                depth_prep_cfg, 
                use_point_cloud_tensor,
                pc_prep_cfg
            )

        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, [0]*len(train_paths))) # Dummy labels for slice structure
        train_dataset = train_dataset.shuffle(buffer_size=len(train_paths), seed=random_seed, reshuffle_each_iteration=True)
        train_dataset = train_dataset.map(lambda p, l: _map_fn(p, l, tf.constant(True)), num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, [0]*len(val_paths)))
        val_dataset = val_dataset.map(lambda p, l: _map_fn(p, l, tf.constant(False)), num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        test_dataset = None
        if test_paths:
            test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, [0]*len(test_paths)))
            test_dataset = test_dataset.map(lambda p, l: _map_fn(p, l, tf.constant(False)), num_parallel_calls=tf.data.AUTOTUNE)
            test_dataset = test_dataset.batch(batch_size)
            test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_dataset, val_dataset, test_dataset, num_train_samples, num_val_samples, num_test_samples, num_classes

    except KeyError as e:
        logger.error(f"Configuration key error: {e}. Please check your segmentation config.yaml.")
        return None, None, None, 0, 0, 0, 0
    except Exception as e:
        logger.error(f"An unexpected error occurred in load_segmentation_data: {e}", exc_info=True)
        return None, None, None, 0, 0, 0, 0