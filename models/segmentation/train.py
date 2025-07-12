
import os

# TPU-specific environment configuration
# Allow TPU library loading for universal compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TensorFlow logging
import sys
import yaml
import logging
import tensorflow as tf

# Configure TensorFlow for TPU compatibility
try:
    tf.config.experimental.enable_tensor_float_32(False)  # Disable TF32 for TPU compatibility
except AttributeError:
    # TF32 control not available in this TensorFlow version
    pass
from pathlib import Path
from datetime import datetime
import traceback
import argparse # Add argparse
import numpy as np

# Add _get_project_root function definition
def _get_project_root() -> Path:
    """Assumes this script is in Food-Detection/models/segmentation/"""
    return Path(__file__).resolve().parent.parent.parent

# Assuming data.py is in the same directory or accessible in PYTHONPATH
from data import load_segmentation_data # Use relative import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_mixed_precision_policy(config: dict, strategy: tf.distribute.Strategy):
    """Set mixed precision policy based on hardware and config."""
    training_config = config.get('training', {})
    if training_config.get('use_mixed_precision', False):
        policy_name = ''
        if isinstance(strategy, tf.distribute.TPUStrategy):
            policy_name = 'mixed_bfloat16'
            logger.info("TPU detected, using 'mixed_bfloat16' for mixed precision.")
        else:
            gpu_compatible = False
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    details = tf.config.experimental.get_device_details(gpus[0])
                    if details.get('compute_capability', (0,0))[0] >= 7:
                        gpu_compatible = True
            except Exception as e:
                logger.warning(f"Could not get GPU details for mixed precision check: {e}")

            if gpu_compatible:
                policy_name = 'mixed_float16'
                logger.info("Compatible GPU detected, using 'mixed_float16' for mixed precision.")
            else:
                logger.warning("Mixed precision enabled in config, but no compatible GPU (Compute Capability >= 7.0) or TPU found. Mixed precision will not be used effectively for GPU.")
                return

        if policy_name:
            logger.info(f"Setting mixed precision policy to '{policy_name}'.")
            try:
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy(policy_name)
                mixed_precision.set_global_policy(policy)
                logger.info(f"Mixed precision policy set. Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}")
            except Exception as e:
                logger.warning(f"Could not set mixed precision policy: {e}")
    else:
        logger.info("Mixed precision training not enabled in config.")

def initialize_strategy() -> tf.distribute.Strategy:
    """Initialize distributed strategy for TPU, GPU, or CPU."""
    import os
    import time
    
    logger.info("Initializing distributed strategy...")
    
    # Clear TensorFlow session for subprocess compatibility
    tf.keras.backend.clear_session()
    
    # Check for TPU environment variables first
    tpu_name = os.environ.get('TPU_NAME')
    if tpu_name:
        logger.info(f"TPU_NAME environment variable found: {tpu_name}")
        resolver_address = tpu_name
    else:
        resolver_address = 'local'
        logger.info("TPU_NAME not found, trying 'local' resolver")
    
    # Try TPU detection with retry mechanism
    for attempt in range(3):
        try:
            logger.info(f"TPU initialization attempt {attempt + 1}/3")
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(resolver_address)
            logger.info(f"TPU resolver created: {resolver}")
            
            tf.config.experimental_connect_to_cluster(resolver)
            logger.info("Successfully connected to TPU cluster")
            
            tf.tpu.experimental.initialize_tpu_system(resolver)
            logger.info("TPU system initialized")
            
            strategy = tf.distribute.TPUStrategy(resolver)
            logger.info(f"TPU strategy initialized with {strategy.num_replicas_in_sync} replicas")
            return strategy
            
        except Exception as e:
            logger.info(f"TPU initialization attempt {attempt + 1} failed: {e}")
            if attempt < 2:  # Don't sleep on last attempt
                time.sleep(2)
    
    # If TPU fails, try alternative resolver addresses
    if resolver_address == 'local':
        try:
            logger.info("Trying empty string resolver as fallback")
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            logger.info(f"TPU strategy initialized with fallback resolver: {strategy.num_replicas_in_sync} replicas")
            return strategy
        except Exception as e:
            logger.info(f"TPU fallback initialization failed: {e}")
    
    # Fallback to GPU/CPU
    logger.info("TPU initialization failed, falling back to GPU/CPU")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            logger.info(f"Multi-GPU strategy: {len(gpus)} GPUs")
        else:
            strategy = tf.distribute.get_strategy()
            logger.info(f"Single GPU strategy")
        return strategy
    else:
        logger.info("Using CPU strategy")
        return tf.distribute.get_strategy()

# Path to the specific config file for segmentation training
SEGMENTATION_CONFIG_PATH = os.path.join(_get_project_root(), "models", "segmentation", "config.yaml")

from tensorflow.keras.callbacks import Callback
# Import Keras applications for backbones
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, MobileNetV3Small # Add more as needed
from tensorflow.keras import layers # Explicitly import layers for clarity

def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def _test_data_loading(train_dataset: tf.data.Dataset, data_config: dict, num_batches_to_test: int = 2):
    if train_dataset is None:
        logger.error("Training dataset is None. Skipping data loading test.")
        return

    use_depth = data_config.get('use_depth_map', False)
    use_pc = data_config.get('use_point_cloud', False)

    for i, batch_data in enumerate(train_dataset.take(num_batches_to_test)):
        inputs_dict, mask_tensor = batch_data

        if not isinstance(inputs_dict, dict):
            logger.error(f"Expected inputs_dict to be a dict, but got {type(inputs_dict)}. Batch structure might be incorrect.")
            continue

        # RGB Input
        if 'rgb_input' in inputs_dict:
            rgb = inputs_dict['rgb_input']
            try:
                min_val = tf.reduce_min(rgb).numpy()
                max_val = tf.reduce_max(rgb).numpy()
                logger.info(f"  rgb_input: Shape={rgb.shape}, Dtype={rgb.dtype}, Min={min_val}, Max={max_val}")
            except Exception as e:
                logger.warning(f"  rgb_input: Shape={rgb.shape}, Dtype={rgb.dtype}, Stats unavailable: {e}")
        else:
            logger.warning("  rgb_input not found in batch.")

        # Depth Input (if enabled)
        if use_depth:
            if 'depth_input' in inputs_dict:
                depth = inputs_dict['depth_input']
                try:
                    min_val = tf.reduce_min(depth).numpy()
                    max_val = tf.reduce_max(depth).numpy()
                    logger.info(f"  depth_input: Shape={depth.shape}, Dtype={depth.dtype}, Min={min_val}, Max={max_val}")
                except Exception as e:
                    logger.warning(f"  depth_input: Shape={depth.shape}, Dtype={depth.dtype}, Stats unavailable: {e}")
            else:
                logger.warning("  depth_input is expected (use_depth_map=True) but not found in batch.")
        
        # Point Cloud Input (if enabled)
        if use_pc:
            if 'pc_input' in inputs_dict:
                pc = inputs_dict['pc_input']
                try:
                    min_val = tf.reduce_min(pc).numpy()
                    max_val = tf.reduce_max(pc).numpy()
                    logger.info(f"  pc_input: Shape={pc.shape}, Dtype={pc.dtype}, Min={min_val}, Max={max_val}")
                except Exception as e:
                    logger.warning(f"  pc_input: Shape={pc.shape}, Dtype={pc.dtype}, Stats unavailable: {e}")
            else:
                logger.warning("  pc_input is expected (use_point_cloud=True) but not found in batch.")

        # Mask Tensor
        try:
            min_val = tf.reduce_min(mask_tensor).numpy()
            max_val = tf.reduce_max(mask_tensor).numpy()
            logger.info(f"  mask_tensor: Shape={mask_tensor.shape}, Dtype={mask_tensor.dtype}, Min={min_val}, Max={max_val}")
            unique_values, _ = tf.unique(tf.reshape(mask_tensor, [-1]))
            unique_np = unique_values.numpy()
            logger.info(f"  mask_tensor: Unique values={unique_np}")
        except Exception as e:
            logger.warning(f"  mask_tensor: Shape={mask_tensor.shape}, Dtype={mask_tensor.dtype}, Stats unavailable: {e}")
        


def build_fused_encoder_decoder_model(output_channels: int, image_size: tuple, model_config: dict, data_config: dict):
    """Builds a multi-modal encoder-decoder model with parallel backbones and feature fusion.
    Args:
        output_channels: Number of output channels (e.g., 1 for binary segmentation).
        image_size: Tuple (height, width) for the input image.
        model_config: Dictionary containing model-specific configurations (e.g., backbone, dropout).
        data_config: Dictionary containing data-specific configurations (for input modalities).
    Returns:
        A Keras encoder-decoder model with fused features.
    """
    # --- Input Layers (Define with explicit names) ---
    rgb_input = layers.Input(shape=[*image_size, 3], name='rgb_input')
    depth_input = layers.Input(shape=[*image_size, 3], name='depth_input')
    
    num_points = data_config.get('modalities_preprocessing', {}).get('point_cloud',{}).get('num_points', 4096)
    pc_input = layers.Input(shape=[num_points, 3], name='pc_input')
    
    # This dictionary is for internal use to pass to the data pipeline
    input_layers_dict = {
        'rgb_input': rgb_input,
        'depth_input': depth_input,
        'pc_input': pc_input
    }
    logger.info("Created inputs for RGB, Depth, and Point Cloud.")
    


    # --- Branch 1: RGB Backbone (Pre-trained) ---
    rgb_backbone_name = model_config.get('backbone', 'EfficientNetB0')
    logger.info(f"Building RGB branch with pre-trained {rgb_backbone_name}...")
    
    def get_preprocess_fn(arch_name):
        if arch_name.lower() == 'efficientnetb0':
            from tensorflow.keras.applications.efficientnet import preprocess_input
            return preprocess_input
        elif arch_name.lower() == 'mobilenetv3small':
            from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
            return preprocess_input
        else:
            return lambda x: (x / 127.5) - 1.0

    rgb_preprocess_fn = get_preprocess_fn(rgb_backbone_name)
    rgb_processed = rgb_preprocess_fn(rgb_input)
    
    rgb_base_model = EfficientNetB0(input_tensor=rgb_processed, include_top=False, weights='imagenet')
    rgb_base_model.trainable = model_config.get('backbone_trainable', True)
    
    rgb_features = rgb_base_model.get_layer('block6a_expand_activation').output
    rgb_features = layers.GlobalAveragePooling2D(name='rgb_gap')(rgb_features)
    logger.info(f"RGB features extracted. Shape: {rgb_features.shape}")

    # --- Branch 2: Depth Backbone (Trained from Scratch) ---
    depth_backbone_name = model_config.get('depth_backbone', 'MobileNetV3Small')
    logger.info(f"Building Depth branch with {depth_backbone_name} (trained from scratch)...")
    
    depth_preprocess_fn = get_preprocess_fn(depth_backbone_name)
    depth_processed = depth_preprocess_fn(depth_input)

    depth_base_model = MobileNetV3Small(input_tensor=depth_processed, include_top=False, weights=None)
    depth_base_model._name = "depth_backbone"
    depth_base_model.trainable = True
    
    depth_features = depth_base_model.get_layer('multiply_12').output
    depth_features = layers.GlobalAveragePooling2D(name='depth_gap')(depth_features)
    logger.info(f"Depth features extracted. Shape: {depth_features.shape}")

    # --- Branch 3: Point Cloud Backbone (PointNet-style) ---
    logger.info("Building Point Cloud branch (PointNet-style)...")
    def conv_bn(x, filters):
        x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        return layers.ReLU()(x)

    pc_x = layers.LayerNormalization()(pc_input)
    pc_x = conv_bn(pc_x, 64)
    pc_x = conv_bn(pc_x, 128)
    pc_x = conv_bn(pc_x, 256)
    pc_features = layers.GlobalMaxPooling1D(name='pc_gmp')(pc_x)
    logger.info(f"Point Cloud features extracted. Shape: {pc_features.shape}")
    
    # --- Fusion ---
    logger.info("Fusing features from all three branches...")
    
    rgb_features_norm = layers.BatchNormalization(name='rgb_feat_norm')(rgb_features)
    depth_features_norm = layers.BatchNormalization(name='depth_feat_norm')(depth_features)
    pc_features_norm = layers.BatchNormalization(name='pc_feat_norm')(pc_features)

    fused_features = layers.Concatenate(name='initial_fusion')([
        rgb_features_norm, 
        depth_features_norm, 
        pc_features_norm
    ])
    
    # --- Decoder (simpler, as we are not building a U-Net anymore) ---
    logger.info("Building decoder to reconstruct mask from fused features...")

    # Start by creating a small spatial map from the 1D vector
    x = layers.Dense(8 * 8 * 256, use_bias=False, name='decoder_start_dense')(fused_features)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((8, 8, 256))(x)

    # Upsampling blocks to get back to full resolution
    def upsample_block(x, filters, name):
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding='same', name=f"{name}_transpose")(x)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        return layers.ReLU(name=f"{name}_relu")(x)

    x = upsample_block(x, 128, name='decoder_block1') # 8x8 -> 16x16
    x = upsample_block(x, 64, name='decoder_block2')  # 16x16 -> 32x32
    x = upsample_block(x, 32, name='decoder_block3')  # 32x32 -> 64x64
    x = upsample_block(x, 16, name='decoder_block4')  # 64x64 -> 128x128
    x = upsample_block(x, 8, name='decoder_block5')   # 128x128 -> 256x256
    
    # Final output layer
    output_activation = model_config.get('output_activation', 'sigmoid')
    outputs = layers.Conv2D(output_channels, 1, padding="same", activation=output_activation, name='final_output_conv')(x)
    
    # Cast to float32 for mixed precision stability
    outputs = layers.Activation('linear', dtype='float32')(outputs)

    # Pass the dictionary of named Input tensors directly - canonical Keras approach
    model = tf.keras.Model(inputs=input_layers_dict, outputs=outputs)
    logger.info(f"Built fused encoder-decoder model with output shape: {model.output_shape}")

    # Save model summary to a file
    summary_file_path = "model_summary.txt" # Will be saved in the CWD of train.py
    try:
        with open(summary_file_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        logger.info(f"Model summary saved to {summary_file_path}")
    except Exception as e:
        logger.error(f"Could not save model summary to {summary_file_path}: {e}")

    # Log model summary if verbosity is high enough
    if logger.getEffectiveLevel() <= logging.DEBUG:
        model.summary(print_fn=logger.info)

    logger.info(f"Fused encoder-decoder model built with final activation: {output_activation}.")
    return model

# Custom metrics for segmentation
class BinaryIoU(tf.keras.metrics.Metric):
    """Binary IoU metric for segmentation."""
    
    def __init__(self, threshold=0.5, name='binary_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Handle shape mismatch: model output [batch, H, W, 1] vs mask [batch, H, W]
        if len(y_pred.shape) == 4 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)
        if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)
    
    def result(self):
        return self.intersection / (self.union + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

class DiceCoefficient(tf.keras.metrics.Metric):
    """Dice coefficient metric for segmentation."""
    
    def __init__(self, threshold=0.5, smooth=1.0, name='dice_coefficient', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.smooth = smooth
        self.dice_sum = self.add_weight(name='dice_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Handle shape mismatch: model output [batch, H, W, 1] vs mask [batch, H, W]
        if len(y_pred.shape) == 4 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)
        if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        
        # Now both should be [batch, H, W]
        reduce_axes = [1, 2]
            
        intersection = tf.reduce_sum(y_true * y_pred, axis=reduce_axes)
        union = tf.reduce_sum(y_true, axis=reduce_axes) + tf.reduce_sum(y_pred, axis=reduce_axes)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        self.dice_sum.assign_add(tf.reduce_sum(dice))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        return self.dice_sum / (self.count + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)

# Custom loss functions
def dice_loss(y_true, y_pred, smooth=1.0):
    """Dice loss for segmentation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Handle shape mismatch: model output [batch, H, W, 1] vs mask [batch, H, W]
    if len(y_pred.shape) == 4 and y_pred.shape[-1] == 1:
        y_pred = tf.squeeze(y_pred, axis=-1)
    if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1)
    
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for addressing class imbalance."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Handle shape mismatch: model output [batch, H, W, 1] vs mask [batch, H, W]
    if len(y_pred.shape) == 4 and y_pred.shape[-1] == 1:
        y_pred = tf.squeeze(y_pred, axis=-1)
    if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1)
    
    # Ensure both tensors have the same shape
    y_true = tf.ensure_shape(y_true, y_pred.shape)
    
    # Clip predictions to prevent log(0)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    
    # Compute cross entropy manually to ensure shape consistency
    ce_loss = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    
    # Compute p_t (predicted probability for the true class)
    p_t = tf.where(y_true == 1, y_pred, 1 - y_pred)
    
    # Compute alpha_t (class weighting)
    alpha_t = tf.where(y_true == 1, alpha, 1 - alpha)
    
    # Compute focal weight
    focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
    
    # Apply focal weight and reduce to scalar
    focal_loss_value = focal_weight * ce_loss
    return tf.reduce_mean(focal_loss_value)

def combined_loss(y_true, y_pred, bce_weight=0.5, dice_weight=0.3, focal_weight=0.2, 
                 label_smoothing=0.1, smooth=1.0, alpha=0.25, gamma=2.0):
    """Combined loss function for better segmentation."""
    # Handle shape mismatch: model output [batch, H, W, 1] vs mask [batch, H, W]
    if len(y_pred.shape) == 4 and y_pred.shape[-1] == 1:
        y_pred = tf.squeeze(y_pred, axis=-1)
    if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1)
    
    # Ensure both tensors have the same shape
    y_true = tf.ensure_shape(y_true, y_pred.shape)
    
    # Binary crossentropy with label smoothing
    if label_smoothing > 0:
        y_true_smooth = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
    else:
        y_true_smooth = y_true
    
    # Calculate BCE manually to ensure shape consistency
    y_pred_clipped = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    bce = -(y_true_smooth * tf.math.log(y_pred_clipped) + (1 - y_true_smooth) * tf.math.log(1 - y_pred_clipped))
    bce = tf.reduce_mean(bce)  # Reduce to scalar
    
    # Dice loss (already returns scalar)
    dice = dice_loss(y_true, y_pred, smooth=smooth)
    
    # Focal loss (already returns scalar)
    focal = focal_loss(y_true, y_pred, alpha=alpha, gamma=gamma)
    
    return bce_weight * bce + dice_weight * dice + focal_weight * focal

def main():
    # --- Setup (Unchanged) ---
    project_root = _get_project_root()
    strategy = initialize_strategy()
    logger.info(f"Training will use strategy: {strategy.__class__.__name__}")
    
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument('--config', type=str, default=str(project_root / 'models' / 'segmentation' / 'config.yaml'), help='Path to the configuration YAML file.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (overrides some config settings for quick testing).')
    args = parser.parse_args()

    config = load_config(args.config)
    set_mixed_precision_policy(config, strategy)

    # --- Data Loading (once, with augmentations configured) ---
    logger.info("Loading data for training...")
    train_ds, val_ds, test_ds, num_train, num_val, num_test, num_classes = load_segmentation_data(config)
    if train_ds is None:
        logger.error("Data loading failed. Exiting.")
        return

    # --- Calculate Steps ---
    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})
    optimizer_cfg = config.get('optimizer', {})
    loss_cfg = config.get('loss', {})
    
    per_replica_batch_size = data_cfg.get('batch_size', 16)
    steps_per_epoch = num_train // per_replica_batch_size
    validation_steps = num_val // per_replica_batch_size if val_ds else None

    # Enhanced Loss Function Configuration
    loss_fn_name = loss_cfg.get('name', 'binary_crossentropy').lower()
    
    if loss_fn_name == 'binary_crossentropy': 
        model_final_activation = config.get('model', {}).get('activation', 'sigmoid')
        label_smoothing = loss_cfg.get('label_smoothing', 0.0)
        loss_function = tf.keras.losses.BinaryCrossentropy(
            from_logits=(model_final_activation != 'sigmoid'),
            label_smoothing=label_smoothing
        )
    elif loss_fn_name == 'combined_loss':
        bce_config = loss_cfg.get('binary_crossentropy', {})
        dice_config = loss_cfg.get('dice_loss', {})
        focal_config = loss_cfg.get('focal_loss', {})
        
        bce_weight = bce_config.get('weight', 0.5)
        dice_weight = dice_config.get('weight', 0.3)
        focal_weight = focal_config.get('weight', 0.2)
        label_smoothing = bce_config.get('label_smoothing', 0.1)
        smooth = dice_config.get('smooth', 1.0)
        alpha = focal_config.get('alpha', 0.25)
        gamma = focal_config.get('gamma', 2.0)
        
        loss_function = lambda y_true, y_pred: combined_loss(
            y_true, y_pred, 
            bce_weight=bce_weight, 
            dice_weight=dice_weight, 
            focal_weight=focal_weight,
            label_smoothing=label_smoothing,
            smooth=smooth,
            alpha=alpha,
            gamma=gamma
        )
        logger.info(f"Using combined loss: BCE({bce_weight}) + Dice({dice_weight}) + Focal({focal_weight})")
    else:
        logger.warning(f"Unsupported loss function: {loss_fn_name}. Defaulting to binary crossentropy.")
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # Enhanced Metrics Configuration
    metrics_cfg = training_cfg.get('metrics', ['binary_accuracy'])
    metrics_list = []
    for m_name in metrics_cfg:
        m_name_lower = m_name.lower()
        if m_name_lower == 'accuracy' or m_name_lower == 'binary_accuracy':
            metrics_list.append('binary_accuracy')
        elif m_name_lower == 'binary_iou':
            metrics_list.append(BinaryIoU(threshold=0.5, name='binary_iou'))
        elif m_name_lower == 'dice_coefficient':
            metrics_list.append(DiceCoefficient(threshold=0.5, name='dice_coefficient'))
        elif m_name_lower == 'precision':
            metrics_list.append(tf.keras.metrics.Precision(name='precision'))
        elif m_name_lower == 'recall':
            metrics_list.append(tf.keras.metrics.Recall(name='recall'))

    logger.info(f"Using metrics: {[m.name if hasattr(m, 'name') else str(m) for m in metrics_list]}")

    # =================================================================
    #                  STAGE 1: Pre-train New Branches
    # =================================================================
    logger.info("\n" + "="*60 + "\n=== STAGE 1: Pre-training Depth & Point Cloud Branches ===\n" + "="*60)
    
    with strategy.scope():
        model = build_fused_encoder_decoder_model(
            output_channels=num_classes, 
            image_size=tuple(data_cfg.get('image_size')), 
            model_config=config.get('model', {}), 
            data_config=data_cfg
        )
        
        # Freeze the pre-trained RGB backbone
        backbone_name_from_config = config.get('model', {}).get('backbone', 'efficientnetb0').lower()
        for layer in model.layers:
            if layer.name == backbone_name_from_config:
                layer.trainable = False
                logger.info(f"Freezing RGB Backbone: {layer.name}")
                break

        # Compile for Stage 1
        stage1_lr = optimizer_cfg.get('stage1_learning_rate', 1e-3)
        optimizer_stage1 = tf.keras.optimizers.AdamW(learning_rate=stage1_lr)
        model.compile(optimizer=optimizer_stage1, loss=loss_function, metrics=metrics_list)
        logger.info(f"Model compiled for Stage 1 with LR: {stage1_lr}")
    
    stage1_epochs = training_cfg.get('stage1_epochs', 5)
    logger.info(f"--- Starting Stage 1 training for {stage1_epochs} epochs ---")
    
    # Setup callbacks for Stage 1
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_dir_rel = config.get('paths', {}).get('model_save_dir', 'trained_models/segmentation')
    model_dir_abs = project_root / model_save_dir_rel
    model_dir_abs.mkdir(parents=True, exist_ok=True)
    
    stage1_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir_abs / f'stage1_best_{timestamp}.h5'),
            monitor='val_binary_iou',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_binary_iou',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=stage1_callbacks
    )

    # =================================================================
    #                  STAGE 2: Fine-tune Entire Model
    # =================================================================
    logger.info("\n" + "="*60 + "\n=== STAGE 2: Fine-tuning all branches together ===\n" + "="*60)
    
    with strategy.scope():
        # Unfreeze all layers for fine-tuning
        for layer in model.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
        logger.info("All model layers UNFROZEN for Stage 2.")
            
        # Compile with a very low learning rate
        stage2_lr = optimizer_cfg.get('stage2_learning_rate', 1e-5)
        optimizer_stage2 = tf.keras.optimizers.AdamW(learning_rate=stage2_lr)
        model.compile(optimizer=optimizer_stage2, loss=loss_function, metrics=metrics_list)
        logger.info(f"Model re-compiled for Stage 2 with LR: {stage2_lr}")

    total_epochs = training_cfg.get('epochs', 30)
    stage2_epochs = total_epochs - stage1_epochs
    logger.info(f"--- Starting Stage 2 training for {stage2_epochs} epochs ---")
    
    # Setup callbacks for Stage 2
    stage2_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir_abs / f'stage2_best_{timestamp}.h5'),
            monitor='val_binary_iou',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_binary_iou',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_binary_iou',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=stage1_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=stage2_callbacks
    )

    logger.info("🎉 Training finished.")
    
    # Save the final model
    final_model_path = model_dir_abs / f'fused_segmentation_final_{timestamp}.h5'
    model.save(str(final_model_path)) 
    logger.info(f"Final trained model saved to: {final_model_path}")

    # Evaluate on test set if available
    if test_ds and num_test > 0:
        logger.info("Evaluating model on the test set...")
        test_results = model.evaluate(test_ds, verbose=1)
        logger.info("Test Set Evaluation Results:")
        if isinstance(test_results, list):
            for metric_name, value in zip(model.metrics_names, test_results):
                logger.info(f"  {metric_name}: {value}")
        else:
            logger.info(f"  loss: {test_results}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("Exception in main execution:", exc_info=True)
        # Ensure the traceback is printed to stderr as well for visibility
        traceback.print_exc() 
        sys.exit(1) # Exit with a non-zero code to indicate failure