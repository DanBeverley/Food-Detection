
import yaml
import argparse
import logging
import os
import json
import traceback
from datetime import datetime
from pathlib import Path # Added import

# TPU-specific imports and configuration
# Allow TPU library loading in subprocess context
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TensorFlow logging

import tensorflow as tf

# Configure GPU memory growth before any TensorFlow operations
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth configuration failed: {e}")

from tensorflow.keras import models, layers, optimizers, losses, callbacks, applications, metrics
from tensorflow.keras.applications import mobilenet_v2, mobilenet_v3, efficientnet_v2, convnext
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K

class DebugCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batch_count = 0
        
    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % 100 == 0:
            current_loss = logs.get('loss', 0)
            current_acc = logs.get('categorical_accuracy', 0)
            logger.info(f"Batch {self.batch_count}: Loss={current_loss:.4f}, Acc={current_acc:.4f}")
            
            # Check learning rate more robustly
            try:
                # K.get_value is safer for retrieving tensor values
                lr_val = K.get_value(self.model.optimizer.learning_rate)
                logger.info(f"Learning Rate: {float(lr_val)}")
            except Exception as e:
                logger.warning(f"Could not retrieve learning rate: {e}")

from typing import Dict, Tuple, Any, List, Optional

# Configure TensorFlow for TPU compatibility
try:
    tf.config.experimental.enable_tensor_float_32(False)  # Disable TF32 for TPU compatibility
except AttributeError:
    # TF32 control not available in this TensorFlow version
    pass

logger = logging.getLogger(__name__)

def diagnose_tpu_environment():
    """Diagnose TPU environment and configuration"""
    import os
    
    logger.info("=== TPU Environment Diagnostics ===")
    
    # Check environment variables
    tpu_vars = ['TPU_NAME', 'TPU_LOAD_LIBRARY', 'COLAB_TPU_ADDR', 'KFAC_DEVICE']
    for var in tpu_vars:
        value = os.environ.get(var)
        logger.info(f"{var}: {value}")
    
    # Check TensorFlow TPU devices
    try:
        tpu_devices = tf.config.experimental.list_physical_devices('TPU')
        logger.info(f"TPU devices detected: {tpu_devices}")
    except Exception as e:
        logger.info(f"TPU device detection failed: {e}")
    
    logger.info("=== End TPU Diagnostics ===")

def initialize_strategy() -> tf.distribute.Strategy:
    import os
    import time
    
    logger.info("Initializing distributed strategy...")
    
    # Diagnose TPU environment first
    diagnose_tpu_environment()
    
    # Clear TensorFlow session for subprocess compatibility
    tf.keras.backend.clear_session()
    
    # Check for TPU environment variables first
    tpu_name = os.environ.get('TPU_NAME')
    colab_tpu_addr = os.environ.get('COLAB_TPU_ADDR')
    
    if tpu_name:
        logger.info(f"TPU_NAME environment variable found: {tpu_name}")
        resolver_address = tpu_name
    elif colab_tpu_addr:
        logger.info(f"COLAB_TPU_ADDR found: {colab_tpu_addr}")
        resolver_address = colab_tpu_addr
    else:
        # For Kaggle TPU, try empty string first (standard approach)
        resolver_address = ''
        logger.info("No TPU environment variables found, trying empty string resolver for Kaggle TPU")
    
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
    if resolver_address == '':
        try:
            logger.info("Trying 'local' resolver as fallback")
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver('local')
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            logger.info(f"TPU strategy initialized with fallback resolver: {strategy.num_replicas_in_sync} replicas")
            return strategy
        except Exception as e:
            logger.info(f"TPU fallback initialization failed: {e}")
    elif resolver_address == 'local':
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

def set_mixed_precision_policy(config: Dict, strategy: tf.distribute.Strategy):
    if config.get('training', {}).get('use_mixed_precision') is True:
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
            mixed_precision.set_global_policy(policy_name)
            policy = mixed_precision.global_policy()
            logger.info(f"Mixed precision policy set. Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}")
    else:
        logger.info("Mixed precision training not enabled in config.")

from data import load_classification_data, _get_project_root

tf.get_logger().setLevel('INFO')

logger = logging.getLogger(__name__)

class DetailedLoggingCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_message = f"Epoch {epoch+1}/{self.params['epochs']} completed."
        for metric, value in logs.items():
            log_message += f" - {metric}: {value:.4f}"
        logger.info(log_message)
        print(log_message, flush=True)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if batch % 10 == 0:
            log_message = f"Epoch {self.params['epochs']}, Batch {batch}"
            for metric, value in logs.items():
                log_message += f" - {metric}: {value:.4f}"
            logger.info(log_message)
            print(log_message, flush=True)

def build_model(num_classes: int, config: Dict, learning_rate_to_use) -> models.Model:
    # Custom model class to handle mixed precision loss computation
    class CustomModel(tf.keras.Model):
        def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
            # This method overrides the default loss computation to handle mixed precision
            
            # 1. Compute the main loss from the compiled loss function.
            # This will be float32 because of our MixedPrecisionLoss wrapper.
            main_loss = super().compute_loss(x, y, y_pred, sample_weight)

            # 2. self.losses contains the regularization penalties from the layers.
            # In mixed precision, these are float16. We must cast and sum them.
            if self.losses:
                # Cast each regularization loss to float32 before summing
                reg_loss = tf.add_n([tf.cast(loss, tf.float32) for loss in self.losses])
                # Add the float32 regularization loss to the float32 main loss
                return main_loss + reg_loss
            else:
                return main_loss
    
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    optimizer_cfg = config.get('optimizer', {})
    loss_cfg = config.get('loss', {})
    metrics_cfg = config.get('metrics', ['accuracy'])

    architecture = model_cfg.get('architecture', 'MobileNetV3Small')
    image_size_list = data_cfg.get('image_size', [224, 224])
    image_size = tuple(image_size_list)

    use_pretrained = model_cfg.get('use_pretrained_weights', True)
    fine_tune_rgb = model_cfg.get('fine_tune', False)
    fine_tune_layers_rgb = model_cfg.get('fine_tune_layers', 10)
    weights = 'imagenet' if use_pretrained else None

    # Determine if multi-modal input is configured
    modalities_cfg = data_cfg.get('modalities_config', {})
    is_multimodal_enabled = modalities_cfg.get('enabled', False)
    logger.info(f"Multi-modal input enabled: {is_multimodal_enabled}")

    active_input_layers_list = []
    all_branch_features = []

    # --- RGB Branch (always present) ---
    rgb_input_shape = (*image_size, 3)
    rgb_input_tensor = layers.Input(shape=rgb_input_shape, name='rgb_input')
    # Add to list only if multi-modal, otherwise it's the sole input.
    if is_multimodal_enabled:
        active_input_layers_list.append(rgb_input_tensor)
    logger.info(f"RGB Input: shape={rgb_input_shape}")

    logger.info(f"Building RGB branch with architecture: {architecture}")
    preprocess_fn = None
    base_model_class = None

    if architecture == "MobileNetV2":
        base_model_class = applications.MobileNetV2
        preprocess_fn = mobilenet_v2.preprocess_input
    elif architecture == "MobileNetV3Small":
        base_model_class = applications.MobileNetV3Small
        preprocess_fn = mobilenet_v3.preprocess_input
    elif architecture == "MobileNetV3Large":
        base_model_class = applications.MobileNetV3Large
        preprocess_fn = mobilenet_v3.preprocess_input
    elif architecture.startswith('EfficientNetV2'):
        if hasattr(applications, architecture):
            base_model_class = getattr(applications, architecture)
            preprocess_fn = efficientnet_v2.preprocess_input
        else:
            raise ValueError(f"Unsupported EfficientNetV2 variant: {architecture}")
    elif architecture.startswith('ConvNeXt'):
        if hasattr(applications, architecture):
            base_model_class = getattr(applications, architecture)
            preprocess_fn = convnext.preprocess_input
        else:
            raise ValueError(f"Unsupported ConvNeXt variant: {architecture}")
    else:
        raise ValueError(f"Unsupported architecture for RGB base model: {architecture}")
    
    if preprocess_fn is None:
        raise ValueError(f"Preprocess function not found for architecture: {architecture}")

    preprocessed_rgb = preprocess_fn(rgb_input_tensor)

    rgb_base_model = base_model_class(include_top=False, input_shape=rgb_input_shape, weights=weights)
    
    rgb_base_model.trainable = fine_tune_rgb
    if fine_tune_rgb and fine_tune_layers_rgb == -1:
        # Unfreeze ALL layers
        logger.info(f"RGB Fine-tuning: Unfreezing ALL layers of {architecture}.")
        for layer in rgb_base_model.layers:
            layer.trainable = True
    elif fine_tune_rgb and fine_tune_layers_rgb > 0:
        # Unfreeze only top N layers
        total_layers = len(rgb_base_model.layers)
        actual_layers_to_unfreeze = min(fine_tune_layers_rgb, total_layers)
        for layer in rgb_base_model.layers[:-actual_layers_to_unfreeze]:
            layer.trainable = False
        for layer in rgb_base_model.layers[-actual_layers_to_unfreeze:]:
            layer.trainable = True
        logger.info(f"RGB Fine-tuning: Unfreezing the top {actual_layers_to_unfreeze}/{total_layers} layers of {architecture}.")
    elif fine_tune_rgb:
        logger.info(f"RGB Fine-tuning: Unfreezing all layers of {architecture}.")
        for layer in rgb_base_model.layers:
            layer.trainable = True
    else:
        logger.info(f"RGB Feature Extraction: Freezing all layers of {architecture}.")
        for layer in rgb_base_model.layers:
            layer.trainable = False
    
    # Debug: Count trainable parameters
    trainable_count = sum([1 for layer in rgb_base_model.layers if layer.trainable])
    total_count = len(rgb_base_model.layers)
    logger.info(f"RGB backbone: {trainable_count}/{total_count} layers are trainable")

    rgb_features_map = rgb_base_model(preprocessed_rgb)

    head_config = model_cfg.get('classification_head', {})
    pooling_layer_name = head_config.get('pooling', 'GlobalAveragePooling2D')
    
    if pooling_layer_name and hasattr(layers, pooling_layer_name):
        pooled_rgb_features = getattr(layers, pooling_layer_name)(name='rgb_pool')(rgb_features_map)
    elif pooling_layer_name:
        logger.warning(f"Pooling layer '{pooling_layer_name}' for RGB branch not found in tf.keras.layers. Defaulting to GlobalAveragePooling2D.")
        pooled_rgb_features = layers.GlobalAveragePooling2D(name='rgb_gap_fallback')(rgb_features_map)
    else:
        logger.info("No pooling layer specified for RGB branch, using GlobalAveragePooling2D.")
        pooled_rgb_features = layers.GlobalAveragePooling2D(name='rgb_gap_default')(rgb_features_map)
    
    all_branch_features.append(pooled_rgb_features)

    # --- Depth Branch (conditional on multi-modal AND specific depth config) ---
    use_depth = modalities_cfg.get('depth', {}).get('enabled', False) if is_multimodal_enabled else False
    if use_depth:
        depth_input_shape_config = modalities_cfg.get('depth', {}).get('input_shape', [*image_size, 1])
        depth_input_shape = tuple(depth_input_shape_config) # Ensure it's a tuple
        depth_input_tensor = layers.Input(shape=depth_input_shape, name='depth_input')
        active_input_layers_list.append(depth_input_tensor)
        logger.info(f"Depth Input: shape={depth_input_shape}, adding Depth processing branch.")

        depth_arch_cfg = modalities_cfg.get('depth', {}).get('architecture', {})
        depth_conv_layers = depth_arch_cfg.get('conv_layers', [
            {'filters': 32, 'kernel_size': 3, 'pool': True, 'batch_norm': True},
            {'filters': 64, 'kernel_size': 3, 'pool': True, 'batch_norm': True}
        ])
        depth_pooling_type = depth_arch_cfg.get('pooling', 'GlobalAveragePooling2D')

        depth_x = depth_input_tensor
        # Pre-process depth: potentially scale to [0,1] or repeat channels if necessary for pretrained models expecting 3 channels
        # For a custom CNN, ensure it's appropriate. Example: If depth_map is not [0,1], normalize it.
        # Example: depth_x = layers.Rescaling(1./255)(depth_input_tensor) if values are 0-255
        # If a pretrained model adapted for depth were used, it might expect 3 channels:
        # if depth_arch_cfg.get('repeat_channels_for_pretrained', False):
        #     logger.info("Repeating depth channel 3 times for potentially pretrained model input.")
        #     depth_x = layers.Concatenate(axis=-1)([depth_x, depth_x, depth_x])

        for i, layer_params in enumerate(depth_conv_layers):
            depth_x = layers.Conv2D(layer_params['filters'], 
                                    kernel_size=layer_params.get('kernel_size', 3), 
                                    padding='same', 
                                    activation='relu', name=f'depth_conv{i+1}')(depth_x)
            if layer_params.get('batch_norm', False):
                depth_x = layers.BatchNormalization(name=f'depth_bn{i+1}')(depth_x)
            if layer_params.get('pool', False):
                depth_x = layers.MaxPooling2D((2, 2), name=f'depth_pool{i+1}')(depth_x)
        
        if hasattr(layers, depth_pooling_type):
            pooled_depth_features = getattr(layers, depth_pooling_type)(name='depth_pool')(depth_x)
        else:
            logger.warning(f"Depth pooling layer '{depth_pooling_type}' not found. Defaulting to GlobalAveragePooling2D.")
            pooled_depth_features = layers.GlobalAveragePooling2D(name='depth_gap_fallback')(depth_x)

        all_branch_features.append(pooled_depth_features)
        logger.info(f"Depth branch added with {len(depth_conv_layers)} conv blocks and {depth_pooling_type}.")
    elif is_multimodal_enabled and modalities_cfg.get('depth', {}).get('enabled', False):
        logger.info("Depth modality is configured but 'use_depth_map' (old flag) or specific 'depth.enabled' is effectively false. Depth branch NOT added.")

    # --- Point Cloud Branch (conditional on multi-modal AND specific PC config) ---
    use_pc = modalities_cfg.get('point_cloud', {}).get('enabled', False) if is_multimodal_enabled else False
    if use_pc:
        pc_cfg = modalities_cfg.get('point_cloud', {})
        num_points = pc_cfg.get('num_points', 1024)
        pc_input_shape = (num_points, 3) # (Num points, 3 coords)
        pc_input_tensor = layers.Input(shape=pc_input_shape, name='point_cloud_input')
        active_input_layers_list.append(pc_input_tensor)
        logger.info(f"Point Cloud Input: shape={pc_input_shape}, num_points={num_points}. Adding Point Cloud processing branch.")

        pc_arch_cfg = pc_cfg.get('architecture', {})
        pc_conv1d_layers = pc_arch_cfg.get('conv1d_layers', [
            {'filters': 64, 'kernel_size': 1, 'batch_norm': True},
            {'filters': 128, 'kernel_size': 1, 'batch_norm': True},
            {'filters': pc_arch_cfg.get('bottleneck_size', 256), 'kernel_size': 1, 'batch_norm': False} # Match PointNet-like global feature size
        ])
        pc_pooling_type = pc_arch_cfg.get('pooling', 'GlobalMaxPooling1D') # PointNet uses MaxPooling

        pc_x = pc_input_tensor
        for i, layer_params in enumerate(pc_conv1d_layers):
            pc_x = layers.Conv1D(layer_params['filters'], 
                                kernel_size=layer_params.get('kernel_size', 1), 
                                activation='relu', name=f'pc_conv1d_{i+1}')(pc_x)
            if layer_params.get('batch_norm', True):
                pc_x = layers.BatchNormalization(name=f'pc_bn_{i+1}')(pc_x)

        if hasattr(layers, pc_pooling_type):
            pooled_pc_features = getattr(layers, pc_pooling_type)(name='pc_pool')(pc_x)
        else:
            logger.warning(f"Point cloud pooling layer '{pc_pooling_type}' not found. Defaulting to GlobalMaxPooling1D.")
            pooled_pc_features = layers.GlobalMaxPooling1D(name='pc_gmp_fallback')(pc_x)

        all_branch_features.append(pooled_pc_features)
        logger.info(f"Point Cloud branch added with {len(pc_conv1d_layers)} conv1d blocks and {pc_pooling_type}.")
    elif is_multimodal_enabled and modalities_cfg.get('point_cloud', {}).get('enabled', False):
        logger.info("Point Cloud modality is configured but 'use_point_cloud' (old flag) or specific 'point_cloud.enabled' is effectively false. Point Cloud branch NOT added.")

    # --- Fusion and Classification Head ---
    if is_multimodal_enabled and len(all_branch_features) > 1:
        logger.info(f"Fusing {len(all_branch_features)} feature branches.")
        fused_features = layers.Concatenate(name='feature_fusion')(all_branch_features)
    elif len(all_branch_features) == 1:
        fused_features = all_branch_features[0] # Single branch, no concatenation needed
    else:
        # This case should not be reached if RGB branch is always added.
        logger.error("No feature branches were created. Cannot build model head.")
        raise ValueError("Model construction failed: No feature branches were created.")

    # Classification Head configuration from model_cfg
    head_config = model_cfg.get('classification_head', {})
    dropout_rate = head_config.get('dropout_rate', head_config.get('dropout', 0.2)) # Use 'dropout' if 'dropout_rate' not present, then default
    dense_layers_config = head_config.get('dense_layers', [{'units': 128}]) # Default to a single layer dict if not present
    default_head_activation = head_config.get('activation', 'relu') # Default activation for the head
    
    # Enhanced regularization parameters
    kernel_l2_factor = head_config.get('kernel_l2_factor', 0.001)
    activity_l2_factor = head_config.get('activity_l2_factor', 0.0)
    use_batch_norm = head_config.get('use_batch_norm', False)
    batch_norm_momentum = head_config.get('batch_norm_momentum', 0.99)

    x = fused_features
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='head_dropout')(x)
    
    for i, layer_conf_item in enumerate(dense_layers_config):
        units = None
        activation = default_head_activation
        layer_dropout = 0.0
        layer_batch_norm = use_batch_norm
        layer_l2_reg = kernel_l2_factor

        if isinstance(layer_conf_item, dict):
            units = layer_conf_item.get('units')
            activation = layer_conf_item.get('activation', default_head_activation)
            layer_dropout = layer_conf_item.get('dropout', 0.0)
            layer_batch_norm = layer_conf_item.get('batch_norm', use_batch_norm)
            layer_l2_reg = layer_conf_item.get('l2_regularization', kernel_l2_factor)
        elif isinstance(layer_conf_item, int):
            units = layer_conf_item
            # Other parameters remain default
        else:
            logger.warning(f"Invalid item in dense_layers configuration: {layer_conf_item}. Skipping.")
            continue

        if units:
            # Add L2 regularization to dense layer
            regularizer = None
            if layer_l2_reg > 0:
                regularizer = tf.keras.regularizers.l2(layer_l2_reg)
            
            # Add activity regularization if specified
            activity_regularizer = None
            if activity_l2_factor > 0:
                activity_regularizer = tf.keras.regularizers.l2(activity_l2_factor)
            
            x = layers.Dense(
                units, 
                activation=activation, 
                kernel_regularizer=regularizer,
                activity_regularizer=activity_regularizer,
                name=f'head_dense_{i+1}'
            )(x)
            
            # Add batch normalization if enabled
            if layer_batch_norm:
                x = layers.BatchNormalization(momentum=batch_norm_momentum, name=f'head_bn_{i+1}')(x)
            
            # Add layer-specific dropout
            if layer_dropout > 0:
                x = layers.Dropout(layer_dropout, name=f'head_dropout_{i+1}')(x)
        else:
            logger.warning(f"No units specified for dense layer {i+1} in classification_head (item: {layer_conf_item}). Skipping layer.")

    output_activation = loss_cfg.get('output_activation', head_config.get('final_activation', 'softmax')) # Use final_activation from head if specified
    if num_classes == 1 and output_activation == 'softmax':
        # For binary classification (num_classes=1), sigmoid is typical with BinaryCrossentropy
        # If loss is CategoricalCrossentropy with num_classes=1, it's unusual but possible (one-hot encoded binary)
        # If truly num_classes=1 means a single output neuron for regression-like or custom tasks, softmax is wrong.
        logger.warning("num_classes is 1 and output_activation is 'softmax'. Consider 'sigmoid' for binary classification.")
    elif num_classes > 1 and output_activation == 'sigmoid':
        logger.warning(f"num_classes is {num_classes} and output_activation is 'sigmoid'. Consider 'softmax' for multi-class classification.")

    # Final output layer with regularization
    output_regularizer = None
    if kernel_l2_factor > 0:
        output_regularizer = tf.keras.regularizers.l2(kernel_l2_factor)
    
    outputs = layers.Dense(
        num_classes, 
        activation=output_activation, 
        kernel_regularizer=output_regularizer,
        name='output_layer'
    )(x)
    
    # Cast output to float32 for loss computation when using mixed precision
    training_cfg = config.get('training', {})
    if training_cfg.get('use_mixed_precision') is True:
        # Use explicit float32 casting for mixed precision compatibility
        outputs = layers.Activation('linear', dtype=tf.float32, name='cast_to_float32')(outputs)

    # Determine model inputs
    if is_multimodal_enabled:
        model_inputs = active_input_layers_list
        if not model_inputs: # Should have at least RGB if multimodal was intended
             logger.error("Multimodal enabled, but no input layers were configured in active_input_layers_list. Defaulting to RGB only.")
             model_inputs = rgb_input_tensor # Fallback, though this state indicates config issue
    else:
        model_inputs = rgb_input_tensor # Single input: RGB only

    # Custom model class to handle mixed precision regularization loss dtype conflicts
    class CustomModel(tf.keras.Model):
        def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
            # The issue: Keras' parent compute_loss() tries to sum mixed dtype tensors
            # Solution: Use the actual loss function directly, handle regularization separately
            
            if y is not None and y_pred is not None:
                # 1. Cast inputs to float32 for consistent dtype
                y_true = tf.cast(y, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                
                # 2. Get the actual loss function from the compiled loss object
                # This avoids recursion by not calling compute_loss methods
                loss_fn = self.loss  # This should be our MixedPrecisionLoss wrapper
                if loss_fn is not None:
                    main_loss = loss_fn(y_true, y_pred)
                    if sample_weight is not None:
                        main_loss = main_loss * tf.cast(sample_weight, tf.float32)
                    main_loss = tf.reduce_mean(tf.cast(main_loss, tf.float32))
                else:
                    # Fallback to categorical crossentropy
                    main_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
                    main_loss = tf.reduce_mean(tf.cast(main_loss, tf.float32))
                
                # 3. Handle regularization losses separately
                if self.losses:
                    # Cast float16 regularization losses to float32 and sum them
                    reg_loss = tf.add_n([tf.cast(loss, tf.float32) for loss in self.losses])
                    return tf.cast(main_loss + reg_loss, tf.float32)
                
                return main_loss
            else:
                # Fallback case
                return tf.constant(0.0, dtype=tf.float32)

    # Check if we need to use CustomModel for mixed precision
    training_cfg = config.get('training', {})
    if training_cfg.get('use_mixed_precision') is True:
        model = CustomModel(inputs=model_inputs, outputs=outputs)
        logger.info("Created CustomModel for mixed precision training")
    else:
        model = models.Model(inputs=model_inputs, outputs=outputs)
        logger.info("Created standard Model")

    # Optimizer setup
    optimizer_name = optimizer_cfg.get('name', 'AdamW')  # Default to AdamW
    clipnorm = optimizer_cfg.get('clipnorm', None)
    clipvalue = optimizer_cfg.get('clipvalue', None)
    weight_decay = optimizer_cfg.get('weight_decay', 0.01)  # Default weight decay

    if clipnorm is not None and clipvalue is not None:
        logger.warning("Both clipnorm and clipvalue are specified for the optimizer. It's usually one or the other. Clipnorm will be preferred if supported, otherwise behavior depends on optimizer.")

    # Create optimizer with enhanced parameters
    if optimizer_name.lower() == 'adamw':
        optimizer_instance = optimizers.AdamW(
            learning_rate=learning_rate_to_use, 
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue
        )
    elif optimizer_name.lower() == 'adam':
        optimizer_instance = optimizers.Adam(
            learning_rate=learning_rate_to_use,
            clipnorm=clipnorm,
            clipvalue=clipvalue
        )
    elif optimizer_name.lower() == 'sgd':
        momentum = optimizer_cfg.get('momentum', 0.9)
        optimizer_instance = optimizers.SGD(
            learning_rate=learning_rate_to_use, 
            momentum=momentum,
            clipnorm=clipnorm,
            clipvalue=clipvalue
        )
    else:
        logger.warning(f"Unsupported optimizer: {optimizer_name}. Defaulting to AdamW.")
        optimizer_instance = optimizers.AdamW(
            learning_rate=learning_rate_to_use, 
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue
        )
    
    # Apply loss scaling for mixed precision  
    training_cfg = config.get('training', {})
    if training_cfg.get('use_mixed_precision') is True:
        optimizer_instance = mixed_precision.LossScaleOptimizer(optimizer_instance)
        logger.info("Applied loss scaling for mixed precision training")
    
    # Log gradient clipping configuration
    if clipnorm is not None:
        logger.info(f"Applied gradient clipping with clipnorm={clipnorm}")
    elif clipvalue is not None:
        logger.info(f"Applied gradient clipping with clipvalue={clipvalue}")
    
    logger.info(f"Created optimizer: {optimizer_name} with learning_rate={learning_rate_to_use}, weight_decay={weight_decay}")
    
    # Loss function setup
    loss_function_name = loss_cfg.get('name', 'CategoricalCrossentropy')
    loss_params = loss_cfg.get('params', {})
    if 'from_logits' not in loss_params and output_activation != 'linear' and loss_function_name not in ['SparseCategoricalCrossentropy']:
        # For most losses like CategoricalCrossentropy, BinaryCrossentropy, if output layer has activation, from_logits should be False.
        # SparseCategoricalCrossentropy handles logits internally based on its setup.
        # For custom losses, user needs to be aware.
        loss_params['from_logits'] = False 
        logger.info(f"Setting from_logits=False for loss {loss_function_name} as output layer has activation '{output_activation}'.")
    elif 'from_logits' not in loss_params and output_activation == 'linear':
        loss_params['from_logits'] = True
        logger.info(f"Setting from_logits=True for loss {loss_function_name} as output layer has 'linear' activation.")

    selected_loss = _get_loss_function(loss_function_name, loss_params, num_classes, config)
    
    # Metrics setup
    compiled_metrics = _get_metrics(metrics_cfg, num_classes, loss_cfg.get('multilabel', False), config)

    model.compile(optimizer=optimizer_instance, loss=selected_loss, metrics=compiled_metrics)
    
    logger.info(f"Model compiled with optimizer: {optimizer_name}, loss: {loss_function_name}, metrics: {[m.name if hasattr(m, 'name') else str(m) for m in compiled_metrics]}.")
    
    # Check trainable parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    if trainable_params == 0:
        logger.error("Model has 0 trainable parameters - all layers are frozen")
        return
    
    # Test forward pass
    logger.info("Testing forward pass...")
    try:
        dummy_batch = tf.random.normal((2, 224, 224, 3))
        dummy_output = model(dummy_batch, training=False)
        logger.info(f"Forward pass successful. Output shape: {dummy_output.shape}")
        
        # Check if output is always the same (indicating frozen model)
        dummy_output2 = model(dummy_batch, training=False)
        outputs_identical = tf.reduce_all(tf.equal(dummy_output, dummy_output2))
        logger.info(f"Repeated calls identical: {outputs_identical}")
        
        # CRITICAL: Check optimizer learning rate
        actual_lr = model.optimizer.learning_rate
        if hasattr(actual_lr, 'numpy'):
            actual_lr_value = actual_lr.numpy()
        else:
            actual_lr_value = actual_lr
        logger.info(f"Optimizer LR: {actual_lr_value}, Type: {type(model.optimizer).__name__}")
        
        if actual_lr_value < 1e-8:
            logger.error(f"CRITICAL: Learning rate is too small: {actual_lr_value}")
        
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        return
    
    # Optionally print model summary
    if model_cfg.get('print_summary', True):
        model.summary(print_fn=logger.info)
        
    return model


def _create_optimizer(optimizer_name: str, learning_rate: float, clipnorm: Optional[float] = None, clipvalue: Optional[float] = None) -> optimizers.Optimizer:
    if optimizer_name.lower() == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'adamw':
        weight_decay = 0.01  # Default weight decay for AdamW
        optimizer = optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        momentum = 0.9 # Common default for SGD
        optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        logger.warning(f"Unsupported optimizer: {optimizer_name}. Defaulting to AdamW.")
        optimizer = optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01)

    if clipnorm is not None:
        optimizer = optimizers.ClipNormOptimizer(optimizer, clipnorm=clipnorm)
        logger.info(f"Applied gradient clipping with clipnorm={clipnorm}")
    elif clipvalue is not None:
        optimizer = optimizers.ClipValueOptimizer(optimizer, clipvalue=clipvalue)
        logger.info(f"Applied gradient clipping with clipvalue={clipvalue}")

    return optimizer


def _get_loss_function(loss_function_name: str, loss_params: Dict, num_classes: int, config: Dict) -> losses.Loss:
    # Check if mixed precision is enabled
    training_cfg = config.get('training', {})
    reduction = tf.keras.losses.Reduction.AUTO
    if training_cfg.get('use_mixed_precision') is True:
        # Use SUM_OVER_BATCH_SIZE for mixed precision compatibility
        reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    
    if loss_function_name.lower() == 'categoricalcrossentropy' or loss_function_name.lower() == 'categorical_crossentropy':
        # Create a custom loss wrapper for mixed precision compatibility
        base_loss = losses.CategoricalCrossentropy(
            label_smoothing=loss_params.get('label_smoothing', 0.0),
            from_logits=loss_params.get('from_logits', False), # Usually False if softmax is last layer
            reduction=reduction
        )
        
        if training_cfg.get('use_mixed_precision') is True:
            # Wrap loss to ensure dtype compatibility
            class MixedPrecisionLoss(tf.keras.losses.Loss):
                def __init__(self, base_loss):
                    super().__init__(name=base_loss.name, reduction=base_loss.reduction)
                    self.base_loss = base_loss
                
                def call(self, y_true, y_pred):
                    # Ensure both tensors are float32 for loss computation
                    y_true = tf.cast(y_true, tf.float32)
                    y_pred = tf.cast(y_pred, tf.float32)
                    return self.base_loss(y_true, y_pred)
            
            loss_instance = MixedPrecisionLoss(base_loss)
        else:
            loss_instance = base_loss
    elif loss_function_name.lower() == 'sparsecategoricalcrossentropy' or loss_function_name.lower() == 'sparse_categorical_crossentropy':
        # This branch should ideally not be hit if MixUp/CutMix are used, but handle for completeness
        base_loss = losses.SparseCategoricalCrossentropy(
            from_logits=loss_params.get('from_logits', False),
            reduction=reduction
        )
        
        if training_cfg.get('use_mixed_precision') is True:
            class MixedPrecisionSparseLoss(tf.keras.losses.Loss):
                def __init__(self, base_loss):
                    super().__init__(name=base_loss.name, reduction=base_loss.reduction)
                    self.base_loss = base_loss
                
                def call(self, y_true, y_pred):
                    y_true = tf.cast(y_true, tf.float32)
                    y_pred = tf.cast(y_pred, tf.float32)
                    return self.base_loss(y_true, y_pred)
            
            loss_instance = MixedPrecisionSparseLoss(base_loss)
        else:
            loss_instance = base_loss
    else:
        logger.error(f"Unsupported loss function: {loss_function_name}. Defaulting to CategoricalCrossentropy.")
        base_loss = losses.CategoricalCrossentropy(
            label_smoothing=loss_params.get('label_smoothing', 0.0),
            reduction=reduction
        )
        
        if training_cfg.get('use_mixed_precision') is True:
            class MixedPrecisionLoss(tf.keras.losses.Loss):
                def __init__(self, base_loss):
                    super().__init__(name=base_loss.name, reduction=base_loss.reduction)
                    self.base_loss = base_loss
                
                def call(self, y_true, y_pred):
                    y_true = tf.cast(y_true, tf.float32)
                    y_pred = tf.cast(y_pred, tf.float32)
                    return self.base_loss(y_true, y_pred)
            
            loss_instance = MixedPrecisionLoss(base_loss)
        else:
            loss_instance = base_loss

    return loss_instance


def _get_metrics(metrics_cfg: List[str], num_classes: int, multilabel: bool, config: Dict) -> List[metrics.Metric]:
    compiled_metrics = []
    for metric_name in metrics_cfg:
        if metric_name.lower() == 'accuracy':
            # If using CategoricalCrossentropy, CategoricalAccuracy is more appropriate.
            # 'accuracy' can sometimes alias to SparseCategoricalAccuracy depending on context.
            training_cfg = config.get('training', {})
            if training_cfg.get('use_mixed_precision') is True:
                compiled_metrics.append(metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=tf.float32))
            else:
                compiled_metrics.append(metrics.CategoricalAccuracy(name='categorical_accuracy'))
            logger.info("Using CategoricalAccuracy metric (aliased from 'accuracy').")
        elif metric_name.lower() == 'top_5_accuracy':
            # Skip TopK metrics when using mixed precision due to dtype incompatibility
            training_cfg = config.get('training', {})
            if training_cfg.get('use_mixed_precision') is True:
                logger.warning("Skipping top_5_accuracy metric due to mixed precision dtype incompatibility")
            else:
                compiled_metrics.append(metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'))
                logger.info("Using TopKCategoricalAccuracy metric for top_5_accuracy.")
        elif metric_name.lower() == 'top_3_accuracy':
            # Skip TopK metrics when using mixed precision due to dtype incompatibility
            training_cfg = config.get('training', {})
            if training_cfg.get('use_mixed_precision') is True:
                logger.warning("Skipping top_3_accuracy metric due to mixed precision dtype incompatibility")
            else:
                compiled_metrics.append(metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'))
                logger.info("Using TopKCategoricalAccuracy metric for top_3_accuracy.")
        elif hasattr(metrics, metric_name):
            metric_class = getattr(metrics, metric_name)
            if callable(metric_class):
                training_cfg = config.get('training', {})
                if training_cfg.get('use_mixed_precision') is True and metric_name.lower() in ['precision', 'recall']:
                    compiled_metrics.append(metric_class(dtype=tf.float32))
                else:
                    compiled_metrics.append(metric_class())
                logger.info(f"Using {metric_name} metric.")
            else:
                logger.warning(f"Metric {metric_name} is not callable. Skipping.")
        else:
            logger.warning(f"Unknown metric '{metric_name}'. Skipping.")

    return compiled_metrics


def train_model(model: models.Model, 
                train_dataset: tf.data.Dataset, 
                val_dataset: tf.data.Dataset, 
                config: Dict, 
                index_to_label_map: Dict, 
                strategy: tf.distribute.Strategy,
                epochs: int, 
                steps_per_epoch: Optional[int], 
                validation_steps: Optional[int]
                ):
    project_root = _get_project_root()
    training_cfg = config.get('training', {})
    paths_cfg = config.get('paths', {})
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})

    logger.info(f"Starting model training with {epochs} epochs.")
    if steps_per_epoch:
        logger.info(f"Steps per epoch: {steps_per_epoch}")
    if val_dataset and validation_steps:
        logger.info(f"Validation steps: {validation_steps}")

    # Callbacks
    callbacks = []
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Add debug callback to monitor gradient flow
    debug_callback = DebugCallback()
    callbacks.append(debug_callback)
    logger.info("Added debug callback to monitor gradient flow")
    
    # Model Checkpoint
    checkpoint_dir_rel = paths_cfg.get('checkpoint_dir', f'trained_models/classification/checkpoints_{timestamp}')
    checkpoint_dir = project_root / checkpoint_dir_rel
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_filename = paths_cfg.get('checkpoint_filename', 'cp-epoch-{epoch:03d}.weights.h5') 
    checkpoint_filepath = checkpoint_dir / checkpoint_filename
    
    save_best_only = training_cfg.get('callbacks', {}).get('model_checkpoint', {}).get('save_best_only', True)
    save_weights_only = training_cfg.get('callbacks', {}).get('model_checkpoint', {}).get('save_weights_only', True)
    
    # Use float32-safe accuracy metric instead of potentially float16 loss
    training_cfg_inner = config.get('training', {})
    if training_cfg_inner.get('use_mixed_precision') is True:
        monitor_metric = training_cfg.get('callbacks', {}).get('model_checkpoint', {}).get('monitor', 'val_categorical_accuracy')
    else:
        monitor_metric = training_cfg.get('callbacks', {}).get('model_checkpoint', {}).get('monitor', 'val_loss')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_filepath),
        save_weights_only=save_weights_only,
        monitor=monitor_metric,
        mode='max' if 'accuracy' in monitor_metric else 'min',
        save_best_only=save_best_only,
        verbose=1)
    callbacks.append(model_checkpoint_callback)
    logger.info(f"ModelCheckpoint configured: path='{checkpoint_filepath}', monitor='{monitor_metric}', save_best_only={save_best_only}, save_weights_only={save_weights_only}")

    # Early Stopping
    early_stopping_cfg = training_cfg.get('callbacks', {}).get('early_stopping', {})
    if early_stopping_cfg.get('enabled', True):
        # Use float32-safe accuracy metric for mixed precision
        if training_cfg_inner.get('use_mixed_precision') is True:
            early_stopping_monitor = early_stopping_cfg.get('monitor', 'val_categorical_accuracy')
        else:
            early_stopping_monitor = early_stopping_cfg.get('monitor', 'val_loss')
            
        early_stopping_patience = early_stopping_cfg.get('patience', 10)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=early_stopping_monitor,
            patience=early_stopping_patience,
            verbose=1,
            mode='max' if 'accuracy' in early_stopping_monitor else 'min',
            restore_best_weights=early_stopping_cfg.get('restore_best_weights', True))
        callbacks.append(early_stopping_callback)
        logger.info(f"EarlyStopping configured: monitor='{early_stopping_monitor}', patience={early_stopping_patience}")

    # ReduceLROnPlateau
    reduce_lr_cfg = training_cfg.get('callbacks', {}).get('reduce_lr_on_plateau', {})
    if reduce_lr_cfg.get('enabled', True):
        # Use float32-safe accuracy metric for mixed precision
        if training_cfg_inner.get('use_mixed_precision') is True:
            reduce_lr_monitor = reduce_lr_cfg.get('monitor', 'val_categorical_accuracy')
        else:
            reduce_lr_monitor = reduce_lr_cfg.get('monitor', 'val_loss')
            
        reduce_lr_factor = reduce_lr_cfg.get('factor', 0.2)
        reduce_lr_patience = reduce_lr_cfg.get('patience', 5)
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=reduce_lr_monitor,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            verbose=1,
            mode='max' if 'accuracy' in reduce_lr_monitor else 'min',
            min_lr=reduce_lr_cfg.get('min_lr', 1e-7))
        callbacks.append(reduce_lr_callback)
        logger.info(f"ReduceLROnPlateau configured: monitor='{reduce_lr_monitor}', factor={reduce_lr_factor}, patience={reduce_lr_patience}")

    # TensorBoard
    tensorboard_cfg = training_cfg.get('callbacks', {}).get('tensorboard', {})
    if tensorboard_cfg.get('enabled', True):
        log_dir_rel = paths_cfg.get('log_dir', f'logs/classification/fit_{timestamp}')
        log_dir = project_root / log_dir_rel
        log_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=tensorboard_cfg.get('histogram_freq', 1))
        callbacks.append(tensorboard_callback)
        logger.info(f"TensorBoard logging to: {log_dir}")

    # Custom Detailed Logging Callback
    detailed_logging_cfg = training_cfg.get('callbacks', {}).get('detailed_logging', {})
    if detailed_logging_cfg.get('enabled', False):
        callbacks.append(DetailedLoggingCallback(config=config, model=model, strategy=strategy))
        logger.info("DetailedLoggingCallback enabled.")

    # Load initial weights if specified
    initial_epoch = 0
    load_weights_path_str = model_cfg.get('load_weights_path', None)
    if load_weights_path_str:
        load_weights_path = project_root / load_weights_path_str
        if load_weights_path.exists():
            try:
                model.load_weights(str(load_weights_path))
                logger.info(f"Successfully loaded initial weights from: {load_weights_path}")
                # Try to infer initial_epoch from filename if pattern like 'cp-epoch-005.weights.h5'
                try:
                    epoch_num_str = Path(load_weights_path_str).stem.split('-')[-1].split('.')[0]
                    if epoch_num_str.isdigit():
                        initial_epoch = int(epoch_num_str)
                        logger.info(f"Inferred initial_epoch: {initial_epoch} from weights filename.")
                    else:
                        logger.warning(f"Could not parse epoch number from '{epoch_num_str}' in filename {load_weights_path_str}. Starting from epoch 0.")
                except (IndexError, ValueError):
                    logger.warning(f"Could not infer initial_epoch from weights filename: {load_weights_path_str}. Starting from epoch 0.")
            except Exception as e_load:
                logger.error(f"Error loading initial weights from {load_weights_path}: {e_load}. Training from scratch or with backbone weights.")
        else:
            logger.warning(f"Specified load_weights_path {load_weights_path} not found. Training from scratch or with backbone weights.")
    else:
        logger.info("No initial weights path specified. Training from scratch or with backbone's pre-trained weights.")


    # Inspect training data
    logger.info("Inspecting training data samples...")
    try:
        sample_count = 0
        batch_hashes = []
        for batch_inputs, batch_labels in train_dataset.take(3):
            sample_count += 1
            # Compute hash of batch to detect repetition
            if isinstance(batch_inputs, dict):
                batch_hash = hash(str(batch_inputs['rgb_input'].numpy().mean()))
            else:
                batch_hash = hash(str(batch_inputs.numpy().mean()))
            batch_hashes.append(batch_hash)
            
            logger.info(f"Batch {sample_count} - Input: {batch_inputs.shape if hasattr(batch_inputs, 'shape') else 'dict'}, Labels: {batch_labels.shape}")
            label_classes = tf.argmax(batch_labels, axis=1)
            unique_labels = tf.unique(label_classes)[0]
            logger.info(f"Batch {sample_count} - Label classes: {len(unique_labels)} unique")
            
            if len(unique_labels) == 1:
                logger.warning(f"All labels in batch {sample_count} are identical (class: {unique_labels[0].numpy()})")
            
            label_sum = tf.reduce_sum(label_classes)
            logger.info(f"Batch {sample_count} - Label sum: {label_sum.numpy()}")
        
        if len(set(batch_hashes)) == 1:
            logger.error("All batches have identical hashes - data pipeline may be stuck")
        else:
            logger.info("Batches have different hashes - data pipeline OK")
            
    except Exception as e:
        logger.error(f"Error inspecting training data: {e}")

    # Debug tensor dtypes before training
    for batch in train_dataset.take(1):
        inputs, targets = batch
        logger.info(f"Dataset inputs dtype: {inputs.dtype}")
        logger.info(f"Dataset targets dtype: {targets.dtype}")
        break
    
    logger.info("Starting model.fit()...")
    
    # Enhanced logging for dtype debugging
    try:
        logger.info("=== DTYPE DEBUGGING INFO ===")
        logger.info(f"Mixed precision policy: {tf.keras.mixed_precision.global_policy().name}")
        logger.info(f"Model input dtype: {model.input.dtype}")
        logger.info(f"Model output dtype: {model.output.dtype}")
        
        # Test a single batch to isolate the error
        logger.info("Testing single batch prediction...")
        test_batch = train_dataset.take(1)
        for test_images, test_labels in test_batch:
            logger.info(f"Test batch input dtype: {test_images.dtype}")
            logger.info(f"Test batch label dtype: {test_labels.dtype}")
            
            # Test forward pass
            try:
                test_pred = model(test_images, training=False)
                logger.info(f"Test prediction dtype: {test_pred.dtype}")
                logger.info("Forward pass successful")
            except Exception as forward_error:
                logger.error(f"Forward pass failed: {forward_error}")
                raise
                
            # Test loss computation
            try:
                test_loss = model.compute_loss(test_images, test_labels, test_pred)
                logger.info(f"Test loss dtype: {test_loss.dtype}")
                logger.info("Loss computation successful")
            except Exception as loss_error:
                logger.error(f"Loss computation failed: {loss_error}")
                raise
                
            # Test metrics computation
            try:
                for metric in model.metrics:
                    if hasattr(metric, 'update_state'):
                        metric.reset_state()
                        metric.update_state(test_labels, test_pred)
                        result = metric.result()
                        if not isinstance(result, dict):
                            logger.info(f"Metric {metric.name} dtype: {result.dtype}")
                logger.info("Metrics computation successful")
            except Exception as metric_error:
                logger.error(f"Metrics computation failed: {metric_error}")
                raise
            break
        
        logger.info("=== END DTYPE DEBUGGING ===")
        
        with strategy.scope():
            history = model.fit(
                train_dataset,
                epochs=epochs,  
                validation_data=val_dataset,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch, 
                validation_steps=validation_steps, 
                verbose=training_cfg.get('verbose', 1),
                initial_epoch=initial_epoch
            )
    except Exception as e:
        logger.error("=== DETAILED ERROR ANALYSIS ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        
        # Enhanced tensor debugging
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"Full traceback:\n{tb_str}")
        
        # Try to identify the problematic tensor
        if "Mul_3" in str(e):
            logger.error("The error involves Mul_3 tensor - this is likely from:")
            logger.error("1. Learning rate scheduling")
            logger.error("2. Loss scaling operations")
            logger.error("3. Optimizer gradient calculations")
            logger.error("4. Metric calculations involving multiplication")
            
        if "float16" in str(e) and "float32" in str(e):
            logger.error("Mixed precision dtype conversion error detected")
            logger.error("Current mixed precision policy:")
            policy = tf.keras.mixed_precision.global_policy()
            logger.error(f"  Policy name: {policy.name}")
            logger.error(f"  Compute dtype: {policy.compute_dtype}")
            logger.error(f"  Variable dtype: {policy.variable_dtype}")
        
        raise
    logger.info("model.fit() finished.")

    # Save the final model
    final_model_filename = paths_cfg.get('final_model_filename', f'classification_model_final_epoch-{epochs}_{timestamp}.keras')
    final_model_save_path = project_root / paths_cfg.get('model_save_dir', 'trained_models/classification') / final_model_filename
    final_model_save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(final_model_save_path))
        logger.info(f"Final trained model saved to: {final_model_save_path}")
    except Exception as e_save:
        logger.error(f"Error saving final model to {final_model_save_path}: {e_save}")

    return history


def main(args):
    config = yaml.safe_load(Path(args.config).read_text())
    logger.info(f"Loaded configuration from: {args.config}")

    strategy = initialize_strategy()

    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    optimizer_cfg = config.get('optimizer', {})

    # --- Determine if debug mode is active --- 
    cli_debug_active = args.debug
    yaml_debug_mode_active = training_cfg.get('debug_mode', False)
    is_debug_active = cli_debug_active or yaml_debug_mode_active

    if cli_debug_active:
        logger.info("CLI --debug flag is set. Activating debug settings for data loading and training.")
    elif yaml_debug_mode_active:
        logger.info("config.yaml training.debug_mode is true. Activating debug settings.")
    
    max_samples_for_loader = None
    if is_debug_active:
        max_samples_for_loader = data_cfg.get('debug_max_total_samples', None)
        if max_samples_for_loader is not None:
            logger.info(f"DEBUG MODE: Will attempt to load at most {max_samples_for_loader} total samples.")
        else:
            logger.info("DEBUG MODE: debug_max_total_samples not set in config, loading all available samples.")

    # Load data using the function from data.py
    logger.info("Loading classification data...")
    data_result = load_classification_data(
        config=config, 
        max_samples_to_load=max_samples_for_loader
    )
    
    if data_result is None:
        logger.error("load_classification_data returned None. Exiting.")
        return
    
    train_ds, val_ds, test_ds, num_train_samples, num_val_samples, num_test_samples, class_names, num_classes = data_result
    
    if num_train_samples == 0:
        logger.error("No training samples loaded. This typically means:")
        logger.error("1. Dataset images are not accessible at the expected paths")
        logger.error("2. Metadata contains only Windows absolute paths (E:, C:, etc.)")
        logger.error("3. Dataset needs to be uploaded to Kaggle input directory")
        logger.error("4. Metadata file paths need to be updated for the current environment")
        return

    logger.info(f"Number of training samples: {num_train_samples}")
    logger.info(f"Number of validation samples: {num_val_samples}")
    logger.info(f"Number of test samples: {num_test_samples}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")

    index_to_label_map = {i: name for i, name in enumerate(class_names)}

    with strategy.scope():
        # Set mixed precision policy within strategy scope
        set_mixed_precision_policy(config, strategy)
        
        model = build_model(num_classes=num_classes, config=config, learning_rate_to_use=optimizer_cfg.get('learning_rate', 0.001))
    
    logger.info(f"Model built. Num classes: {num_classes}")

    # FIXED: Read batch_size from data config, not training config
    batch_size = data_cfg.get('batch_size', 32)
    epochs_to_run = training_cfg.get('epochs', 50)
    steps_per_epoch_val = None
    validation_steps_val = None

    if is_debug_active:
        epochs_to_run = training_cfg.get('debug_epochs', 1)
        logger.info(f"DEBUG MODE: Training for {epochs_to_run} epochs.")

    if num_train_samples > 0:
        steps_per_epoch_val = (num_train_samples + batch_size - 1) // batch_size # Ceiling division
    else:
        logger.error("num_train_samples is 0, cannot determine steps_per_epoch. Exiting.")
        return

    if val_ds and num_val_samples > 0:
        validation_steps_val = (num_val_samples + batch_size - 1) // batch_size # Ceiling division
    elif val_ds:
        logger.warning("Validation dataset exists but num_val_samples is 0. Setting validation_steps to 0.")
        validation_steps_val = 0

    logger.info(f"Effective batch_size: {batch_size}")
    logger.info(f"Calculated steps_per_epoch: {steps_per_epoch_val}")
    if val_ds:
        logger.info(f"Calculated validation_steps: {validation_steps_val}")

    # Add data pipeline validation
    logger.info(f"Expected total batches per epoch: {steps_per_epoch_val}")
    
    # Check if train_ds is None before trying to access its attributes
    if train_ds is None:
        logger.error("Training dataset is None. Cannot proceed with training.")
        return
    
    try:
        dataset_cardinality = tf.data.experimental.cardinality(train_ds).numpy()
        logger.info(f"Training dataset cardinality: {dataset_cardinality}")
        
        # Verify dataset can provide enough batches
        if dataset_cardinality != -1 and dataset_cardinality < steps_per_epoch_val:
            logger.warning(f"Dataset cardinality ({dataset_cardinality}) is less than expected steps_per_epoch ({steps_per_epoch_val}). This may cause the 'ran out of data' warning.")
            # Option 1: Use dataset cardinality as steps_per_epoch
            steps_per_epoch_val = int(dataset_cardinality)  # Ensure it's an integer
            logger.info(f"Adjusted steps_per_epoch to dataset cardinality: {steps_per_epoch_val}")
    except Exception as e:
        logger.error(f"Error accessing training dataset cardinality: {e}")
        logger.warning("Proceeding without cardinality validation.")

    history = train_model(
        model=model, 
        train_dataset=train_ds, 
        val_dataset=val_ds, 
        config=config, 
        index_to_label_map=index_to_label_map, 
        strategy=strategy,
        epochs=epochs_to_run,
        steps_per_epoch=steps_per_epoch_val,
        validation_steps=validation_steps_val
    )

    logger.info("Training process finished.")

    # Optional: Evaluate on test set after training
    if training_cfg.get('evaluate_on_test_set_after_training', True) and test_ds and num_test_samples > 0:
        logger.info("Evaluating model on the test set after final training...")
        test_steps = (num_test_samples + batch_size - 1) // batch_size
        test_loss, *test_metrics_values = model.evaluate(test_ds, steps=test_steps, verbose=1)
        logger.info(f"Test Loss: {test_loss}")
        # Assuming model.metrics_names includes 'loss' as the first element
        for metric_name, metric_value in zip(model.metrics_names[1:], test_metrics_values):
            logger.info(f"Test {metric_name}: {metric_value}")
    elif test_ds and num_test_samples == 0:
        logger.info("Test dataset is present but num_test_samples is 0. Skipping final evaluation.")
    elif not test_ds:
        logger.info("No test dataset provided. Skipping final evaluation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classification model using a YAML configuration file.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the classification model's YAML configuration file.")
    parser.add_argument('--debug', action='store_true', 
                        help='Run in debug mode (overrides some config settings for quick testing).')
    args = parser.parse_args()
    main(args) # Pass parsed args to main