print("<<<<< EXECUTING LATEST train.py - TOP OF FILE >>>>>", flush=True)

import yaml
import argparse
import logging
import os
import json
import traceback

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, callbacks, applications, metrics
from tensorflow.keras.applications import mobilenet_v2, mobilenet_v3, efficientnet_v2, convnext
from tensorflow.keras import mixed_precision

from typing import Dict, Tuple, Any, List, Optional

logger = logging.getLogger(__name__)

def initialize_strategy() -> tf.distribute.Strategy:
    try:
        tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        if tpu_resolver:
            logger.info(f'Running on TPU: {tpu_resolver.master()}')
            tf.config.experimental_connect_to_cluster(tpu_resolver)
            tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
            strategy = tf.distribute.TPUStrategy(tpu_resolver)
            logger.info(f"TPU strategy initialized with {strategy.num_replicas_in_sync} replicas.")
            return strategy
    except ValueError as e:
        logger.info(f"TPU not found or error connecting: {e}. Checking for GPUs.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during TPU initialization: {e}. Checking for GPUs.")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found GPUs: {gpus}")
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logger.info(f"Restricted TensorFlow to use GPU: {gpus[0].name} and enabled memory growth.")
        strategy = tf.distribute.get_strategy()
        logger.info(f"Using default strategy (CPU or single GPU): {strategy.__class__.__name__} with {strategy.num_replicas_in_sync} replicas.")
        return strategy
    else:
        logger.warning("No GPUs found by TensorFlow. Training will use CPU.")
        strategy = tf.distribute.get_strategy()
        logger.info(f"Using default strategy (CPU): {strategy.__class__.__name__} with {strategy.num_replicas_in_sync} replicas.")
        return strategy

def set_mixed_precision_policy(config: Dict, strategy: tf.distribute.Strategy):
    if config.get('training', {}).get('use_mixed_precision', False):
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
            if hasattr(tf.keras.mixed_precision, 'set_global_policy'):
                policy = mixed_precision.Policy(policy_name)
                mixed_precision.set_global_policy(policy)
                logger.info(f"Using tf.keras.mixed_precision.set_global_policy. Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}")
            else:
                logger.warning(f"Could not set mixed precision policy. tf.keras.mixed_precision.set_global_policy API not found. Ensure TensorFlow/Keras version is compatible.")
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
    if fine_tune_rgb and fine_tune_layers_rgb > 0:
        for layer in rgb_base_model.layers[:-fine_tune_layers_rgb]:
            layer.trainable = False
        for layer in rgb_base_model.layers[-fine_tune_layers_rgb:]:
            layer.trainable = True
        logger.info(f"RGB Fine-tuning: Unfreezing the top {fine_tune_layers_rgb} layers of {architecture}.")
    elif fine_tune_rgb:
        logger.info(f"RGB Fine-tuning: Unfreezing all layers of {architecture}.")
        for layer in rgb_base_model.layers:
            layer.trainable = True
    else:
        logger.info(f"RGB Feature Extraction: Freezing all layers of {architecture}.")
        for layer in rgb_base_model.layers:
            layer.trainable = False

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

    x = fused_features
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='head_dropout')(x)
    
    for i, layer_conf_item in enumerate(dense_layers_config):
        units = None
        activation = default_head_activation

        if isinstance(layer_conf_item, dict):
            units = layer_conf_item.get('units')
            activation = layer_conf_item.get('activation', default_head_activation)
        elif isinstance(layer_conf_item, int):
            units = layer_conf_item
            # Activation remains default_head_activation
        else:
            logger.warning(f"Invalid item in dense_layers configuration: {layer_conf_item}. Skipping.")
            continue

        if units:
            x = layers.Dense(units, activation=activation, name=f'head_dense_{i+1}')(x)
            # Optional: Add Batch Norm or further Dropout here if desired for dense layers
            # This part would need more config if we want per-dense-layer dropout configurable
            # e.g., if layer_conf_item is a dict and has 'dropout': layer_conf_item.get('dropout', 0.0)
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

    outputs = layers.Dense(num_classes, activation=output_activation, name='output_layer')(x)

    # Determine model inputs
    if is_multimodal_enabled:
        model_inputs = active_input_layers_list
        if not model_inputs: # Should have at least RGB if multimodal was intended
             logger.error("Multimodal enabled, but no input layers were configured in active_input_layers_list. Defaulting to RGB only.")
             model_inputs = rgb_input_tensor # Fallback, though this state indicates config issue
    else:
        model_inputs = rgb_input_tensor # Single input: RGB only

    model = models.Model(inputs=model_inputs, outputs=outputs)

    # Optimizer setup
    optimizer_name = optimizer_cfg.get('name', 'Adam')
    clipnorm = optimizer_cfg.get('clipnorm', None)
    clipvalue = optimizer_cfg.get('clipvalue', None)

    if clipnorm is not None and clipvalue is not None:
        logger.warning("Both clipnorm and clipvalue are specified for the optimizer. It's usually one or the other. Clipnorm will be preferred if supported, otherwise behavior depends on optimizer.")

    optimizer_instance = _create_optimizer(optimizer_name, learning_rate_to_use, clipnorm, clipvalue)
    
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
    
    # Optionally print model summary
    if model_cfg.get('print_summary', True):
        model.summary(print_fn=logger.info)
        
    return model


def _create_optimizer(optimizer_name: str, learning_rate: float, clipnorm: Optional[float] = None, clipvalue: Optional[float] = None) -> optimizers.Optimizer:
    if optimizer_name.lower() == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'adamw':
        weight_decay = 0.004 # Default from many papers
        optimizer = optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        momentum = 0.9 # Common default for SGD
        optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        logger.warning(f"Unsupported optimizer: {optimizer_name}. Defaulting to AdamW.")
        optimizer = optimizers.AdamW(learning_rate=learning_rate)

    if clipnorm is not None:
        optimizer = optimizers.ClipNormOptimizer(optimizer, clipnorm=clipnorm)
    elif clipvalue is not None:
        optimizer = optimizers.ClipValueOptimizer(optimizer, clipvalue=clipvalue)

    return optimizer


def _get_loss_function(loss_function_name: str, loss_params: Dict, num_classes: int, config: Dict) -> losses.Loss:
    if loss_function_name.lower() == 'categoricalcrossentropy' or loss_function_name.lower() == 'categorical_crossentropy':
        loss_instance = losses.CategoricalCrossentropy(
            label_smoothing=loss_params.get('label_smoothing', 0.0),
            from_logits=loss_params.get('from_logits', False) # Usually False if softmax is last layer
        )
    elif loss_function_name.lower() == 'sparsecategoricalcrossentropy' or loss_function_name.lower() == 'sparse_categorical_crossentropy':
        # This branch should ideally not be hit if MixUp/CutMix are used, but handle for completeness
        loss_instance = losses.SparseCategoricalCrossentropy(
            from_logits=loss_params.get('from_logits', False)
        )
    else:
        logger.error(f"Unsupported loss function: {loss_function_name}. Defaulting to CategoricalCrossentropy.")
        loss_instance = losses.CategoricalCrossentropy(label_smoothing=loss_params.get('label_smoothing', 0.0))

    return loss_instance


def _get_metrics(metrics_cfg: List[str], num_classes: int, multilabel: bool, config: Dict) -> List[metrics.Metric]:
    compiled_metrics = []
    for metric_name in metrics_cfg:
        if metric_name.lower() == 'accuracy':
            # If using CategoricalCrossentropy, CategoricalAccuracy is more appropriate.
            # 'accuracy' can sometimes alias to SparseCategoricalAccuracy depending on context.
            compiled_metrics.append(metrics.CategoricalAccuracy(name='categorical_accuracy'))
            logger.info("Using CategoricalAccuracy metric (aliased from 'accuracy').")
        elif hasattr(metrics, metric_name):
            compiled_metrics.append(getattr(metrics, metric_name)()) # Instantiate if it's a class name
        else:
            compiled_metrics.append(metric_name) # Assume it's a string Keras understands or a custom metric object

    return compiled_metrics


def train_model(model: models.Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, config: Dict, index_to_label_map: Dict, strategy: tf.distribute.Strategy) -> None:
    training_cfg = config.get('training', {})
    paths_cfg = config.get('paths', {})
    data_cfg = config.get('data', {}) # Get data_cfg for debug check
    project_root = _get_project_root()

    # Check for debug/subset mode
    is_debug_run = data_cfg.get('debug_max_total_samples', None) is not None
    debug_epochs = training_cfg.get('debug_epochs', 3) # Allow configuring debug epochs, default to 3

    epochs = training_cfg.get('epochs', 50)
    steps_per_epoch = training_cfg.get('steps_per_epoch', None)
    validation_steps = training_cfg.get('validation_steps', None)

    if is_debug_run:
        logger.info(f"*** Debug run detected (debug_max_total_samples is set). Overriding training parameters. ***")
        epochs = debug_epochs
        steps_per_epoch = None # Let TF iterate through the small debug dataset
        validation_steps = None  # Let TF iterate through the small debug dataset
        logger.info(f"Debug run: epochs set to {epochs}, steps_per_epoch and validation_steps set to None.")

    model_dir = paths_cfg.get('model_save_dir', 'trained_models/classification')
    log_dir_rel = paths_cfg.get('log_dir', 'logs/classification')
    label_map_filename = paths_cfg.get('label_map_filename', 'label_map.json')
    checkpoint_dir_rel = config.get('checkpoint_dir', os.path.join(model_dir, 'checkpoints'))

    project_root = _get_project_root()
    model_dir_abs = os.path.join(project_root, model_dir)
    log_dir_abs = os.path.join(project_root, log_dir_rel)
    checkpoint_dir_abs = os.path.join(project_root, checkpoint_dir_rel)
    label_map_path = os.path.join(model_dir_abs, label_map_filename)

    os.makedirs(model_dir_abs, exist_ok=True)
    os.makedirs(log_dir_abs, exist_ok=True)
    os.makedirs(checkpoint_dir_abs, exist_ok=True)
    logger.info(f"Model artifacts will be saved to: {model_dir_abs}")
    logger.info(f"TensorBoard logs will be saved to: {log_dir_abs}")

    callbacks_list = []
    callbacks_config = training_cfg.get('callbacks', {})

    callbacks_list.append(DetailedLoggingCallback())

    mc_cfg = callbacks_config.get('model_checkpoint', {})
    if mc_cfg.get('enabled', True):
        checkpoint_filename = mc_cfg.get('filename_template', 'model_epoch-{epoch:02d}_val_loss-{val_loss:.2f}.h5')
        model_checkpoint_path = os.path.join(checkpoint_dir_abs, checkpoint_filename)
        logger.info(f"ModelCheckpoints will be saved to: {model_checkpoint_path}")
        callbacks_list.append(
            callbacks.ModelCheckpoint(
                filepath=model_checkpoint_path,
                monitor=mc_cfg.get('monitor', 'val_loss'),
                mode=mc_cfg.get('mode', 'min'),
                save_best_only=mc_cfg.get('save_best_only', True),
                save_weights_only=mc_cfg.get('save_weights_only', False),
                verbose=1
            )
        )
    
    es_cfg = callbacks_config.get('early_stopping', {})
    if es_cfg.get('enabled', True):
        callbacks_list.append(
            callbacks.EarlyStopping(
                monitor=es_cfg.get('monitor', 'val_loss'),
                mode=es_cfg.get('mode', 'min'),
                patience=es_cfg.get('patience', 10),
                restore_best_weights=es_cfg.get('restore_best_weights', True),
                verbose=1
            )
        )

    rlrop_cfg = callbacks_config.get('reduce_lr_on_plateau', {})
    if rlrop_cfg.get('enabled', True):
        callbacks_list.append(
            callbacks.ReduceLROnPlateau(
                monitor=rlrop_cfg.get('monitor', 'val_loss'),
                mode=rlrop_cfg.get('mode', 'min'),
                factor=rlrop_cfg.get('factor', 0.2),
                patience=rlrop_cfg.get('patience', 5),
                min_lr=rlrop_cfg.get('min_lr', 0.00001),
                verbose=1
            )
        )
    
    tb_cfg = callbacks_config.get('tensorboard', {})
    if tb_cfg.get('enabled', True):
        tb_log_dir_rel = tb_cfg.get('log_dir', 'logs/classification_tb')
        if not os.path.isabs(tb_log_dir_rel):
            tb_log_dir_abs = os.path.join(_get_project_root(), tb_log_dir_rel)
        else:
            tb_log_dir_abs = tb_log_dir_rel
        
        os.makedirs(tb_log_dir_abs, exist_ok=True)
        logger.info(f"TensorBoard logs (within train_model) will be saved to: {tb_log_dir_abs}")
        callbacks_list.append(
            callbacks.TensorBoard(
                log_dir=tb_log_dir_abs,
                histogram_freq=tb_cfg.get('histogram_freq', 0),
                write_graph=tb_cfg.get('write_graph', True),
                update_freq=tb_cfg.get('update_freq', 'epoch')
            )
        )

    lr_scheduler_cfg = training_cfg.get('lr_scheduler', {})
    logger.info(f"Starting model training for {epochs} epochs.")
    try:
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks_list,
            verbose=2,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
        )
        logger.info("Model training completed.")

        final_model_path = os.path.join(model_dir_abs, "final_model.h5")
        model.save(final_model_path)
        logger.info(f"Final trained model saved to: {final_model_path}")

        with open(label_map_path, 'w') as f:
            json.dump(index_to_label_map, f, indent=4)
        logger.info(f"Label map saved to: {label_map_path}")

    except Exception as e:
        logger.error(f"Error during model training or saving: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError("Model training failed.") from e

def main(config_path: str):
    print("--- train.py main() started ---", flush=True) # ADDED FOR DEBUGGING
    logger.info(f"Using configuration file: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config: {e}")
        return

    strategy = initialize_strategy()

    set_mixed_precision_policy(config, strategy)

    paths_cfg = config.get('paths', {})
    project_root = _get_project_root()

    data_dir_rel = paths_cfg.get('data_dir', 'data/classification')
    data_dir_abs = os.path.join(project_root, data_dir_rel)

    metadata_filename = paths_cfg.get('metadata_filename', 'metadata.json')
    metadata_path_abs = os.path.join(data_dir_abs, metadata_filename)

    label_map_dir_rel = paths_cfg.get('label_map_dir', data_dir_rel)
    label_map_filename = paths_cfg.get('label_map_filename', 'label_map.json')
    label_map_dir_abs = os.path.join(project_root, label_map_dir_rel)
    label_map_path_abs = os.path.join(label_map_dir_abs, label_map_filename)

    logger.info(f"Expecting metadata to be read by load_classification_data using relative path from config: {paths_cfg.get('data_dir', 'data/classification')}/{paths_cfg.get('metadata_filename', 'metadata.json')}")
    logger.info(f"Expecting label map to be read by load_classification_data using relative path from config: {label_map_dir_rel}/{label_map_filename}")

    loaded_data = load_classification_data(config)
    if loaded_data is None or not all(x is not None for x in loaded_data[:5]): # Check first 5 crucial elements
        logger.error("Failed to load data. Exiting training.")
        return
    
    train_dataset, val_dataset, test_dataset, num_classes, index_to_label_map, class_weights_dict = loaded_data

    if train_dataset is None or val_dataset is None:
        logger.error("Data loading returned None for train or validation dataset. Aborting.")
        return

    logger.info(f"Successfully loaded data: {num_classes} classes.")
    if test_dataset:
        logger.info("Test dataset also loaded.")
    if class_weights_dict:
        logger.info(f"Class weights computed: {len(class_weights_dict)} entries.")

    with strategy.scope():
        model = build_model(num_classes=num_classes, config=config, learning_rate_to_use=config.get('optimizer', {}).get('learning_rate'))
        if model is None:
            logger.error("Failed to build model. Exiting training.")
            return

    train_model(model, train_dataset, val_dataset, config, index_to_label_map, strategy)

    # After training, export to TFLite if configured

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classification model using a YAML configuration file.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the classification model's YAML configuration file.")
    args = parser.parse_args()
    main(args.config)