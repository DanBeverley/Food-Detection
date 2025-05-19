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

    active_input_layers_list = []
    all_branch_features = []

    rgb_input_shape = (*image_size, 3)
    rgb_input_tensor = layers.Input(shape=rgb_input_shape, name='rgb_input')
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

    if data_cfg.get('use_depth_map', False):
        depth_input_shape = (*image_size, 1)
        depth_input_tensor = layers.Input(shape=depth_input_shape, name='depth_input')
        active_input_layers_list.append(depth_input_tensor)
        logger.info(f"Depth Input: shape={depth_input_shape}, adding Depth processing branch.")

        depth_x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='depth_conv1')(depth_input_tensor)
        depth_x = layers.BatchNormalization(name='depth_bn1')(depth_x)
        depth_x = layers.MaxPooling2D((2, 2), name='depth_pool1')(depth_x)

        depth_x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='depth_conv2')(depth_x)
        depth_x = layers.BatchNormalization(name='depth_bn2')(depth_x)
        depth_x = layers.MaxPooling2D((2, 2), name='depth_pool2')(depth_x)

        depth_x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='depth_conv3')(depth_x)
        depth_x = layers.BatchNormalization(name='depth_bn3')(depth_x)
        depth_x = layers.MaxPooling2D((2, 2), name='depth_pool3')(depth_x)
        
        processed_depth_features = layers.GlobalAveragePooling2D(name='depth_gap')(depth_x)
        all_branch_features.append(processed_depth_features)
        logger.info(f"Depth branch added with output shape: {processed_depth_features.shape}")

    if data_cfg.get('use_point_cloud', False):
        pc_config = data_cfg.get('point_cloud', {})
        num_points = pc_config.get('num_points', 4096)
        pc_input_shape = (num_points, 3)
        pc_input_tensor = layers.Input(shape=pc_input_shape, name='pc_input')
        active_input_layers_list.append(pc_input_tensor)
        logger.info(f"Point Cloud Input: shape={pc_input_shape}, adding Point Cloud processing branch.")

        pc_x = layers.Conv1D(64, 1, activation='relu', name='pc_conv1d_1')(pc_input_tensor)
        pc_x = layers.BatchNormalization(name='pc_bn_1')(pc_x)
        pc_x = layers.Conv1D(128, 1, activation='relu', name='pc_conv1d_2')(pc_x)
        pc_x = layers.BatchNormalization(name='pc_bn_2')(pc_x)
        pc_x = layers.Conv1D(256, 1, activation='relu', name='pc_conv1d_3')(pc_x)
        pc_x = layers.BatchNormalization(name='pc_bn_3')(pc_x)
        pc_x = layers.Conv1D(512, 1, activation='relu', name='pc_conv1d_4')(pc_x)
        pc_x = layers.BatchNormalization(name='pc_bn_4')(pc_x)
        processed_pc_features = layers.GlobalMaxPooling1D(name='pc_gmp')(pc_x)
        all_branch_features.append(processed_pc_features)
        logger.info(f"Point Cloud branch added with output shape: {processed_pc_features.shape}")

    if len(all_branch_features) > 1:
        logger.info(f"Fusing {len(all_branch_features)} feature branches: {[f.name for f in all_branch_features]}")
        merged_features = layers.Concatenate(name='feature_fusion')(all_branch_features)
    elif all_branch_features:
        merged_features = all_branch_features[0]
    else:
        raise ValueError("No feature branches were built. Check model and data configuration.")
    
    logger.info(f"Merged feature tensor shape: {merged_features.shape}")
    x = merged_features

    head_config = model_cfg.get('classification_head', {})
    dense_layers_units = head_config.get('dense_layers', [128])
    dropout_rate = head_config.get('dropout', 0.5)
    activation = head_config.get('activation', 'relu')
    final_activation = head_config.get('final_activation', 'softmax')
    kernel_l2_factor = head_config.get('kernel_l2_factor', 0.0)

    kernel_regularizer = None
    if kernel_l2_factor > 0:
        kernel_regularizer = tf.keras.regularizers.l2(kernel_l2_factor)
        logger.info(f"Applying L2 kernel regularization with factor: {kernel_l2_factor} to Dense layers in head.")

    for i, units in enumerate(dense_layers_units):
        x = layers.Dense(units, activation=activation, kernel_regularizer=kernel_regularizer, name=f"head_dense_{i}_{units}")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"head_dropout_{i}_{units}")(x)

    outputs = layers.Dense(num_classes, activation=final_activation, name='output_predictions')(x)

    model = models.Model(inputs=active_input_layers_list, outputs=outputs, name=f"{architecture}_multi_modal_classification")
    logger.info(f"Successfully built multi-modal model: {model.name} with {len(active_input_layers_list)} inputs.")

    optimizer_name = optimizer_cfg.get('name', 'Adam').lower()

    if optimizer_name == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate_to_use)
    elif optimizer_name == 'sgd':
        momentum = optimizer_cfg.get('momentum', 0.9)
        optimizer = optimizers.SGD(learning_rate=learning_rate_to_use, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    loss_fn_name = loss_cfg.get('name', 'sparse_categorical_crossentropy').lower()

    if loss_fn_name == 'sparse_categorical_crossentropy':
        loss_fn = losses.SparseCategoricalCrossentropy(from_logits=(final_activation != 'softmax'))
    elif loss_fn_name == 'categorical_crossentropy':
        loss_fn = losses.CategoricalCrossentropy(from_logits=(final_activation != 'softmax'))
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn_name}")

    metrics_list = []
    for m_name in metrics_cfg:
        m_name_lower = m_name.lower()
        if m_name_lower == 'accuracy':
            metrics_list.append('accuracy')
        elif m_name_lower == 'sparse_categorical_accuracy':
            metrics_list.append(metrics.SparseCategoricalAccuracy())
        elif m_name_lower == 'sparse_top_k_categorical_accuracy':
            k = loss_cfg.get('top_k', 5)
            metrics_list.append(metrics.SparseTopKCategoricalAccuracy(k=k, name=f'top_{k}_accuracy'))
        else:
            logger.warning(f"Unsupported metric '{m_name}' specified in config. Skipping.")

    logger.info(f"Compiling model with optimizer: {optimizer_name}, learning_rate: {learning_rate_to_use}, loss: {loss_fn_name}, metrics: {[m.name if hasattr(m, 'name') else m for m in metrics_list]}")
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_list)
    model.summary(print_fn=logger.info)
    return model

def train_model(model: models.Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, config: Dict, index_to_label_map: Dict, strategy: tf.distribute.Strategy) -> None:
    training_cfg = config.get('training', {})
    paths_cfg = config.get('paths', {})

    epochs = training_cfg.get('epochs', 50)

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
            verbose=2
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
    logger.info(f"Using configuration file: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config {config_path}: {e}")
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

    train_dataset, val_dataset, num_classes, index_to_label_map = load_classification_data(config)

    if not train_dataset or not val_dataset:
        logger.error("Failed to load training or validation data. Exiting.")
        return

    logger.info(f"Number of classes determined from data: {num_classes}")
    logger.info(f"Index to label map: {index_to_label_map}")

    lr_schedule_cfg = config.get('training', {}).get('lr_scheduler', {})
    optimizer_cfg = config.get('optimizer', {})
    base_learning_rate = optimizer_cfg.get('learning_rate', 0.001)
    learning_rate_to_use = base_learning_rate

    if lr_schedule_cfg.get('enabled', False):
        schedule_name = lr_schedule_cfg.get('name', '').lower()
        if schedule_name == 'cosine_decay':
            logger.info(f"Using Cosine Decay LR Scheduler (Note: Keras optimizers often take schedule instances directly).")
        else:
            logger.warning(f"Unsupported LR scheduler: {schedule_name}. Using static LR.")
    
    per_replica_batch_size = config.get('data', {}).get('batch_size', 32)
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    logger.info(f"Per-replica batch size: {per_replica_batch_size}")
    logger.info(f"Global batch size (per-replica * num_replicas): {global_batch_size}")
    logger.info(f"Number of replicas for training: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        logger.info("Building model within strategy scope...")
        learning_rate_to_use = optimizer_cfg.get('learning_rate', 0.001)
        
        model = build_model(
            num_classes=len(index_to_label_map),
            config=config,
            learning_rate_to_use=learning_rate_to_use
        )
        if model is None:
            logger.error("Model building failed. Exiting training.")
            return
        logger.info("Model built successfully within strategy scope.")

    if model is None:
        logger.error("Model is None before setting up callbacks. Exiting.")
        return

    train_model(model, train_dataset, val_dataset, config, index_to_label_map, strategy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classification model using a YAML configuration file.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the classification model's YAML configuration file.")
    args = parser.parse_args()
    main(args.config)