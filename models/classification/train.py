import os
import argparse
import logging
import yaml
import json
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, callbacks, applications, metrics
from typing import Dict, Tuple
import traceback

from data import load_classification_data, _get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_model(num_classes: int, config: Dict, learning_rate_to_use) -> models.Model:
    """
    Build and compile a classification model based on the configuration.

    Args:
        num_classes: Number of output classes.
        config: Dictionary containing classification training configuration (the entire content of config.yaml).
        learning_rate_to_use: The learning rate value (float) or a tf.keras.optimizers.schedules.LearningRateSchedule instance.

    Returns:
        Compiled Keras model.

    Raises:
        ValueError: If configuration is invalid or architecture is unsupported.
    """
    model_cfg = config.get('model', {}) 
    data_cfg = config.get('data', {})   
    optimizer_cfg = config.get('optimizer', {}) 
    loss_cfg = config.get('loss', {}) 
    metrics_cfg = config.get('metrics', ['accuracy']) 

    architecture = model_cfg.get('architecture', 'EfficientNetV2B0')
    image_size_list = data_cfg.get('image_size', [224, 224])
    image_size = tuple(image_size_list)

    use_pretrained = model_cfg.get('use_pretrained_weights', True)
    fine_tune = model_cfg.get('fine_tune', False)
    fine_tune_layers = model_cfg.get('fine_tune_layers', 10)
    weights = 'imagenet' if use_pretrained else None

    input_shape = (*image_size, 3)

    logger.info(f"Building model with architecture: {architecture}, input_shape: {input_shape}, num_classes: {num_classes}")

    if architecture.startswith('EfficientNetV2'):
        try:
            base_model_class = getattr(applications, architecture)
            base_model = base_model_class(include_top=False, input_shape=input_shape, weights=weights)
        except AttributeError:
            raise ValueError(f"Unsupported EfficientNetV2 variant: {architecture}")
    elif architecture.startswith('ConvNeXt'):
        try:
            base_model_class = getattr(applications, architecture)
            base_model = base_model_class(include_top=False, input_shape=input_shape, weights=weights)
        except AttributeError:
            raise ValueError(f"Unsupported ConvNeXt variant: {architecture}")
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    base_model.trainable = fine_tune
    if fine_tune and fine_tune_layers > 0:
        for layer in base_model.layers:
            layer.trainable = False
        if fine_tune_layers < len(base_model.layers):
             logger.info(f"Fine-tuning: Unfreezing the top {fine_tune_layers} layers of the base model.")
             for layer in base_model.layers[-fine_tune_layers:]:
                 layer.trainable = True
        else:
            logger.warning(f"fine_tune_layers ({fine_tune_layers}) >= number of layers in base model ({len(base_model.layers)}). Unfreezing all base model layers.")
            for layer in base_model.layers:
                layer.trainable = True
    elif fine_tune: 
        logger.info("Fine-tuning: Unfreezing all layers of the base model.")
        for layer in base_model.layers:
            layer.trainable = True
    else:
         logger.info("Feature Extraction: Freezing all layers of the base model.")
         for layer in base_model.layers:
            layer.trainable = False

    head_config = model_cfg.get('classification_head', {})
    pooling_layer = head_config.get('pooling', 'GlobalAveragePooling2D')
    dense_layers_units = head_config.get('dense_layers', [256])
    dropout_rate = head_config.get('dropout', 0.5)
    activation = head_config.get('activation', 'relu')
    final_activation = head_config.get('final_activation', 'softmax')
    kernel_l2_factor = head_config.get('kernel_l2_factor', 0.0)

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=fine_tune)

    if pooling_layer == 'GlobalAveragePooling2D':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling_layer == 'GlobalMaxPooling2D':
        x = layers.GlobalMaxPooling2D()(x)
    else:
        x = layers.Flatten()(x)

    kernel_regularizer = None
    if kernel_l2_factor > 0:
        kernel_regularizer = tf.keras.regularizers.l2(kernel_l2_factor)
        logger.info(f"Applying L2 kernel regularization with factor: {kernel_l2_factor} to Dense layers in head.")

    for units in dense_layers_units:
        x = layers.Dense(units, activation=activation, kernel_regularizer=kernel_regularizer)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation=final_activation)(x) 
    model = models.Model(inputs, outputs)

    current_learning_rate = learning_rate_to_use 
    optimizer_name = optimizer_cfg.get('name', 'Adam').lower()

    if optimizer_name == 'adam':
        optimizer = optimizers.Adam(learning_rate=current_learning_rate)
    elif optimizer_name == 'sgd':
        momentum = optimizer_cfg.get('momentum', 0.9)
        optimizer = optimizers.SGD(learning_rate=current_learning_rate, momentum=momentum)
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

    logger.info(f"Compiling model with optimizer: {optimizer_name}, learning_rate: {current_learning_rate}, loss: {loss_fn_name}, metrics: {[m.name if hasattr(m, 'name') else m for m in metrics_list]}")
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_list)
    model.summary(print_fn=logger.info)
    return model

def train_model(model: models.Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, config: Dict, index_to_label_map: Dict) -> None:
    """
    Train the classification model with callbacks and save artifacts.

    Args:
        model: Compiled Keras model.
        train_dataset: Training tf.data.Dataset.
        val_dataset: Validation tf.data.Dataset.
        config: Dictionary containing training configuration (the entire content of config.yaml).
        index_to_label_map: Mapping from integer index to string label.

    Raises:
        RuntimeError: If training fails.
    """
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
        tb_log_dir = tb_cfg.get('log_dir', 'logs/classification') 
        if not os.path.isabs(tb_log_dir):
            tb_log_dir_abs = os.path.join(_get_project_root(), tb_log_dir)
        else:
            tb_log_dir_abs = tb_log_dir
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
            verbose=1 
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

    if config.get('training', {}).get('use_mixed_precision', False):
        logger.info("Using mixed precision training.")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        logger.info("Using full precision training (float32).")

    paths_cfg = config.get('paths', {})
    project_root = _get_project_root()

    data_dir_rel = paths_cfg.get('data_dir', 'data/classification')
    data_dir_abs = os.path.join(project_root, data_dir_rel) 

    metadata_filename = paths_cfg.get('metadata_filename', 'metadata.json')
    metadata_path_abs = os.path.join(data_dir_abs, metadata_filename)

    label_map_dir_rel = paths_cfg.get('label_map_dir', data_dir_rel) # Default to data_dir if not specified
    label_map_filename = paths_cfg.get('label_map_filename', 'label_map.json')
    label_map_dir_abs = os.path.join(project_root, label_map_dir_rel)
    label_map_path_abs = os.path.join(label_map_dir_abs, label_map_filename) # This path is now primarily for reference/logging if load_classification_data handles it

    logger.info(f"Expecting metadata to be read by load_classification_data using relative path from config: {paths_cfg.get('data_dir', 'data/classification')}/{paths_cfg.get('metadata_filename', 'metadata.json')}")
    logger.info(f"Expecting label map to be read by load_classification_data using relative path from config: {label_map_dir_rel}/{label_map_filename}")

    # Call load_classification_data with the main config object.
    # The function load_classification_data is responsible for extracting
    # metadata_path, label_map_path, image_size, batch_size, etc., from the config.
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
    
    model = build_model(num_classes, config, learning_rate_to_use)
    train_model(model, train_dataset, val_dataset, config, index_to_label_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classification model using a YAML configuration file.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the classification model's YAML configuration file.")
    args = parser.parse_args()
    main(args.config) 