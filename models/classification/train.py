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

# Path to the specific config file for classification training
CLASSIFICATION_CONFIG_PATH = os.path.join(_get_project_root(), "models", "classification", "config.yaml")

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
    model_cfg = config.get('model', {}) # Changed from model_config
    data_cfg = config.get('data', {})   # Added for image_size
    optimizer_cfg = config.get('optimizer', {}) # Added for learning_rate
    loss_cfg = config.get('loss', {}) # Added for loss_fn_name
    metrics_cfg = config.get('metrics', ['accuracy']) # Added for metrics

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

    # Set trainability
    base_model.trainable = fine_tune
    if fine_tune and fine_tune_layers > 0:
        # Freeze all layers first
        for layer in base_model.layers:
            layer.trainable = False
        # Unfreeze the top 'fine_tune_layers'
        if fine_tune_layers < len(base_model.layers):
             logger.info(f"Fine-tuning: Unfreezing the top {fine_tune_layers} layers of the base model.")
             for layer in base_model.layers[-fine_tune_layers:]:
                 layer.trainable = True
        else:
            logger.warning(f"fine_tune_layers ({fine_tune_layers}) >= number of layers in base model ({len(base_model.layers)}). Unfreezing all base model layers.")
            for layer in base_model.layers:
                layer.trainable = True
    elif fine_tune: # fine_tune is True but fine_tune_layers <= 0 means unfreeze all
        logger.info("Fine-tuning: Unfreezing all layers of the base model.")
        for layer in base_model.layers:
            layer.trainable = True
    else:
         logger.info("Feature Extraction: Freezing all layers of the base model.")
         for layer in base_model.layers:
            layer.trainable = False

    # classification head
    head_config = model_cfg.get('classification_head', {})
    pooling_layer = head_config.get('pooling', 'GlobalAveragePooling2D')
    dense_layers_units = head_config.get('dense_layers', [256])
    dropout_rate = head_config.get('dropout', 0.5)
    activation = head_config.get('activation', 'relu')
    final_activation = head_config.get('final_activation', 'softmax')

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=fine_tune)

    if pooling_layer == 'GlobalAveragePooling2D':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling_layer == 'GlobalMaxPooling2D':
        x = layers.GlobalMaxPooling2D()(x)
    else:
        x = layers.Flatten()(x)

    for units in dense_layers_units:
        x = layers.Dense(units, activation=activation)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation=final_activation)(x)
    model = models.Model(inputs, outputs)

    # learning_rate = optimizer_cfg.get('learning_rate', 0.001) # Old: Read static LR from optimizer_cfg
    # Use the passed learning_rate_to_use, which can be a static value or a schedule
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
             k = loss_cfg.get('top_k', 5) # Assuming top_k might be in loss_cfg or a dedicated metrics_cfg section
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
    log_dir_rel = paths_cfg.get('log_dir', 'logs/classification') # log_dir is often relative in config
    label_map_filename = paths_cfg.get('label_map_filename', 'label_map.json')
    checkpoint_dir_rel = config.get('checkpoint_dir', os.path.join(model_dir, 'checkpoints')) # Use specific 'checkpoint_dir' or default

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
    # print(f"DEBUG: callbacks_config = {callbacks_config}") # Keep for debugging if needed

    if callbacks_config.get('model_checkpoint', {}).get('enabled', True):
        ckpt_config = callbacks_config['model_checkpoint']
        filepath = os.path.join(checkpoint_dir_abs, ckpt_config.get('filename_template', 'model_epoch-{epoch:02d}_val_loss-{val_loss:.2f}.h5'))
        callbacks_list.append(callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=ckpt_config.get('monitor', 'val_loss'),
            mode=ckpt_config.get('mode', 'min'),
            save_best_only=ckpt_config.get('save_best_only', True),
            save_weights_only=ckpt_config.get('save_weights_only', False),
            verbose=1
        ))
        logger.info(f"ModelCheckpoint enabled. Saving to {checkpoint_dir_abs}/")

    if callbacks_config.get('early_stopping', {}).get('enabled', True):
        es_config = callbacks_config['early_stopping']
        callbacks_list.append(callbacks.EarlyStopping(
            monitor=es_config.get('monitor', 'val_loss'),
            mode=es_config.get('mode', 'min'),
            patience=es_config.get('patience', 10),
            restore_best_weights=es_config.get('restore_best_weights', True),
            verbose=1
        ))
        logger.info("EarlyStopping enabled.")

    if callbacks_config.get('reduce_lr_on_plateau', {}).get('enabled', True):
        lr_config = callbacks_config['reduce_lr_on_plateau']
        callbacks_list.append(callbacks.ReduceLROnPlateau(
            monitor=lr_config.get('monitor', 'val_loss'),
            mode=lr_config.get('mode', 'min'),
            factor=lr_config.get('factor', 0.2),
            patience=lr_config.get('patience', 5),
            min_lr=lr_config.get('min_lr', 0.000001),
            verbose=1
        ))
        logger.info("ReduceLROnPlateau enabled.")

    if callbacks_config.get('tensorboard', {}).get('enabled', True):
        tb_config = callbacks_config['tensorboard']
        # Use log_dir_abs which is already resolved
        callbacks_list.append(callbacks.TensorBoard(
            log_dir=log_dir_abs, # Use absolute path
            histogram_freq=tb_config.get('histogram_freq', 0),
            write_graph=tb_config.get('write_graph', True),
            update_freq=tb_config.get('update_freq', 'epoch')
        ))
        logger.info(f"TensorBoard logging enabled to {log_dir_abs}.")

    try:
        logger.info(f"Starting training for {epochs} epochs...")
        model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks_list
        )
        logger.info("Training completed.")

        # Save the final model
        final_model_path = os.path.join(model_dir_abs, "final_model.h5")
        model.save(final_model_path)
        logger.info(f"Final trained model saved to: {final_model_path}")

        # Save label map
        with open(label_map_path, 'w') as f:
            json.dump(index_to_label_map, f, indent=4)
        logger.info(f"Label map saved to: {label_map_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise RuntimeError(f"Training process encountered an error: {traceback.format_exc()}")


def main():
    parser = argparse.ArgumentParser(description="Train a classification model.")
    # parser.add_argument("--config", type=str, default=CLASSIFICATION_CONFIG_PATH, help="Path to the training configuration YAML file.") # Defaulting to CLASSIFICATION_CONFIG_PATH
    args = parser.parse_args()

    # config_path = args.config
    config_path = CLASSIFICATION_CONFIG_PATH # Use the defined constant

    try:
        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config: {e}")
        return

    try:
        logger.info("Loading and preparing datasets...")
        # Pass the entire config dictionary to load_classification_data
        train_ds, val_ds, index_to_label = load_classification_data(config) 
        if train_ds is None:
            raise ValueError("Failed to load training data.")
        
        num_classes = len(index_to_label)
        if num_classes == 0:
            raise ValueError("No classes found. Label map is empty.")
        logger.info(f"Number of classes: {num_classes}")

        # Prepare Learning Rate Value or Schedule
        initial_learning_rate = config.get('optimizer', {}).get('learning_rate', 0.001)
        lr_scheduler_cfg = config.get('training', {}).get('callbacks', {}).get('lr_scheduler', {})
        
        final_lr_for_optimizer = initial_learning_rate # Default to static LR

        if lr_scheduler_cfg.get('enabled', False) and lr_scheduler_cfg.get('name') == 'cosine_decay':
            logger.info("Configuring CosineDecay learning rate schedule.")
            steps_per_epoch_tf = tf.data.experimental.cardinality(train_ds)

            if steps_per_epoch_tf == tf.data.experimental.INFINITE_CARDINALITY:
                logger.error("Cannot use CosineDecay: training dataset has infinite cardinality. Steps per epoch must be finite.")
                # Fallback to static LR or raise error - for now, fallback and warn
                logger.warning("Falling back to static learning rate due to infinite dataset cardinality for CosineDecay.")
            else:
                steps_per_epoch = steps_per_epoch_tf.numpy()
                if steps_per_epoch <= 0:
                    logger.error(f"Cannot use CosineDecay: steps_per_epoch resolved to {steps_per_epoch}. Must be positive.")
                    logger.warning("Falling back to static learning rate.")
                else:
                    num_epochs = config.get('training', {}).get('epochs', 1) # Ensure epochs is read correctly
                    decay_steps = steps_per_epoch * num_epochs
                    alpha = lr_scheduler_cfg.get('alpha', 0.0)
                    
                    cosine_decay_schedule = tf.keras.optimizers.schedules.CosineDecay(
                        initial_learning_rate=initial_learning_rate,
                        decay_steps=decay_steps,
                        alpha=alpha
                    )
                    final_lr_for_optimizer = cosine_decay_schedule
                    logger.info(f"CosineDecay schedule created and will be used by the optimizer. Initial LR: {initial_learning_rate}, Decay Steps: {decay_steps}, Alpha: {alpha}")
        else:
            logger.info(f"Using static learning rate for optimizer: {initial_learning_rate}")

        logger.info("Building model...")
        model = build_model(
            num_classes=num_classes, 
            config=config, 
            learning_rate_to_use=final_lr_for_optimizer
        )

        logger.info("Starting model training...")
        train_model(model, train_ds, val_ds, config=config, index_to_label_map=index_to_label)

        logger.info("Classification model training finished successfully.")

    except ValueError as e:
        logger.error(f"Configuration or data error: {e}", exc_info=True)
    except RuntimeError as e:
        logger.error(f"Training runtime error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during the training process: {e}", exc_info=True)

if __name__ == '__main__':
    main()