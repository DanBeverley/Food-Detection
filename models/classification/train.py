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

def build_model(num_classes: int, config: Dict) -> models.Model:
    """
    Build and compile a classification model based on the configuration.

    Args:
        num_classes: Number of output classes.
        config: Dictionary containing classification training configuration, 
                expected to have 'model_config' and 'image_size'.

    Returns:
        Compiled Keras model.

    Raises:
        ValueError: If configuration is invalid or architecture is unsupported.
    """
    model_params = config.get('model_config', {})
    architecture = model_params.get('architecture', 'EfficientNetV2B0')
    image_size_list = config.get('image_size', [224, 224]) # Get image_size from the main config arg
    image_size = tuple(image_size_list)

    use_pretrained = model_params.get('use_pretrained_weights', True)
    fine_tune = model_params.get('fine_tune', False)
    fine_tune_layers = model_params.get('fine_tune_layers', 10) # Number of layers from the end to unfreeze
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
    head_config = model_params.get('classification_head', {})
    pooling_layer = head_config.get('pooling', 'GlobalAveragePooling2D') # or 'GlobalMaxPooling2D'
    dense_layers_units = head_config.get('dense_layers', [256]) # List of units for dense layers
    dropout_rate = head_config.get('dropout', 0.5)
    activation = head_config.get('activation', 'relu')
    final_activation = head_config.get('final_activation', 'softmax')

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=fine_tune) # Set training=fine_tune for BatchNorm behavior

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

    # Get learning_rate directly from the main config (classification_training_data)
    learning_rate = config.get('learning_rate', 0.001)  # Default if not specified

    optimizer_config = config.get('optimizer', {})
    optimizer_name = optimizer_config.get('name', 'Adam').lower()

    if optimizer_name == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    #Optional: Maybe add other optimizers (RMSprop, AdamW from tfa, etc.)
    # elif optimizer_name == 'adamw' and tfa:
    #     weight_decay = optimizer_config.get('weight_decay', 1e-4)
    #     optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    loss_config = config.get('loss', {})
    loss_fn_name = loss_config.get('name', 'sparse_categorical_crossentropy').lower()

    if loss_fn_name == 'sparse_categorical_crossentropy':
        loss_fn = losses.SparseCategoricalCrossentropy(from_logits=(final_activation != 'softmax'))
    elif loss_fn_name == 'categorical_crossentropy':
         loss_fn = losses.CategoricalCrossentropy(from_logits=(final_activation != 'softmax'))
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn_name}")

    metrics_config = config.get('metrics', ['accuracy'])
    metrics_list = []
    for m_name in metrics_config:
        m_name_lower = m_name.lower()
        if m_name_lower == 'accuracy':
            metrics_list.append('accuracy')
        elif m_name_lower == 'sparse_categorical_accuracy':
             metrics_list.append(metrics.SparseCategoricalAccuracy())
        elif m_name_lower == 'sparse_top_k_categorical_accuracy':
             k = loss_config.get('top_k', 5)
             metrics_list.append(metrics.SparseTopKCategoricalAccuracy(k=k, name=f'top_{k}_accuracy'))
        # Add other metrics as needed
        else:
            logger.warning(f"Unsupported metric '{m_name}' specified in config. Skipping.")

    logger.info(f"Compiling model with optimizer: {optimizer_name}, learning_rate: {learning_rate}, loss: {loss_fn_name}, metrics: {[m.name if hasattr(m, 'name') else m for m in metrics_list]}")
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
        config: Dictionary containing training configuration.
        index_to_label_map: Mapping from integer index to string label.

    Raises:
        RuntimeError: If training fails.
    """
    epochs = config.get('epochs', 50) # Correctly get epochs from the classification_training_data config

    model_dir = config.get('paths', {}).get('model_save_dir', 'trained_models/classification')
    log_dir = config.get('paths', {}).get('log_dir', 'logs/classification')
    checkpoint_dir = os.path.join(model_dir, 'checkpoints') # Subdirectory for checkpoints
    label_map_filename = config.get('paths', {}).get('label_map_filename', 'label_map.json')

    # Resolve paths relative to project root
    project_root = _get_project_root()
    model_dir = os.path.join(project_root, model_dir)
    log_dir = os.path.join(project_root, log_dir)
    checkpoint_dir = os.path.join(project_root, checkpoint_dir)
    label_map_path = os.path.join(model_dir, label_map_filename)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Model artifacts will be saved to: {model_dir}")
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")

    callbacks_list = []
    callbacks_config = config.get('callbacks', {})
    print(f"DEBUG: callbacks_config = {callbacks_config}") 

    # Model Checkpoint
    if callbacks_config.get('model_checkpoint', {}).get('enabled', True):
        ckpt_config = callbacks_config['model_checkpoint']
        filepath = os.path.join(checkpoint_dir, ckpt_config.get('filename_template', 'model_epoch-{epoch:02d}_val_loss-{val_loss:.2f}.h5'))
        callbacks_list.append(callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=ckpt_config.get('monitor', 'val_loss'),
            save_best_only=ckpt_config.get('save_best_only', True),
            save_weights_only=ckpt_config.get('save_weights_only', False),
            mode=ckpt_config.get('mode', 'min'),
            verbose=1
        ))
        logger.info(f"ModelCheckpoint enabled: Monitor='{ckpt_config.get('monitor', 'val_loss')}', SaveBestOnly={ckpt_config.get('save_best_only', True)}")

    # TensorBoard
    if callbacks_config.get('tensorboard', {}).get('enabled', True):
        tb_config = callbacks_config['tensorboard']
        callbacks_list.append(callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=tb_config.get('histogram_freq', 1), # Log histograms every epoch
            write_graph=tb_config.get('write_graph', True),
            write_images=tb_config.get('write_images', False),
            update_freq=tb_config.get('update_freq', 'epoch') # 'epoch' or 'batch'
        ))
        logger.info(f"TensorBoard enabled: Logging to {log_dir}")

    # Early Stopping
    if callbacks_config.get('early_stopping', {}).get('enabled', True):
        es_config = callbacks_config['early_stopping']
        callbacks_list.append(callbacks.EarlyStopping(
            monitor=es_config.get('monitor', 'val_loss'),
            patience=es_config.get('patience', 10),
            min_delta=es_config.get('min_delta', 0.001),
            mode=es_config.get('mode', 'min'),
            restore_best_weights=es_config.get('restore_best_weights', True),
            verbose=1
        ))
        logger.info(f"EarlyStopping enabled: Monitor='{es_config.get('monitor', 'val_loss')}', Patience={es_config.get('patience', 10)}")

    # ReduceLROnPlateau
    if callbacks_config.get('reduce_lr_on_plateau', {}).get('enabled', True):
        lr_config = callbacks_config['reduce_lr_on_plateau']
        callbacks_list.append(callbacks.ReduceLROnPlateau(
            monitor=lr_config.get('monitor', 'val_loss'),
            factor=lr_config.get('factor', 0.1),
            patience=lr_config.get('patience', 5),
            min_lr=lr_config.get('min_lr', 1e-6),
            mode=lr_config.get('mode', 'min'),
            verbose=1
        ))
        logger.info(f"ReduceLROnPlateau enabled: Monitor='{lr_config.get('monitor', 'val_loss')}', Factor={lr_config.get('factor', 0.1)}")

    # Save label map
    try:
        with open(label_map_path, 'w') as f:
            json.dump(index_to_label_map, f, indent=4)
        logger.info(f"Label map saved to: {label_map_path}")
    except IOError as e:
        logger.error(f"Failed to save label map to {label_map_path}: {e}")

    # Start Training
    logger.info(f"Starting training for {epochs} epochs...")
    try:
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks_list
        )
        logger.info("Training finished.")

        # Save the final model (best weights might already be saved by ModelCheckpoint)
        final_model_path = os.path.join(model_dir, 'final_model.h5')
        try:
            model.save(final_model_path)
            logger.info(f"Final model saved to: {final_model_path}")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")

    except Exception as e:
        logger.error(f"An error occurred during model training: {e}", exc_info=True)
        # Force print traceback to stdout
        print("--- DETAILED TRACEBACK ---")
        print(traceback.format_exc())
        print("-------------------------")
        raise RuntimeError("Model training failed.") from e

def main():
    parser = argparse.ArgumentParser(description="Train classification model using a configuration file.")
    parser.add_argument('--config', type=str, default='models/classification/config.yaml', help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config_path = args.config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from: {config_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        return

    # Mixed Precision (Optional) 
    if config.get('training', {}).get('use_mixed_precision', False) and tf.config.list_physical_devices('GPU'):
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled (mixed_float16).")
        except ImportError:
            logger.warning("Mixed precision requested but failed to import/set policy.")
        except Exception as e:
            logger.error(f"Error setting mixed precision policy: {e}")

    try:
        logger.info("Loading and preparing datasets...")
        # Pass only the 'classification_training_data' sub-configuration
        train_dataset, val_dataset, index_to_label_map = load_classification_data(config['classification_training_data'])
        num_classes = len(index_to_label_map)
        if num_classes == 0:
             logger.error("No classes found in the dataset. Check metadata and data paths.")
             return
        logger.info(f"Data loaded successfully. Number of classes: {num_classes}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    try:
        # Pass the 'classification_training_data' sub-configuration (and num_classes)
        model = build_model(num_classes=num_classes, config=config['classification_training_data'])
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        return
    try:
        # Pass the entire 'classification_training_data' sub-configuration
        train_model(model, train_dataset, val_dataset, config['classification_training_data'], index_to_label_map)
    except BaseException as e: # Catch BaseException to include things like SystemExit
        # Error already logged in train_model # This comment might be wrong
        # logger.info("Training process terminated due to error.") # Replace this
        logger.error(f"Error caught in main during training: {e}", exc_info=True) # Add detailed log
        print("--- TRACEBACK FROM MAIN ---") # Add print for redundancy
        print(traceback.format_exc())
        print("-------------------------")
        return

    logger.info("Classification training script finished.")

if __name__ == '__main__':
    main()