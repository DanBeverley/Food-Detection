import os
import argparse
import logging
import yaml
import json
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple

# Assuming data.py is in the same directory or PYTHONPATH is set
from data import load_classification_data, _get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Optional: Add tensorflow-addons if specific loss/optimizers are used
# try:
#     import tensorflow_addons as tfa
# except ImportError:
#     tfa = None
#     logger.warning("TensorFlow Addons not installed. Some loss functions/optimizers might be unavailable.")

def build_model(num_classes: int, config: Dict) -> keras.Model:
    """
    Build and compile a classification model based on the configuration.

    Args:
        num_classes: Number of output classes.
        config: Dictionary containing model configuration.

    Returns:
        Compiled Keras model.

    Raises:
        ValueError: If configuration is invalid or architecture is unsupported.
    """
    model_config = config.get('model', {})
    architecture = model_config.get('architecture', 'EfficientNetV2B0')
    image_size = tuple(config.get('data', {}).get('image_size', [224, 224]))
    use_pretrained = model_config.get('use_pretrained_weights', True)
    fine_tune = model_config.get('fine_tune', False)
    fine_tune_layers = model_config.get('fine_tune_layers', 10) # Number of layers from the end to unfreeze
    weights = 'imagenet' if use_pretrained else None

    input_shape = (*image_size, 3)

    logger.info(f"Building model with architecture: {architecture}, input_shape: {input_shape}, num_classes: {num_classes}")

    if architecture.startswith('EfficientNetV2'):
        try:
            base_model_class = getattr(keras.applications, architecture)
            base_model = base_model_class(include_top=False, input_shape=input_shape, weights=weights)
        except AttributeError:
            raise ValueError(f"Unsupported EfficientNetV2 variant: {architecture}")
    elif architecture.startswith('ConvNeXt'):
        try:
            base_model_class = getattr(keras.applications, architecture)
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
    head_config = model_config.get('classification_head', {})
    pooling_layer = head_config.get('pooling', 'GlobalAveragePooling2D') # or 'GlobalMaxPooling2D'
    dense_layers = head_config.get('dense_layers', [256]) # List of units for dense layers
    dropout_rate = head_config.get('dropout', 0.5)
    activation = head_config.get('activation', 'relu')
    final_activation = head_config.get('final_activation', 'softmax')

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=fine_tune) # Set training=fine_tune for BatchNorm behavior

    if pooling_layer == 'GlobalAveragePooling2D':
        x = keras.layers.GlobalAveragePooling2D()(x)
    elif pooling_layer == 'GlobalMaxPooling2D':
        x = keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = keras.layers.Flatten()(x)

    for units in dense_layers:
        x = keras.layers.Dense(units, activation=activation)(x)
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate)(x)

    outputs = keras.layers.Dense(num_classes, activation=final_activation)(x)
    model = keras.Model(inputs, outputs)

    optimizer_config = config.get('optimizer', {})
    optimizer_name = optimizer_config.get('name', 'Adam').lower()
    learning_rate = optimizer_config.get('learning_rate', 1e-3)

    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    #Optional: Maybe add other optimizers (RMSprop, AdamW from tfa, etc.)
    elif optimizer_name == 'adamw' and tfa:
        weight_decay = optimizer_config.get('weight_decay', 1e-4)
        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    loss_config = config.get('loss', {})
    loss_name = loss_config.get('name', 'sparse_categorical_crossentropy').lower()

    
    if loss_name == 'sparse_categorical_crossentropy':
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=(final_activation != 'softmax'))
    elif loss_name == 'categorical_crossentropy':
         loss = tf.keras.losses.CategoricalCrossentropy(from_logits=(final_activation != 'softmax'))
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

    metrics_config = config.get('metrics', ['accuracy'])
    metrics = [m.lower() for m in metrics_config]
    if 'top_k_categorical_accuracy' in metrics:
        metrics.remove('top_k_categorical_accuracy')
        metrics.append(tf.keras.metrics.TopKCategoricalAccuracy(k=loss_config.get('top_k', 5)))

    logger.info(f"Compiling model with optimizer: {optimizer_name}, learning_rate: {learning_rate}, loss: {loss_name}, metrics: {metrics}")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary(print_fn=logger.info)
    return model

def train_model(model: keras.Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, config: Dict, index_to_label_map: Dict) -> None:
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
    training_config = config.get('training', {})
    epochs = training_config.get('epochs', 50)
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

    callbacks = []
    callbacks_config = training_config.get('callbacks', {})

    # Model Checkpoint
    if callbacks_config.get('model_checkpoint', {}).get('enabled', True):
        ckpt_config = callbacks_config['model_checkpoint']
        filepath = os.path.join(checkpoint_dir, ckpt_config.get('filename_template', 'model_epoch-{epoch:02d}_val_loss-{val_loss:.2f}.h5'))
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=ckpt_config.get('monitor', 'val_loss'),
            save_best_only=ckpt_config.get('save_best_only', True),
            save_weights_only=ckpt_config.get('save_weights_only', False),
            mode=ckpt_config.get('mode', 'min'),
            verbose=1
        ))
        logger.info(f"ModelCheckpoint enabled: Monitor='{ckpt_config.get('monitor', 'val_loss')}', SaveBestOnly={ckpt_config.get('save_best_only', True)}")

    # Early Stopping
    if callbacks_config.get('early_stopping', {}).get('enabled', True):
        es_config = callbacks_config['early_stopping']
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=es_config.get('monitor', 'val_loss'),
            patience=es_config.get('patience', 10),
            restore_best_weights=es_config.get('restore_best_weights', True),
            mode=es_config.get('mode', 'min'),
            verbose=1
        ))
        logger.info(f"EarlyStopping enabled: Monitor='{es_config.get('monitor', 'val_loss')}', Patience={es_config.get('patience', 10)}")

    # Reduce LR on Plateau
    if callbacks_config.get('reduce_lr_on_plateau', {}).get('enabled', True):
        lr_config = callbacks_config['reduce_lr_on_plateau']
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor=lr_config.get('monitor', 'val_loss'),
            factor=lr_config.get('factor', 0.2),
            patience=lr_config.get('patience', 5),
            mode=lr_config.get('mode', 'min'),
            min_lr=lr_config.get('min_lr', 1e-6),
            verbose=1
        ))
        logger.info(f"ReduceLROnPlateau enabled: Monitor='{lr_config.get('monitor', 'val_loss')}', Patience={lr_config.get('patience', 5)}")

    # TensorBoard
    if callbacks_config.get('tensorboard', {}).get('enabled', True):
        tb_config = callbacks_config['tensorboard']
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=tb_config.get('histogram_freq', 1), # Can be expensive
            write_graph=tb_config.get('write_graph', True),
            update_freq=tb_config.get('update_freq', 'epoch') # 'batch' or 'epoch' or integer
        ))
        logger.info(f"TensorBoard logging enabled: Log directory='{log_dir}'")

    # Learning Rate Scheduler (Example: CosineDecay - needs config integration)
    scheduler_config = training_config.get('lr_scheduler', {})
    if scheduler_config.get('name') == 'cosine_decay':
        try:
            # Estimate steps per epoch if possible, otherwise use a large number
            steps_per_epoch = train_dataset.cardinality()
            if steps_per_epoch == tf.data.UNKNOWN_CARDINALITY or steps_per_epoch == tf.data.INFINITE_CARDINALITY:
                logger.warning("Cannot determine dataset cardinality for CosineDecay steps. Estimating based on metadata or using a default.")
                # Attempt estimate from config if available, else fallback
                num_train_samples = config.get('data', {}).get('num_train_samples', 1000) # Get actual number if possible
                batch_size = config.get('data', {}).get('batch_size', 32)
                steps_per_epoch = num_train_samples // batch_size
            
            decay_steps = int(steps_per_epoch * epochs) # Cast to int
            initial_lr = config.get('optimizer', {}).get('learning_rate', 1e-3)
            alpha = scheduler_config.get('alpha', 0.0) # Minimum learning rate factor

            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_lr,
                decay_steps=decay_steps,
                alpha=alpha
            )
            callbacks.append(keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1))
            logger.info(f"CosineDecay LR Scheduler enabled: DecaySteps={decay_steps}, InitialLR={initial_lr}")
        except Exception as e:
             logger.error(f"Error setting up CosineDecay scheduler: {e}. Skipping.")
    # Maybe add other schedulers if necessary(ExponentialDecay, etc.)

    
    logger.info(f"Starting training for {epochs} epochs...")
    try:
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks
        )
        logger.info("Training completed successfully.")

        final_model_path = os.path.join(model_dir, 'final_model.h5')
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

        try:
            with open(label_map_path, 'w') as f:
                json.dump(index_to_label_map, f, indent=4)
            logger.info(f"Label map saved to {label_map_path}")
        except IOError as e:
            logger.error(f"Failed to save label map: {e}")

    except Exception as e:
        logger.error(f"Training error occurred: {e}")
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
        train_dataset, val_dataset, index_to_label_map = load_classification_data(config)
        num_classes = len(index_to_label_map)
        if num_classes == 0:
             logger.error("No classes found in the dataset. Check metadata and data paths.")
             return
        logger.info(f"Data loaded successfully. Number of classes: {num_classes}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    try:
        model = build_model(num_classes=num_classes, config=config)
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        return
    try:
        train_model(model, train_dataset, val_dataset, config, index_to_label_map)
    except Exception as e:
        # Error already logged in train_model
        logger.info("Training process terminated due to error.")
        return

    logger.info("Classification training script finished.")

if __name__ == '__main__':
    main()