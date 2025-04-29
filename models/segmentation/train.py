import os
import logging
import yaml
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from data import load_segmentation_datasets, _get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Loss Functions
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Calculates Dice Loss.

    Args:
        y_true: Ground truth masks (Batch, H, W, C).
        y_pred: Predicted masks (Batch, H, W, C), probabilities from sigmoid/softmax.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice loss score.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # Formula: 1 - (2 * Intersection + smooth) / (Sum(A) + Sum(B) + smooth)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculates Dice Coefficient (complement of Dice Loss)."""
    return 1 - dice_loss(y_true, y_pred, smooth)

# Metrics 
# Using Keras built-in MeanIoU is generally preferred
def get_metrics(config_metrics: list, num_classes: int) -> list:
    """Parses metric names from config and returns Keras metric objects."""
    metrics = []
    for metric_name in config_metrics:
        if metric_name == 'iou_metric' or metric_name == 'mean_iou':
            metrics.append(tf.keras.metrics.MeanIoU(num_classes=num_classes, name='mean_iou'))
        elif metric_name == 'dice_coefficient':
            metrics.append(dice_coefficient) # Add the function directly
        elif metric_name == 'accuracy':
             # For segmentation, sparse_categorical_accuracy might be more relevant if using integer masks
             # Or use binary_accuracy for binary masks
            metrics.append('accuracy') # Standard accuracy
        else:
            logger.warning(f"Unsupported metric specified in config: {metric_name}")
    return metrics

def build_unet(input_shape: tuple, num_classes: int, activation: str) -> keras.Model:
    """Builds a simple U-Net model."""
    inputs = keras.Input(shape=input_shape)

    # Encoder Path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

    # Decoder Path
    u4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u4)
    c4 = layers.Dropout(0.1)(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = layers.Dropout(0.1)(c5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Determine the number of output filters and activation based on num_classes
    if num_classes == 2: # Binary segmentation (foreground/background)
        output_channels = 1 # Single channel output
        if activation not in ['sigmoid', 'softmax']: # Default to sigmoid for binary
             logger.warning(f"Activation '{activation}' invalid for binary (num_classes=2). Using 'sigmoid'.")
             final_activation = 'sigmoid'
        else:
            final_activation = activation
    elif num_classes > 2: # Multi-class segmentation
        output_channels = num_classes
        if activation != 'softmax': # Default to softmax for multi-class
            logger.warning(f"Activation '{activation}' invalid for multi-class (num_classes>2). Using 'softmax'.")
            final_activation = 'softmax'
        else:
            final_activation = activation
    else: 
        logger.warning(f"num_classes={num_classes} treated as binary. Check config if multi-class needed.")
        output_channels = 1
        final_activation = 'sigmoid'
        
    outputs = layers.Conv2D(output_channels, (1, 1), activation=final_activation)(c5)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

def build_segmentation_model(config: dict) -> keras.Model:
    """Builds the segmentation model based on config settings."""
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    architecture = model_config.get('architecture', 'UNet')
    input_shape = tuple(model_config.get('input_shape', data_config.get('image_size', [256, 256]) + [3])) # H, W, C
    num_classes = model_config.get('num_classes', data_config.get('num_classes', 2))
    activation = model_config.get('activation', 'sigmoid') # Final layer activation
    # backbone = model_config.get('backbone', None)
    # use_pretrained = model_config.get('use_pretrained_backbone_weights', True)

    logger.info(f"Building model: Architecture={architecture}, Input Shape={input_shape}, Num Classes={num_classes}, Activation={activation}")

    if architecture.lower() == 'unet':
        model = build_unet(input_shape, num_classes, activation)
    # elif architecture.lower() == 'deeplabv3+':
        # builder for DeepLab or other models 
        # Consider using segmentation_models library: 
        # import segmentation_models as sm
        # model = sm.Unet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation=activation, encoder_weights='imagenet' if use_pretrained else None)
    else:
        raise ValueError(f"Unsupported model architecture in config: {architecture}")

    return model

def get_optimizer(config: dict) -> keras.optimizers.Optimizer:
    """Creates optimizer based on config."""
    opt_config = config.get('optimizer', {})
    opt_name = opt_config.get('name', 'Adam').lower()
    lr = opt_config.get('learning_rate', 0.001)
    
    if opt_name == 'adam':
        return keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == 'sgd':
        momentum = opt_config.get('momentum', 0.9)
        return keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    elif opt_name == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=lr)
    elif opt_name == 'adamw':
        import tensorflow_addons as tfa
        weight_decay = opt_config.get('weight_decay', 0.0001)
        return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    else:
        logger.warning(f"Unsupported optimizer '{opt_name}'. Defaulting to Adam.")
        return keras.optimizers.Adam(learning_rate=lr)

def get_loss(config: dict):
    """Gets loss function based on config."""
    loss_config = config.get('loss', {})
    loss_name = loss_config.get('name', 'dice_loss').lower()
    num_classes = config.get('model', {}).get('num_classes', config.get('data', {}).get('num_classes', 2))

    if loss_name == 'dice_loss':
        return dice_loss
    elif loss_name == 'binary_crossentropy' or loss_name == 'bce':
        return tf.keras.losses.BinaryCrossentropy(from_logits=False) # Assuming sigmoid output
    elif loss_name == 'categorical_crossentropy' or loss_name == 'cce':
        # Ensure output is softmax and masks are one-hot
        return tf.keras.losses.CategoricalCrossentropy(from_logits=False) 
    elif loss_name == 'sparse_categorical_crossentropy' or loss_name == 'scce':
         # Ensure output is softmax and masks are integer labels (0, 1, ..., N-1)
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # Add combined losses or other custom losses here
    # elif loss_name == 'focal_loss': ...
    else:
        logger.warning(f"Unsupported loss '{loss_name}'. Defaulting to Dice Loss.")
        return dice_loss

def get_callbacks(config: dict, model_save_dir: str) -> list:
    """Creates Keras callbacks based on config."""
    cb_config = config.get('training', {}).get('callbacks', {})
    callbacks = []

    # Model Checkpoint
    mcp_conf = cb_config.get('model_checkpoint', {})
    if mcp_conf.get('enabled', True):
        monitor = mcp_conf.get('monitor', 'val_loss')
        mode = mcp_conf.get('mode', 'auto') # Auto-detect min/max based on monitor name
        save_best = mcp_conf.get('save_best_only', True)
        save_weights = mcp_conf.get('save_weights_only', False)
        filename = mcp_conf.get('filename_template', 'model_epoch-{epoch:02d}_val_loss-{val_loss:.4f}.h5')
        filepath = os.path.join(model_save_dir, 'checkpoints', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            mode=mode,
            save_best_only=save_best,
            save_weights_only=save_weights,
            verbose=1
        ))

    # Early Stopping
    es_conf = cb_config.get('early_stopping', {})
    if es_conf.get('enabled', True):
        monitor = es_conf.get('monitor', 'val_loss')
        mode = es_conf.get('mode', 'auto')
        patience = es_conf.get('patience', 10)
        restore_best = es_conf.get('restore_best_weights', True)
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=restore_best,
            verbose=1
        ))

    # Reduce LR on Plateau
    rlr_conf = cb_config.get('reduce_lr_on_plateau', {})
    if rlr_conf.get('enabled', True):
        monitor = rlr_conf.get('monitor', 'val_loss')
        mode = rlr_conf.get('mode', 'auto')
        factor = rlr_conf.get('factor', 0.1)
        patience = rlr_conf.get('patience', 5)
        min_lr = rlr_conf.get('min_lr', 0)
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=1
        ))
        
    # TensorBoard
    tb_conf = cb_config.get('tensorboard', {})
    if tb_conf.get('enabled', True):
        log_dir_rel = tb_conf.get('log_dir', 'logs/segmentation/')
        log_dir = os.path.join(_get_project_root(), log_dir_rel)
        os.makedirs(log_dir, exist_ok=True)
        hist_freq = tb_conf.get('histogram_freq', 0)
        write_graph = tb_conf.get('write_graph', True)
        update_freq = tb_conf.get('update_freq', 'epoch')
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=hist_freq,
            write_graph=write_graph,
            update_freq=update_freq
        ))

    # Optional: Add LR Schedulers 
    # scheduler_conf = cb_config.get('lr_scheduler', {})
    # if scheduler_conf and scheduler_conf.get('name'): ... create and add scheduler callback

    return callbacks

def train(config_path: str):
    # Load Config
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

    project_root = _get_project_root()
    paths_config = config.get('paths', {})
    model_save_dir_rel = paths_config.get('model_save_dir', 'trained_models/segmentation/')
    model_save_dir = os.path.join(project_root, model_save_dir_rel)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Load Data 
    logger.info("Loading datasets...")
    train_ds, val_ds, _ = load_segmentation_datasets(config_path)
    if train_ds is None or val_ds is None:
        logger.error("Failed to load datasets. Exiting training.")
        return
    logger.info("Datasets loaded successfully.")

    # Build Model
    logger.info("Building model...")
    try:
        model = build_segmentation_model(config)
        model.summary(print_fn=logger.info)
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        return

    #  Compile Model
    logger.info("Compiling model...")
    optimizer = get_optimizer(config)
    loss = get_loss(config)
    metrics_conf = config.get('metrics', ['iou_metric'])
    num_classes = config.get('model', {}).get('num_classes', config.get('data', {}).get('num_classes', 2))
    metrics = get_metrics(metrics_conf, num_classes)
    
    # Handle Mixed Precision
    if config.get('training', {}).get('use_mixed_precision', False):
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision policy 'mixed_float16' set.")
            # Loss scaling is handled automatically by model.fit with mixed precision policies
        except Exception as e:
            logger.warning(f"Could not enable mixed precision: {e}")

    try:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logger.info("Model compiled successfully.")
    except Exception as e:
        logger.error(f"Failed to compile model: {e}")
        return

    # Setup Callbacks 
    logger.info("Setting up callbacks...")
    callbacks = get_callbacks(config, model_save_dir)
    logger.info(f"Callbacks configured: {[cb.__class__.__name__ for cb in callbacks]}")

    # Train Model 
    epochs = config.get('training', {}).get('epochs', 50)
    logger.info(f"Starting training for {epochs} epochs...")
    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks
        )
        logger.info("Training finished.")

        # Save Final Model  
        final_model_name = 'final_model.h5'
        final_model_path = os.path.join(model_save_dir, final_model_name)
        logger.info(f"Saving final model to: {final_model_path}")
        model.save(final_model_path)
        logger.info("Final model saved successfully.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)

if __name__ == '__main__':
    config_path = 'models/segmentation/config.yaml'
    train(config_path)