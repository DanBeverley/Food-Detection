import os
import logging
import yaml
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from datetime import datetime

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

# --- Model Building (Standard U-Net) ---

def conv_block(input_tensor, num_filters):
    """Convolutional block: Conv2D -> BatchNormalization -> ReLU -> Conv2D -> BatchNormalization -> ReLU"""
    x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def encoder_block(input_tensor, num_filters):
    """Encoder block: Convolutional block followed by MaxPooling"""
    conv = conv_block(input_tensor, num_filters)
    pool = layers.MaxPooling2D((2, 2))(conv)
    return conv, pool

def decoder_block(input_tensor, skip_tensor, num_filters):
    """Decoder block: Upsampling (Conv2DTranspose), Concatenation with skip connection, Convolutional block"""
    upsample = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    # Ensure skip tensor spatial dimensions match upsample tensor if needed (usually they do with 'same' padding)
    # If there's a mismatch (e.g. due to pooling), cropping might be needed on skip_tensor
    merged = layers.Concatenate()([upsample, skip_tensor])
    conv = conv_block(merged, num_filters)
    return conv

def build_unet(input_shape: tuple, num_classes: int, final_activation: str) -> keras.Model:
    """Builds a standard U-Net model."""
    inputs = keras.Input(shape=input_shape)

    # Encoder Path
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck
    b1 = conv_block(p4, 1024)

    # Decoder Path
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output Layer
    if num_classes == 2:
        # Binary segmentation (foreground/background)
        output_channels = 1
        # Activation is already passed as final_activation ('sigmoid')
    elif num_classes > 2:
        # Multi-class segmentation
        output_channels = num_classes
        # Activation is already passed as final_activation ('softmax')
    else: # num_classes = 1 implies single output channel
        output_channels = 1
        # Activation is likely 'sigmoid' or 'linear' depending on use case
        final_activation = final_activation if final_activation else 'sigmoid' # Default to sigmoid if not specified

    outputs = layers.Conv2D(output_channels, (1, 1), activation=final_activation)(d4)

    model = keras.Model(inputs, outputs, name="UNet")
    logger.info(f"Built U-Net model with input shape {input_shape}, output channels {output_channels}, final activation '{final_activation}'")
    # model.summary() # Optional: Print model summary
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
    if train_ds is None:
        logger.error("Failed to load training dataset. Exiting training.")
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

    # Callbacks setup
    log_dir = config.get('log_dir', 'logs/segmentation/')
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints/segmentation/')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Ensure unique checkpoint path per run if needed, e.g., using timestamp
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, run_id), histogram_freq=1)

    callbacks_list = [tensorboard_callback]
    patience = config.get('early_stopping_patience', 10)

    if val_ds: # Check if validation data exists
        logger.info("Validation dataset provided. Setting up validation callbacks (EarlyStopping, ModelCheckpoint on val_loss).")
        # Use val_loss based checkpointing and early stopping
        checkpoint_path = os.path.join(checkpoint_dir, f"run_{run_id}_best_val_loss.h5")
        callbacks_list.extend([
            keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min', restore_best_weights=True)
            # Add ReduceLROnPlateau(monitor='val_loss', ...) if needed
        ])
    else:
        logger.warning("No validation dataset provided. Setting up callbacks based on training loss (ModelCheckpoint on loss).")
        # Use loss based checkpointing. Early stopping on training loss might stop too early.
        checkpoint_path = os.path.join(checkpoint_dir, f"run_{run_id}_best_loss.h5")
        callbacks_list.extend([
            keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='loss', mode='min', verbose=1)
            # Consider adding EarlyStopping monitoring 'loss' if desired, but be cautious.
            # EarlyStopping(monitor='loss', patience=patience*2, verbose=1, mode='min') # Example: longer patience for train loss
        ])

    # --- Training ---
    logger.info("Starting model training...")
    try:
        history = model.fit(
            train_ds,
            epochs=config.get('training', {}).get('epochs', 50),
            # Pass validation data only if it exists
            validation_data=val_ds if val_ds else None,
            callbacks=callbacks_list,
            # steps_per_epoch=steps_per_epoch, # steps_per_epoch might need adjustment if None (calculated earlier)
            # Pass validation steps only if validation data exists
            # validation_steps=validation_steps if val_ds else None
        )
        logger.info("Training finished.")

        # Save the final model
        final_model_name = f'final_model_run_{run_id}.h5'
        final_model_path = os.path.join(checkpoint_dir, final_model_name)
        logger.info(f"Saving final model to: {final_model_path}")
        model.save(final_model_path)
        logger.info("Final model saved.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)

if __name__ == '__main__':
    config_path = 'models/segmentation/config.yaml'
    train(config_path)