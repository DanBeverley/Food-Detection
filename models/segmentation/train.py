import tensorflow as tf
import yaml
import os
import pathlib
import logging
from datetime import datetime

# Assuming data.py is in the same directory or accessible in PYTHONPATH
from data import load_segmentation_data # Use relative import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent.parent

def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def unet_model(output_channels: int, image_size: tuple) -> tf.keras.Model:
    """Builds a U-Net model.
    Args:
        output_channels: Number of output channels (e.g., 1 for binary segmentation).
        image_size: Tuple (height, width) for the input image.
    Returns:
        A Keras U-Net model.
    """
    inputs = tf.keras.layers.Input(shape=[image_size[0], image_size[1], 3])

    # Downsampling path
    # Block 1
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Block 2
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 3
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Block 4 (Bottleneck)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    # pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4) # Optional deeper bottleneck if needed

    # Upsampling path
    # Up Block 1
    # For upsampling, Conv2DTranspose is common. Alternatively, UpSampling2D followed by Conv2D.
    up5 = tf.keras.layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(drop4)
    merge5 = tf.keras.layers.concatenate([conv3, up5], axis=3)
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    # Up Block 2
    up6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv5)
    merge6 = tf.keras.layers.concatenate([conv2, up6], axis=3)
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    # Up Block 3
    up7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv6)
    merge7 = tf.keras.layers.concatenate([conv1, up7], axis=3)
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    # Output layer: 1 channel for binary segmentation, sigmoid activation for probabilities [0,1]
    outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='sigmoid')(conv7)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    logger.info("U-Net model built.")
    return model

def main():
    project_root = _get_project_root()
    config_file_path = project_root / 'config_pipeline.yaml'
    config = load_config(config_file_path)

    seg_config = config.get('segmentation_training_data')
    if not seg_config:
        logger.error("Segmentation training configuration ('segmentation_training_data') not found in config_pipeline.yaml")
        return

    logger.info("Starting segmentation model training...")

    # Load data using the updated data loader
    train_dataset, val_dataset = load_segmentation_data(seg_config)

    if train_dataset is None:
        logger.error("Failed to load training dataset. Exiting.")
        return

    # Define model
    output_channels = 1 # For binary segmentation (foreground/background)
    image_h, image_w = seg_config['image_size']
    model = unet_model(output_channels=output_channels, image_size=(image_h, image_w))

    # Compile model
    # Using BinaryCrossentropy because the last layer has a sigmoid activation.
    # MeanIoU is a common metric for segmentation. num_classes=2 (background and foreground).
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=seg_config.get('learning_rate', 1e-4)),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2, name='mean_iou')])

    model.summary(print_fn=logger.info)

    # Training parameters from config
    epochs = seg_config.get('epochs', 50)
    early_stopping_patience = seg_config.get('early_stopping_patience', 10)

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = project_root / 'trained_models' / 'segmentation'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the best model based on validation IoU
    checkpoint_filepath_obj = model_dir / f'unet_segmentation_best_iou_{timestamp}.keras'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_filepath_obj), # Convert Path object to string
        save_weights_only=False, # Save entire model in Keras format
        monitor='val_mean_iou',  # Monitor validation IoU
        mode='max',              # We want to maximize IoU
        save_best_only=True,
        verbose=1
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=early_stopping_patience,
        verbose=1,
        restore_best_weights=True # Restores model weights from the epoch with the best value of the monitored quantity.
    )
    
    tensorboard_log_dir = project_root / 'logs' / 'segmentation' / timestamp
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir), histogram_freq=1)

    callbacks_list = [model_checkpoint_callback, early_stopping_callback, tensorboard_callback]

    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Training data: {train_dataset}")
    logger.info(f"Validation data: {val_dataset if val_dataset else 'None'}")

    history = model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        validation_data=val_dataset, # Pass validation data here
        verbose=1
    )

    logger.info("Training finished.")

    # Save the final model (could be the one with restored best weights due to EarlyStopping)
    final_model_path_obj = model_dir / f'unet_segmentation_final_{timestamp}.keras'
    model.save(str(final_model_path_obj)) # Convert Path object to string
    logger.info(f"Final trained model saved to: {str(final_model_path_obj)}")

    # Optionally, load the best saved model and evaluate on a test set
    # logger.info(f"Loading best model from: {str(checkpoint_filepath_obj)}")
    # best_model = tf.keras.models.load_model(str(checkpoint_filepath_obj))
    # if test_dataset:
    #     logger.info("Evaluating best model on test set...")
    #     test_loss, test_accuracy, test_iou = best_model.evaluate(test_dataset)
    #     logger.info(f"Best Model - Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test IoU: {test_iou}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in the segmentation training script: {e}", exc_info=True)