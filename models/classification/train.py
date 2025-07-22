import sys
import yaml
import argparse
import logging
from pathlib import Path
import os
import tensorflow as tf

from data import load_classification_data
from model import build_classification_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def initialize_strategy() -> tf.distribute.Strategy:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s). Using OneDeviceStrategy.")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.error(f"Could not set memory growth: {e}")
        return tf.distribute.OneDeviceStrategy(device="/gpu:0")
    
    logger.warning("No GPU found. Using default CPU strategy.")
    return tf.distribute.get_strategy()

def main(args):
    logger.info(f"Loading configuration from: {args.config}")
    config = yaml.safe_load(Path(args.config).read_text())
    strategy = initialize_strategy()

    logger.info("Preparing data pipeline...")
    data_cfg = config.get('data', {})
    
    KAGGLE_IMAGE_DATA_PATH = "/kaggle/input/metafood3d-rgbd-videos/RGBD_videos"
    if Path(KAGGLE_IMAGE_DATA_PATH).exists():
        logger.info(f"Kaggle data path detected. Setting base_data_dir.")
        data_cfg['base_data_dir'] = KAGGLE_IMAGE_DATA_PATH
    else:
        logger.warning(f"Expected Kaggle data path not found at '{KAGGLE_IMAGE_DATA_PATH}'. Check your dataset path.")

    config['data'] = data_cfg
    
    train_ds, val_ds, num_train, num_val, class_names, num_classes = load_classification_data(config)

    if not train_ds:
        logger.critical("Data loading failed. Cannot proceed with training.")
        sys.exit(1)
        
    logger.info(f"Data loaded successfully: {num_train} train samples, {num_val} validation samples.")

    logger.info("Building and compiling the model...")
    with strategy.scope():
        model = build_classification_model(num_classes=num_classes, config=config)

        optimizer_cfg = config.get('optimizer', {})
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=optimizer_cfg.get('learning_rate', 1e-3),
            weight_decay=optimizer_cfg.get('weight_decay', 1e-4)
        )
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=config.get('loss', {}).get('params', {}).get('label_smoothing', 0.1)
        )
        metrics_list = [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_list)
    
    model.summary(print_fn=logger.info)

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="/kaggle/working/best_classification_model.keras",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=5,
            mode='max',
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir="/kaggle/working/logs")
    ]

    logger.info("Starting model training...")
    training_cfg = config.get('training', {})
    epochs = training_cfg.get('epochs', 30)
    batch_size = data_cfg.get('batch_size', 32)
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=num_train // batch_size,
        validation_steps=num_val // batch_size if val_ds else None,
        callbacks=callbacks_list
    )

    logger.info("Training has completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a food classification model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args)