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
    if len(gpus) > 1:
        logger.info(f"Found {len(gpus)} GPUs. Using MirroredStrategy.")
        return tf.distribute.MirroredStrategy()
    elif len(gpus) == 1:
        logger.info("Found 1 GPU. Using default strategy.")
        return tf.distribute.get_strategy()
    
    logger.warning("No GPU found. Using CPU strategy.")
    return tf.distribute.get_strategy()

def main(args):
    MODEL_CHECKPOINT_PATH = "/kaggle/working/best_classification_model.keras"

    logger.info(f"Loading configuration from: {args.config}")
    config = yaml.safe_load(Path(args.config).read_text())
    strategy = initialize_strategy()

    logger.info("Preparing data pipeline...")
    data_cfg = config.get('data', {})
    if args.base_data_dir:
        data_cfg['base_data_dir'] = args.base_data_dir
    config['data'] = data_cfg
    train_ds, val_ds, num_train, num_val, class_names, num_classes = load_classification_data(config)
    if not train_ds:
        logger.critical("Data loading failed. Cannot proceed.")
        sys.exit(1)
    logger.info(f"Data loaded: {num_train} train, {num_val} val samples.")

    with strategy.scope():
        model = build_classification_model(num_classes=num_classes, config=config)
        
        optimizer_cfg = config.get('optimizer', {})
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=optimizer_cfg.get('learning_rate', 1e-4),
            clipnorm=1.0
        )
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=config.get('loss', {}).get('params', {}).get('label_smoothing', 0.1)
        )
        metrics_list = [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_list)
    
    if Path(MODEL_CHECKPOINT_PATH).exists():
        logger.info(f"Found existing model at {MODEL_CHECKPOINT_PATH}. Loading weights to resume training.")
        model.load_weights(MODEL_CHECKPOINT_PATH)
        logger.info("Weights loaded successfully.")
    else:
        logger.info("No existing model found. Starting a new training run from scratch.")

    model.summary(print_fn=logger.info)

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_CHECKPOINT_PATH,
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
            patience=3,
            mode='max',
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir="/kaggle/working/logs")
    ]

    logger.info("Starting model training...")
    training_cfg = config.get('training', {})
    epochs = training_cfg.get('epochs', 50)
    batch_size = data_cfg.get('batch_size', 64)
    
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
    parser.add_argument("--base_data_dir", type=str, default=None, help="Absolute path to images.")
    args = parser.parse_args()
    main(args)