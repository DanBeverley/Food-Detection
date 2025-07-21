import os
import sys
import yaml
import logging
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import numpy as np
from data import load_segmentation_data 
from model import build_simple_fused_model
from utils import TqdmProgressCallback, StableIoU
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def set_mixed_precision_policy(config, strategy):
    mixed_precision_enabled = config.get('training', {}).get('mixed_precision', False)
    if mixed_precision_enabled:
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled with policy: mixed_bfloat16")
    else:
        logger.info("Mixed precision disabled")

def initialize_strategy():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPUs. Using OneDeviceStrategy.")
        return tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        logger.info("No GPUs found, using CPU strategy.")
        return tf.distribute.get_strategy()

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def main():
    project_root = _get_project_root()
    strategy = initialize_strategy()
    logger.info(f"Training will use strategy: {strategy.__class__.__name__}")
    
    config_path = project_root / 'models' / 'segmentation' / 'config.yaml'
    config = load_config(str(config_path))
    
    # Disable mixed precision for stability
    logger.info("Mixed precision disabled for stability")
    
    train_ds, val_ds, test_ds, num_train, num_val, num_test, num_classes = load_segmentation_data(config)
    if train_ds is None:
        logger.error("Data loading failed. Exiting.")
        return

    data_cfg = config.get('data', {})
    per_replica_batch_size = data_cfg.get('batch_size', 16)
    steps_per_epoch = num_train // per_replica_batch_size
    validation_steps = num_val // per_replica_batch_size if val_ds else None

    with strategy.scope():
        model = build_simple_fused_model(
            output_channels=num_classes, 
            image_size=tuple(data_cfg.get('image_size')),
            data_config=data_cfg
        )
        
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics_list = [
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", from_logits=True),
            StableIoU(name='stable_iou', from_logits=True)
        ]
        optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
        
        model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_list)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_dir_rel = config.get('paths', {}).get('model_save_dir', 'trained_models/segmentation')
    model_dir_abs = project_root / model_save_dir_rel
    model_dir_abs.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        TqdmProgressCallback(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir_abs / f'simple_model_{timestamp}.h5'),
            monitor='val_stable_iou',
            save_best_only=True,
            verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_stable_iou',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
    ]
    
    # Single training run - no staged training
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=0
    )

    logger.info("Training finished.")
    
    final_model_path = model_dir_abs / f'simple_segmentation_final_{timestamp}.h5'
    model.save(str(final_model_path)) 
    logger.info(f"Final trained model saved to: {final_model_path}")

    if test_ds and num_test > 0:
        logger.info("Evaluating model on the test set...")
        test_results = model.evaluate(test_ds, verbose=1)
        logger.info("Test Set Evaluation Results:")
        if isinstance(test_results, list):
            for metric_name, value in zip(model.metrics_names, test_results):
                logger.info(f"  {metric_name}: {value}")
        else:
            logger.info(f"  loss: {test_results}")

if __name__ == '__main__':
    main()