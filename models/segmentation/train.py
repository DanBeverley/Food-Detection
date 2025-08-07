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

# Custom loss functions and metrics for model loading
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss function for binary segmentation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss function for binary segmentation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Clip predictions to prevent log(0)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Compute focal loss
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    focal_weight = alpha_t * tf.pow(1 - pt, gamma)
    focal_loss = -focal_weight * tf.math.log(pt)
    
    return tf.reduce_mean(focal_loss)

def combined_loss(y_true, y_pred, dice_weight=0.5, focal_weight=0.5):
    """Combined dice and focal loss."""
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    return dice_weight * dice + focal_weight * focal

class BinaryIoU(tf.keras.metrics.Metric):
    """Binary IoU metric."""
    def __init__(self, threshold=0.5, name='binary_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred > self.threshold, self.dtype)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)
    
    def result(self):
        return tf.math.divide_no_nan(self.intersection, self.union)
    
    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

class DiceCoefficient(tf.keras.metrics.Metric):
    """Dice coefficient metric."""
    def __init__(self, threshold=0.5, smooth=1e-6, name='dice_coefficient', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.smooth = smooth
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred > self.threshold, self.dtype)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        
        self.intersection.assign_add(intersection)
        self.total.assign_add(total)
    
    def result(self):
        return (2.0 * self.intersection + self.smooth) / (self.total + self.smooth)
    
    def reset_state(self):
        self.intersection.assign(0.0)
        self.total.assign(0.0)

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
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 1:
            logger.info(f"Found {len(gpus)} GPUs. Using MirroredStrategy.")
            return tf.distribute.MirroredStrategy()
        elif len(gpus) == 1:
            logger.info("Found 1 GPU. Using OneDeviceStrategy.")
            return tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            logger.info("No GPUs found, using default CPU strategy.")
            return tf.distribute.get_strategy()
    except Exception as e:
        logger.error(f"Could not initialize distribution strategy: {e}")
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
    
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    
    logger.info(f"Number of replicas: {strategy.num_replicas_in_sync}")
    logger.info(f"Per-replica batch size: {per_replica_batch_size}")
    logger.info(f"GLOBAL batch size: {global_batch_size}")
    
    steps_per_epoch = num_train // global_batch_size
    validation_steps = num_val // global_batch_size if val_ds else None
    
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")

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
        
        peak_lr = 1e-4
        warmup_steps = 1000
        
        boundaries = [warmup_steps] 
        values = [peak_lr / 2, peak_lr]
        
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries,
            values=values
        )
        
        optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)
        
        model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_list)
        logger.info("Model compiled with Adam optimizer and a step-based warm-up schedule (Core TF).")
    
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