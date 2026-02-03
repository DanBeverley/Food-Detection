import sys
import yaml
import argparse
import logging
from pathlib import Path
import os
import math
import tensorflow as tf
import keras
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

from data import load_classification_data
from model import build_classification_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup followed by cosine decay.
    
    This helps stabilize training in the early epochs when gradients can be noisy,
    then smoothly decays the learning rate for better convergence.
    """
    
    def __init__(
        self,
        base_learning_rate: float,
        total_steps: int,
        warmup_steps: int,
        min_learning_rate: float = 1e-7
    ):
        super().__init__()
        self.base_learning_rate = base_learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_learning_rate = min_learning_rate
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        
        # Linear warmup
        warmup_lr = self.base_learning_rate * (step / tf.maximum(warmup_steps, 1.0))
        
        # Cosine decay after warmup
        decay_steps = total_steps - warmup_steps
        decay_step = tf.maximum(step - warmup_steps, 0.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(math.pi * decay_step / tf.maximum(decay_steps, 1.0)))
        decay_lr = self.min_learning_rate + (self.base_learning_rate - self.min_learning_rate) * cosine_decay
        
        # Use warmup LR during warmup phase, decay LR after
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)
    
    def get_config(self):
        return {
            'base_learning_rate': self.base_learning_rate,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'min_learning_rate': self.min_learning_rate
        }


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
    
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    optimizer_cfg = config.get('optimizer', {})
    training_cfg = config.get('training', {})
    
    if args.base_data_dir:
        data_cfg['base_data_dir'] = args.base_data_dir
    config['data'] = data_cfg
    
    train_ds, val_ds, num_train, num_val, class_names, num_classes = load_classification_data(config)
    if not train_ds:
        logger.critical("Data loading failed. Cannot proceed.")
        sys.exit(1)
    logger.info(f"Data loaded: {num_train} train, {num_val} val samples.")

    with strategy.scope():
        model = build_classification_model(num_classes, config)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=optimizer_cfg.get('stage1_learning_rate', 1e-3),
            clipnorm=optimizer_cfg.get('clipnorm', 1.0)
        )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=config['loss']['params']['label_smoothing']),
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')],
            jit_compile=False
        )

    logger.info("--- Stage 1: Training classification head only ---")
    architecture = model_cfg.get('architecture', 'EfficientNetV2B0')
    
    # Keras layer names are derived from the class name (usually) if _name is ignored
    # Based on error logs: efficientnetv2-b0_rgb -> efficientnetv2-b0
    base_model_name = {
        'EfficientNetV2B0': 'efficientnetv2-b0',
        'MobileNetV2': 'mobilenetv2_1.00_224', # Typical default, might need adjustment if used
        'MobileNetV3Small': 'MobileNetV3Small'
    }.get(architecture)
    
    # Fallback search if exact name fails
    try:
        base_model = model.get_layer(name=base_model_name)
    except ValueError:
        # If strict lookup fails, try finding the backbone by class type partial match
        logging.warning(f"Layer {base_model_name} not found. Searching for backbone layer...")
        candidates = [l.name for l in model.layers if 'efficientnet' in l.name or 'mobilenet' in l.name]
        if candidates:
             # Heuristic: pick the first one that looks like a backbone (not fusion/dense)
             base_model_name = candidates[0]
             logging.info(f"Found candidate backbone layer: {base_model_name}")
             base_model = model.get_layer(name=base_model_name)
        else:
             raise

    base_model.trainable = False
    
    model.optimizer.learning_rate.assign(optimizer_cfg.get('stage1_learning_rate', 1e-3))
    
    stage1_epochs = training_cfg.get('stage1_epochs', 5)
    if stage1_epochs > 0:
        logger.info(f"Starting Stage 1 fit for {stage1_epochs} epochs.")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=stage1_epochs,
            steps_per_epoch=num_train // data_cfg['batch_size'],
            validation_steps=num_val // data_cfg['batch_size']
        )
    
    logger.info("--- Stage 2: Fine-tuning the model ---")
    base_model.trainable = True
    num_fine_tune_layers = model_cfg.get('stage2_trainable_layers', 0)
    if num_fine_tune_layers > 0:
        for layer in base_model.layers[:-num_fine_tune_layers]:
            layer.trainable = False
        logger.info(f"Fine-tuning top {num_fine_tune_layers} layers.")

    # Calculate warmup schedule parameters
    stage2_epochs = training_cfg.get('stage2_epochs', 100)
    steps_per_epoch = num_train // data_cfg['batch_size']
    total_steps = stage2_epochs * steps_per_epoch
    warmup_epochs = training_cfg.get('warmup_epochs', 3)
    warmup_steps = warmup_epochs * steps_per_epoch
    
    base_lr = optimizer_cfg.get('stage2_learning_rate', 1e-4)
    use_warmup = training_cfg.get('use_warmup', True)
    
    if use_warmup:
        lr_schedule = WarmupCosineDecay(
            base_learning_rate=base_lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_learning_rate=optimizer_cfg.get('min_learning_rate', 1e-7)
        )
        model.optimizer.learning_rate = lr_schedule
        logger.info(f"Using WarmupCosineDecay: base_lr={base_lr}, warmup_epochs={warmup_epochs}, total_epochs={stage2_epochs}")
    else:
        model.optimizer.learning_rate.assign(base_lr)
        logger.info(f"Set constant learning rate for Stage 2 to: {base_lr}")

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_CHECKPOINT_PATH, monitor='val_accuracy', save_best_only=True,
            mode='max', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, mode='max',
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, mode='min',
            min_lr=1e-7, verbose=1
        ),
    ]
    
    logger.info("Starting Stage 2 fit.")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training_cfg.get('stage2_epochs', 100),
        initial_epoch=stage1_epochs,
        steps_per_epoch=num_train // data_cfg['batch_size'],
        validation_steps=num_val // data_cfg['batch_size'],
        callbacks=callbacks_list
    )
    
    if history.history:
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        overfitting_gap = final_train_acc - final_val_acc
        
        logger.info(f"Final training accuracy: {final_train_acc:.4f}")
        logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
        logger.info(f"Overfitting gap: {overfitting_gap:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a food classification model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--base_data_dir", type=str, default=None, help="Absolute path to images.")
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default="/kaggle/working/best_classification_model.keras",
        help="Path to the model checkpoint file."
    )
    args = parser.parse_args()
    main(args)