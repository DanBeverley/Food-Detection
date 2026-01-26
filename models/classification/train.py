import sys
import yaml
import argparse
import logging
from pathlib import Path
import os
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
    base_model_name = {
        'EfficientNetV2B0': 'efficientnetv2-b0',
        'MobileNetV2': 'mobilenetv2_1.00_224',
        'MobileNetV3Small': 'mobilenetv3_small'
    }.get(architecture)
    base_model = model.get_layer(name=base_model_name)
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

    new_lr = optimizer_cfg.get('stage2_learning_rate', 1e-4)
    model.optimizer.learning_rate.assign(new_lr)
    logger.info(f"Set learning rate for Stage 2 to: {new_lr}")

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