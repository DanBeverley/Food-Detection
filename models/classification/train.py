import sys
import yaml
import argparse
import logging
from pathlib import Path
import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from data import load_classification_data
from model import build_classification_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def create_regularized_model(num_classes, config):
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    
    architecture = model_cfg.get('architecture', 'EfficientNetV2B0')
    image_size = tuple(data_cfg.get('image_size', [224, 224]))
    use_pretrained = model_cfg.get('use_pretrained_weights', True)
    weights = 'imagenet' if use_pretrained else None
    
    inputs = layers.Input(shape=(*image_size, 3), name='input')
    
    augmented = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
    ], name='augmentation')(inputs)
    
    if architecture == "EfficientNetV2B0":
        base_model = tf.keras.applications.EfficientNetV2B0(
            input_shape=(*image_size, 3),
            include_top=False,
            weights=weights,
            input_tensor=augmented
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.6)(x)
    
    residual = layers.Dense(128)(x)
    x = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Add()([x, residual])
    x = layers.Dropout(0.7)(x)
    
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(0.02),
        activity_regularizer=tf.keras.regularizers.l1(0.01)
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model, base_model


def main(args):
    MODEL_CHECKPOINT_PATH = "/kaggle/working/best_classification_model.keras"
    
    logger.info(f"Loading configuration from: {args.config}")
    config = yaml.safe_load(Path(args.config).read_text())
    
    config['data']['batch_size'] = 32
    
    train_ds, val_ds, num_train, num_val, class_names, num_classes = load_classification_data(config)
    if not train_ds:
        logger.critical("Data loading failed. Cannot proceed.")
        sys.exit(1)
    
    model, base_model = create_regularized_model(num_classes, config)
    
    logger.info("Stage 1: Training classification head only")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        steps_per_epoch=num_train // 32,
        validation_steps=num_val // 32
    )
    
    logger.info("Stage 2: Progressive unfreezing")
    base_model.trainable = True
    
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
        ]
    )
    
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
            patience=15,
            mode='max',
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            mode='min',
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-4 * (0.95 ** epoch)
        ),
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        initial_epoch=5,
        steps_per_epoch=num_train // 32,
        validation_steps=num_val // 32,
        callbacks=callbacks_list,
        verbose=1
    )
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    overfitting_gap = final_train_acc - final_val_acc
    
    logger.info(f"Final training accuracy: {final_train_acc:.4f}")
    logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
    logger.info(f"Overfitting gap: {overfitting_gap:.4f}")
    
    if overfitting_gap > 0.2:
        logger.warning("Model is still overfitting significantly.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a food classification model with anti-overfitting measures.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--base_data_dir", type=str, default=None, help="Absolute path to images.")
    args = parser.parse_args()
    main(args)