import os
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from data import load_classification_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_model(num_classes:int, image_size:Tuple[int, int]) -> keras.Model:
    """
    Build and compile a classification model using EfficientNetV2.

    Args:
        num_classes: Number of output classes.
        image_size: Tuple of (height, width) for input shape.

    Returns:
        Compiled Keras model.

    Raises:
        ValueError: If invalid input shape.
    """
    base_model = keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", input_shape = (*image_size, 3))
    base_model.trainable = False

    model = keras.Sequential([base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax")])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
    return model

def train_model(train_dataset:tf.data.Dataset, val_dataset:tf.data.Dataset,
                epochs:int = 50, model_dir:str = "trained_models/classification") -> None:
    """
    Train the classification model with checkpoints and early stopping.

    Args:
        train_dataset: Training tf.data.Dataset.
        val_dataset: Validation tf.data.Dataset.
        epochs: Number of training epochs.
        model_dir: Directory to save model checkpoints.

    Raises:
        RuntimeError: If training fails.
    """
    try:
        num_classes = len(np.unique([label for _, label in train_dataset.unbatch().as_numpy_iterator()]))
        model = build_model(num_classes, image_size = (train_dataset.element_spec[0].shape[0], train_dataset.element_spec[0].shape[1]))
        callbacks = [
            keras.callbacks.ModelCheckpoint(os.path.join(model_dir, "model_{epoch}.h5"), save_best_only=True, monitor="val_loss"),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.2)
        ]
        os.makedirs(model_dir, exist_ok=True)
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)
        logger.info("Training completed. History saved")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train classification model for food datasets.")
    parser.add_argument('--metadata_path', required=True, help="Path to JSON metadata from data loading.")
    parser.add_argument('--data_dir', required=True, help="Root directory of processed images (e.g., data/classification/processed/).")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--image_size', nargs=2, type=int, default=[224, 224], help="Image dimensions (height width).")
    args = parser.parse_args()
    
    try:
        train_dataset, val_dataset = load_classification_data(args.metadata_path, args.data_dir, tuple(args.image_size), args.batch_size)
        train_model(train_dataset, val_dataset, args.epochs)
    except Exception as e:
        logger.error(f"Script error: {e}")
        exit(1)