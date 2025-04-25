import os
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from data import load_classification_data
import yaml
import tensorflow_addons as tfa
from tensorflow.keras import backend as K

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def build_model(num_classes:int, image_size:Tuple[int, int], architecture:str = 'EfficientNetV2B0', loss_function:str = 'sparse_categorical_crossentropy') -> keras.Model:
    """
    Build and compile a classification model using EfficientNetV2.

    Args:
        num_classes: Number of output classes.
        image_size: Tuple of (height, width) for input shape.
        architecture: Model architecture to use (e.g., EfficientNetV2B0, ConvNeXt, MobileViT)
        loss_function: Loss function to use (e.g., sparse_categorical_crossentropy, focal_loss)

    Returns:
        Compiled Keras model.

    Raises:
        ValueError: If invalid input shape or unsupported architecture.
    """
    if architecture == 'EfficientNetV2B0':
        base_model = keras.applications.EfficientNetV2B0(include_top=False, input_shape=(*image_size, 3), weights='imagenet')
    elif architecture == 'ConvNeXt':
        base_model = keras.applications.ConvNeXtBase(include_top=False, input_shape=(*image_size, 3), weights='imagenet')
    elif architecture == 'MobileViT':
        base_model = keras.applications.MobileViT_S(input_shape=(*image_size, 3), weights='imagenet')  # Use Small variant for efficiency
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    base_model.trainable = False

    model = keras.Sequential([base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax")])
    
    if loss_function == 'focal_loss':
        loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, alpha=0.25, gamma=2.0)
    else:
        loss = loss_function
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=loss, metrics=['accuracy'])
    return model

def train_model(model:keras.Model, train_dataset:tf.data.Dataset, val_dataset:tf.data.Dataset, epochs:int = 50, model_dir:str = 'trained_models/classification') -> None:
    """
    Train the classification model with checkpoints and early stopping.

    Args:
        model: Compiled Keras model.
        train_dataset: Training tf.data.Dataset.
        val_dataset: Validation tf.data.Dataset.
        epochs: Number of training epochs.
        model_dir: Directory to save model checkpoints.

    Raises:
        RuntimeError: If training fails.
    """
    try:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-3, decay_steps=epochs * train_dataset.cardinality().numpy())
        callbacks = [
            keras.callbacks.LearningRateScheduler(lr_schedule),
            keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'model_{epoch}.h5'), save_best_only=True, monitor='val_loss'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.2)
        ]
        os.makedirs(model_dir, exist_ok=True)
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)
        logger.info("Training completed. History saved")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

def ensemble_predict(models:list, dataset:tf.data.Dataset) -> np.ndarray:
    predictions = []
    for model in models:
        preds = model.predict(dataset)
        predictions.append(preds)
    averaged_preds = np.mean(predictions, axis=0)
    return averaged_preds  # Can be used in evaluation for ensembling

def main():
    parser = argparse.ArgumentParser(description="Train classification model for food datasets.")
    parser.add_argument('--metadata_path', required=True, help="Path to JSON metadata from data loading.")
    parser.add_argument('--data_dir', required=True, help="Root directory of processed images (e.g., data/classification/processed/).")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--image_size', nargs=2, type=int, default=[224, 224], help="Image dimensions (height width).")
    parser.add_argument('--architecture', type=str, default='EfficientNetV2B0', help='Model architecture to use (e.g., EfficientNetV2B0, ConvNeXt, MobileViT)')
    parser.add_argument('--loss_function', type=str, default='sparse_categorical_crossentropy', help='Loss function to use (e.g., sparse_categorical_crossentropy, focal_loss)')
    parser.add_argument('--augment_data', action='store_true', help='Enable data augmentation')

    args = parser.parse_args()
    
    # Load config from YAML if available, otherwise use args
    config_path = 'models/classification/config.yaml'  # Path to config file
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        architecture = config.get('model_architecture', 'EfficientNetV2B0')
        image_size = tuple(config.get('image_size', [224, 224]))
        batch_size = config.get('batch_size', 32)
        epochs = config.get('epochs', 50)
        use_mixed_precision = config.get('use_mixed_precision', False) and tf.config.list_physical_devices('GPU')
    except FileNotFoundError:
        logger.warning("Config file not found, using command-line arguments only.")
        architecture = args.architecture if hasattr(args, 'architecture') else 'EfficientNetV2B0'
        image_size = tuple(args.image_size) if hasattr(args, 'image_size') else (224, 224)
        batch_size = args.batch_size if hasattr(args, 'batch_size') else 32
        epochs = args.epochs if hasattr(args, 'epochs') else 50
        use_mixed_precision = False  # Default to off if no config

    if use_mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        logger.info("Mixed precision enabled.")

    train_dataset, val_dataset = load_classification_data(args.metadata_path, args.data_dir, image_size, batch_size)
    
    if args.augment_data:
        def augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            return image, label
        train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    model = build_model(num_classes=256, image_size=image_size, architecture=architecture, loss_function=args.loss_function)
    train_model(model, train_dataset, val_dataset, epochs)

if __name__ == '__main__':
    main()