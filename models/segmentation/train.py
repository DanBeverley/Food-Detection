import os
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
from data import load_segmentation_data
from transformers import TFSeiFormerForSemanticSegmentation
import yaml
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from tensorflow.keras import backend as K

def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def build_segmentation_model(input_shape: Tuple[int, int, int] = (512, 512, 3), num_classes: int = 21, architecture: str = 'ResNet50', loss_function: str = 'dice_loss') -> keras.Model:
    """
    Build and compile a segmentation model (e.g., using a pre-trained backbone like ResNet50 for Mask R-CNN style).

    Args:
        input_shape: Tuple of (height, width, channels) for input images.
        num_classes: Number of segmentation classes (e.g., 21 for COCO-like datasets).
        architecture: Model architecture to use (e.g., ResNet50, SegFormer, BiSeNetV2).
        loss_function: Loss function to use (e.g., dice_loss, sparse_categorical_crossentropy).

    Returns:
        Compiled Keras model for segmentation.

    Raises:
        ValueError: If invalid input shape or num_classes.
    """
    try:
        if num_classes < 1:
            raise ValueError("Number of classes must be at least 1")
        
        if architecture == 'ResNet50':
            base_model = keras.applications.ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')
            base_model.trainable = False

            x = base_model.output
            x = keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)
            x = keras.layers.UpSampling2D(size=(2,2))(x)
            x = keras.layers.Conv2D(num_classes, 1, activation="softmax")(x)

            model = keras.Model(input=base_model.input, outputs=x)
            if loss_function == 'dice_loss':
                loss = dice_loss
            else:
                loss = 'sparse_categorical_crossentropy'
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=loss, metrics=["accuracy"])
            return model
        elif architecture == 'SegFormer':
            # Use Hugging Face Transformers model; ensure 'transformers' is installed
            model = TFSeiFormerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512', num_labels=num_classes)
            return model  # SegFormer may need adaptation; this is a direct integration
        elif architecture == 'BiSeNetV2':
            # Simplified BiSeNet V2 approximation using Keras layers (custom implementation for demonstration)
            inputs = keras.Input(shape=input_shape)
            x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
            x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)  # Basic spatial path; expand for full BiSeNet if needed
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dense(num_classes, activation='softmax')(x)  # Context path simplified
            model = keras.Model(inputs=inputs, outputs=x)
            if loss_function == 'dice_loss':
                loss = dice_loss
            else:
                loss = 'sparse_categorical_crossentropy'
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=loss, metrics=['accuracy'])
            return model
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    except ValueError as e:
        logger.error(f"Model building error: {e}")
        raise

def train_segmentation_model(model: keras.Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, epochs: int = 50, model_dir: str = 'trained_models/segmentation') -> None:
    """
    Train the segmentation model with checkpoints and early stopping.

    Args:
        model: Compiled Keras model for segmentation.
        train_dataset: Training tf.data.Dataset yielding (image, mask) pairs.
        val_dataset: Validation tf.data.Dataset yielding (image, mask) pairs.
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
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
        ]
        
        os.makedirs(model_dir, exist_ok=True)
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)
        logger.info("Segmentation training completed. History saved.")
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

def ensemble_predict(models:list, dataset:tf.data.Dataset) -> np.ndarray:
    predictions = []
    for model in models:
        preds = model.predict(dataset)
        predictions.append(preds)
    averaged_preds = np.mean(predictions, axis=0)
    return averaged_preds  # For use in evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train segmentation model for food datasets.")
    parser.add_argument('--metadata_path', required=True, help="Path to JSON metadata from data loading.")
    parser.add_argument('--data_dir', required=True, help="Root directory of processed images and masks (e.g., data/segmentation/processed/).")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--image_size', nargs=2, type=int, default=[512, 512], help="Image and mask dimensions (height width).")
    parser.add_argument('--split_ratio', type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument('--architecture', type=str, default='ResNet50', help='Model architecture to use (e.g., ResNet50, SegFormer, BiSeNetV2)')
    parser.add_argument('--loss_function', type=str, default='dice_loss', help='Loss function to use (e.g., dice_loss, sparse_categorical_crossentropy)')
    parser.add_argument('--augment_data', action='store_true', help='Enable data augmentation')
    args = parser.parse_args()
    
    try:
        config_path = 'models/segmentation/config.yaml'  # Assume similar config setup; create if not exist
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            architecture = config.get('model_architecture', 'ResNet50')
            # Other params can be loaded similarly
        except FileNotFoundError:
            logger.warning("Config file not found, using command-line arguments.")
            architecture = args.architecture if hasattr(args, 'architecture') else 'ResNet50'
        
        train_ds, val_ds = load_segmentation_data(args.metadata_path, args.data_dir, tuple(args.image_size), args.batch_size, args.split_ratio)
        if args.augment_data:
            def augment(image, mask):
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_flip_up_down(image)
                image = tf.image.random_brightness(image, max_delta=0.1)
                return image, mask
            train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        model = build_segmentation_model(input_shape=(train_ds.element_spec[0].shape[1], train_ds.element_spec[0].shape[2], 3), num_classes=config.get('num_classes', 21), architecture=architecture, loss_function=args.loss_function)
        train_segmentation_model(model, train_ds, val_ds, args.epochs)
    except Exception as e:
        logger.error(f"Script error: {e}")
        exit(1)