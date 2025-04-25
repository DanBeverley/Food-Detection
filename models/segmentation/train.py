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
import tensorflow_addons as tfa  # For potential additional metrics if needed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input, Dense

def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def mean_iou(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.int32)
    intersection = tf.reduce_sum(y_true * tf.cast(tf.equal(y_true, y_pred), y_true.dtype))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return tf.reduce_mean(intersection / (union + tf.keras.backend.epsilon()))

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
            inputs = Input(shape=input_shape)
            # Encoder
            c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
            c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
            p1 = MaxPooling2D((2, 2))(c1)
            c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
            c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
            p2 = MaxPooling2D((2, 2))(c2)
            # Bottleneck
            c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
            c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
            # Decoder with skip connections
            u4 = UpSampling2D((2, 2))(c3)
            u4 = Concatenate()([u4, c2])
            c4 = Conv2D(128, 3, activation='relu', padding='same')(u4)
            c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)
            u5 = UpSampling2D((2, 2))(c4)
            u5 = Concatenate()([u5, c1])
            c5 = Conv2D(64, 3, activation='relu', padding='same')(u5)
            c5 = Conv2D(64, 3, activation='relu', padding='same')(c5)
            outputs = Conv2D(num_classes, 1, activation='softmax')(c5)
            model = keras.Model(inputs=[inputs], outputs=[outputs])
        elif architecture == 'SegFormer':
            model = TFSeiFormerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512', num_labels=num_classes)
        elif architecture == 'BiSeNetV2':
            # Improved BiSeNet V2-like architecture with efficiency in mind (using depthwise separable convolutions)
            inputs = Input(shape=input_shape)
            # Spatial path with depthwise separable convolutions for efficiency
            x = keras.layers.SeparableConv2D(64, 3, padding='same', activation='relu')(inputs)
            x = keras.layers.SeparableConv2D(64, 3, padding='same', activation='relu')(x)
            # Context path
            c = keras.layers.GlobalAveragePooling2D()(x)
            c = keras.layers.Dense(128, activation='relu')(c)
            c = keras.layers.Reshape((1, 1, 128))(c)
            c = UpSampling2D(size=(input_shape[0]//8, input_shape[1]//8))(c)  # Approximate upsampling
            c = Conv2D(64, 1, activation='relu')(c)
            # Feature fusion (simplified)
            fused = Concatenate()([x, c])
            fused = Conv2D(64, 3, padding='same', activation='relu')(fused)
            outputs = Conv2D(num_classes, 1, activation='softmax')(fused)
            model = keras.Model(inputs=inputs, outputs=outputs)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Compile with custom loss and metrics for production readiness
        if loss_function == 'dice_loss':
            loss = dice_loss
        else:
            loss = 'sparse_categorical_crossentropy'
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=loss, metrics=['accuracy', mean_iou])  # Add IoU metric for better evaluation
        return model
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
        # Check config for mixed precision
        config_path = 'models/segmentation/config.yaml'
        use_mixed_precision = False
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            use_mixed_precision = config.get('use_mixed_precision', False)
        except FileNotFoundError:
            logger.warning("Config file not found, defaulting to no mixed precision.")
        
        if use_mixed_precision:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            logger.info("Mixed precision enabled.")
        
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-3, decay_steps=epochs * int(train_dataset.cardinality() or 1000))  # Fallback if cardinality is None
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        if use_mixed_precision:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        callbacks = [
            keras.callbacks.LearningRateScheduler(lr_schedule),
            keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'model_{epoch}.h5'), save_best_only=True, monitor='val_loss'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
            keras.callbacks.TensorBoard(log_dir=os.path.join(model_dir, 'logs'), histogram_freq=1)
        ]
        
        os.makedirs(model_dir, exist_ok=True)
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)
        logger.info("Segmentation training completed. History saved.")
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

def augment(image, mask):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))  # Add random rotation
    image = tf.image.random_zoom(image, (0.8, 1.2))  # Add random zoom
    return image, mask

def ensemble_predict(models:list, dataset:tf.data.Dataset) -> np.ndarray:
    predictions = []
    for model in models:
        preds = model.predict(dataset)
        predictions.append(preds)
    averaged_preds = np.mean(predictions, axis=0)
    return averaged_preds 

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
    parser.add_argument('--use_mixed_precision', action='store_true', help='Enable mixed precision training if supported')
    args = parser.parse_args()
    
    try:
        config_path = 'models/segmentation/config.yaml'  
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
            train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        use_mixed_precision = args.use_mixed_precision or (config and config.get('use_mixed_precision', False))
        if use_mixed_precision:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            logger.info("Mixed precision enabled.")
        model = build_segmentation_model(input_shape=(train_ds.element_spec[0].shape[1], train_ds.element_spec[0].shape[2], 3), num_classes=config.get('num_classes', 21), architecture=architecture, loss_function=args.loss_function)
        train_segmentation_model(model, train_ds, val_ds, args.epochs)
    except Exception as e:
        logger.error(f"Script error: {e}")
        exit(1)