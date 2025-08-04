# new_model.py

import tensorflow as tf
from keras import models, layers, applications
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def build_classification_model(num_classes: int, config: Dict) -> models.Model:
    """Builds a self-contained Keras model with augmentation and preprocessing."""
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    
    architecture = model_cfg.get('architecture', 'MobileNetV2')
    image_size = tuple(data_cfg.get('image_size', [224, 224]))
    use_pretrained = model_cfg.get('use_pretrained_weights', True)
    fine_tune = model_cfg.get('fine_tune', True)
    fine_tune_layers = model_cfg.get('fine_tune_layers', 20)
    weights = 'imagenet' if use_pretrained else None

    # --- 1. Define Input Layer ---
    # The model expects raw images in the [0, 255] range.
    input_layer = layers.Input(shape=(*image_size, 3), name='rgb_input')

    # --- 2. Define Augmentation Layers (Optional) ---
    aug_cfg = data_cfg.get('augmentation', {})
    x = input_layer
    if aug_cfg.get('enabled', False):
        logger.info("Adding augmentation layers to the model.")
        augmentation_layers = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(aug_cfg.get('rotation_range', 30) / 360.0),
            layers.RandomZoom(height_factor=aug_cfg.get('zoom_range', 0.2)),
        ], name='augmentation')
        x = augmentation_layers(x)

    # --- 3. Define Preprocessing Layer ---
    # This layer handles the scaling from [0, 255] to the correct range for the model.
    if architecture == "MobileNetV2":
        preprocess_fn = applications.mobilenet_v2.preprocess_input
    elif architecture == "MobileNetV3Small":
        preprocess_fn = applications.mobilenet_v3.preprocess_input
    else: # Fallback
        logger.warning(f"Architecture {architecture} not explicitly handled, using generic scaling.")
        preprocess_fn = lambda img: img / 255.0

    preprocessing_layer = layers.Lambda(preprocess_fn, name='preprocessing')(x)

    # --- 4. Define the Base Model (Backbone) ---
    if architecture == "MobileNetV2":
        base_model = applications.MobileNetV2(input_shape=(*image_size, 3), include_top=False, weights=weights)
    elif architecture == "MobileNetV3Small":
        base_model = applications.MobileNetV3Small(input_shape=(*image_size, 3), include_top=False, weights=weights)
    else: # Fallback
        base_model = applications.MobileNetV2(input_shape=(*image_size, 3), include_top=False, weights=weights)

    # Set trainability for fine-tuning
    if fine_tune:
        base_model.trainable = True
        # Freeze all layers except the last N
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        logger.info(f"Fine-tuning enabled. Unfreezing the top {fine_tune_layers} layers.")
    else:
        base_model.trainable = False
        logger.info("Fine-tuning disabled. Freezing all backbone layers.")

    x = base_model(preprocessing_layer, training=fine_tune)

    # --- 5. Define the Classification Head ---
    head_cfg = model_cfg.get('classification_head', {})
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    for i, layer_info in enumerate(head_cfg.get('dense_layers', [])):
        x = layers.Dense(layer_info.get('units', 512), activation='relu', name=f'head_dense_{i}')(x)
        if layer_info.get('batch_norm', False):
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(layer_info.get('dropout', 0.5))(x)

    # --- 6. Define the Output Layer ---
    output_layer = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    # --- 7. Create and Return the Final Model ---
    model = models.Model(inputs=input_layer, outputs=output_layer)
    logger.info(f"Model '{model.name}' built successfully.")
    
    return model