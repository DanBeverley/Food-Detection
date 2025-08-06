# new_model.py

import tensorflow as tf
from keras import models, layers, applications
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def build_classification_model(num_classes: int, config: Dict) -> models.Model:
    """Builds a regularized classification model for preventing overfitting."""
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    
    architecture = model_cfg.get('architecture', 'EfficientNetV2B0')
    image_size = tuple(data_cfg.get('image_size', [224, 224]))
    use_pretrained = model_cfg.get('use_pretrained_weights', True)
    fine_tune = model_cfg.get('fine_tune', True)
    fine_tune_layers = model_cfg.get('fine_tune_layers', 20)
    weights = 'imagenet' if use_pretrained else None

    input_layer = layers.Input(shape=(*image_size, 3), name='rgb_input')

    if architecture == "EfficientNetV2B0":
        preprocess_fn = applications.efficientnet_v2.preprocess_input
        base_model = applications.EfficientNetV2B0(input_shape=(*image_size, 3), include_top=False, weights=weights)
    elif architecture == "MobileNetV2":
        preprocess_fn = applications.mobilenet_v2.preprocess_input
        base_model = applications.MobileNetV2(input_shape=(*image_size, 3), include_top=False, weights=weights)
    elif architecture == "MobileNetV3Small":
        preprocess_fn = applications.mobilenet_v3.preprocess_input
        base_model = applications.MobileNetV3Small(input_shape=(*image_size, 3), include_top=False, weights=weights)
    else:
        logger.warning(f"Architecture {architecture} not supported, using EfficientNetV2B0")
        preprocess_fn = applications.efficientnet_v2.preprocess_input
        base_model = applications.EfficientNetV2B0(input_shape=(*image_size, 3), include_top=False, weights=weights)

    preprocessing_layer = layers.Lambda(preprocess_fn, name='preprocessing')(input_layer)

    if fine_tune:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        logger.info(f"Fine-tuning enabled. Unfreezing the top {fine_tune_layers} layers.")
    else:
        base_model.trainable = False
        logger.info("Fine-tuning disabled. Freezing all backbone layers.")

    x = base_model(preprocessing_layer, training=fine_tune)

    head_cfg = model_cfg.get('classification_head', {})
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    for i, layer_info in enumerate(head_cfg.get('dense_layers', [])):
        units = layer_info.get('units', 256)
        activation = layer_info.get('activation', 'relu')
        dropout_rate = layer_info.get('dropout', 0.5)
        
        x = layers.Dense(
            units, 
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            name=f'head_dense_{i}'
        )(x)
        
        if layer_info.get('batch_norm', True):
            x = layers.BatchNormalization()(x)
            
        x = layers.Activation(activation)(x)
        x = layers.Dropout(dropout_rate)(x)

    output_layer = layers.Dense(
        num_classes, 
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(0.02),
        activity_regularizer=tf.keras.regularizers.l1(0.01),
        name='predictions'
    )(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    logger.info(f"Model '{model.name}' built with regularization.")
    
    return model