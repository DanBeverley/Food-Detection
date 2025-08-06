import tensorflow as tf
from keras import models, layers, applications
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def build_classification_model(num_classes: int, config: Dict) -> models.Model:
    """Builds a classification model. Trainable status is controlled by the training script."""
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    
    architecture = model_cfg.get('architecture', 'EfficientNetV2B0')
    image_size = tuple(data_cfg.get('image_size', [224, 224]))
    use_pretrained = model_cfg.get('use_pretrained_weights', True)
    weights = 'imagenet' if use_pretrained else None

    input_layer = layers.Input(shape=(*image_size, 3), name='rgb_input')

    if architecture == "EfficientNetV2B0":
        preprocess_fn = applications.efficientnet_v2.preprocess_input
        base_model = applications.EfficientNetV2B0(
            input_shape=(*image_size, 3), include_top=False, weights=weights
        )
    elif architecture == "MobileNetV2":
        preprocess_fn = applications.mobilenet_v2.preprocess_input
        base_model = applications.MobileNetV2(
            input_shape=(*image_size, 3), include_top=False, weights=weights
        )
    elif architecture == "MobileNetV3Small":
        preprocess_fn = applications.mobilenet_v3.preprocess_input
        base_model = applications.MobileNetV3Small(
            input_shape=(*image_size, 3), include_top=False, weights=weights
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    base_model.trainable = True

    preprocessing_layer = layers.Lambda(preprocess_fn, name='preprocessing')(input_layer)
    x = base_model(preprocessing_layer, training=True)

    head_cfg = model_cfg.get('classification_head', {})
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    for i, layer_info in enumerate(head_cfg.get('dense_layers', [])):
        x = layers.Dense(layer_info.get('units', 512), activation=layer_info.get('activation','relu'), name=f'head_dense_{i}')(x)
        if layer_info.get('batch_norm', False):
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
        x = layers.Dropout(layer_info.get('dropout', 0.5), name=f'dropout_{i}')(x)

    output_layer = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    logger.info(f"Model '{model.name}' built successfully with all layers initially trainable.")
    
    return model