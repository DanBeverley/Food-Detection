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
        # RGB Branch
        base_model_rgb = applications.EfficientNetV2B0(
            input_shape=(*image_size, 3), include_top=False, weights=weights
        )
        base_model_rgb._name = "efficientnetv2-b0_rgb"
    elif architecture == "MobileNetV2":
        preprocess_fn = applications.mobilenet_v2.preprocess_input
        base_model_rgb = applications.MobileNetV2(
            input_shape=(*image_size, 3), include_top=False, weights=weights
        )
        base_model_rgb._name = "mobilenetv2_rgb"
    elif architecture == "MobileNetV3Small":
        preprocess_fn = applications.mobilenet_v3.preprocess_input
        base_model_rgb = applications.MobileNetV3Small(
            input_shape=(*image_size, 3), include_top=False, weights=weights
        )
        base_model_rgb._name = "mobilenetv3small_rgb"
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    base_model_rgb.trainable = True

    # RGB Processing
    preprocessing_layer = layers.Lambda(preprocess_fn, name='preprocessing')(input_layer)
    x_rgb = base_model_rgb(preprocessing_layer, training=True)
    x_rgb = layers.GlobalAveragePooling2D(name='global_avg_pool_rgb')(x_rgb)

    # Check for Depth usage
    use_depth = model_cfg.get('use_depth', False)
    
    if use_depth:
        logger.info("Building Multimodal RGB-D Model")
        depth_input = layers.Input(shape=(*image_size, 1), name='depth_input')
        
        # Depth Branch - Replicate depth to 3 channels to use pretrained weights or simple convs
        # Using a smaller backbone for depth is usually sufficient
        depth_backbone_name = model_cfg.get('depth_backbone', 'MobileNetV3Small')
        
        # Depth preprocessing: repeat channels
        x_depth = layers.Concatenate(axis=-1)([depth_input, depth_input, depth_input]) # 1 -> 3 channels
        
        if depth_backbone_name == "MobileNetV3Small":
             base_model_depth = applications.MobileNetV3Small(
                input_shape=(*image_size, 3), include_top=False, weights='imagenet' if use_pretrained else None
            )
             x_depth = layers.Rescaling(255.0)(x_depth)
             x_depth = applications.mobilenet_v3.preprocess_input(x_depth)
             
        else:
            # Fallback to simple CNN if not using a backbone
            x_depth = layers.Conv2D(32, 3, padding='same', activation='relu')(depth_input)
            x_depth = layers.MaxPooling2D()(x_depth)
            x_depth = layers.Conv2D(64, 3, padding='same', activation='relu')(x_depth)
            x_depth = layers.MaxPooling2D()(x_depth)
            base_model_depth = models.Model(inputs=depth_input, outputs=x_depth)

        base_model_depth.trainable = True
        base_model_depth._name = "depth_backbone"
        
        x_depth_feat = base_model_depth(x_depth, training=True)
        x_depth_feat = layers.GlobalAveragePooling2D(name='global_avg_pool_depth')(x_depth_feat)
        
        # Fusion
        x = layers.Concatenate(name='rgb_depth_fusion')([x_rgb, x_depth_feat])
        inputs = {'rgb_input': input_layer, 'depth_input': depth_input}
    else:
        logger.info("Building RGB-Only Model")
        x = x_rgb
        inputs = input_layer

    head_cfg = model_cfg.get('classification_head', {})
    l2_value = head_cfg.get('l2_regularization', 1e-5)
    dropout_rate = head_cfg.get('dropout', 0.2)

    # Classification Head (Shared)
    for i, layer_info in enumerate(head_cfg.get('dense_layers', [])):
        x = layers.Dense(
            layer_info.get('units', 512),
            activation=layer_info.get('activation','relu'),
            kernel_regularizer=tf.keras.regularizers.l2(l2_value),
            name=f'head_dense_{i}'
        )(x)
        if layer_info.get('batch_norm', False):
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)

    output_layer = layers.Dense(num_classes, activation='softmax', name='predictions', dtype='float32')(x)

    model = models.Model(inputs=inputs, outputs=output_layer)
    logger.info(f"Model '{model.name}' built successfully.")
    
    return model