import tensorflow as tf
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)

def build_simple_fused_model(output_channels: int, image_size: tuple, data_config: dict):
    logger.info("Building simple fused model")
    rgb_input = layers.Input(shape=[*image_size, 3], name='rgb_input')
    depth_input = layers.Input(shape=[*image_size, 3], name='depth_input')
    num_points = data_config.get('modalities_preprocessing', {}).get('point_cloud',{}).get('num_points', 4096)
    pc_input = layers.Input(shape=[num_points, 3], name='pc_input')
    input_layers_dict = {'rgb_input': rgb_input, 'depth_input': depth_input, 'pc_input': pc_input}

    rgb_scaled = layers.Rescaling(1./127.5, offset=-1)(rgb_input)
    depth_scaled = layers.Rescaling(1./127.5, offset=-1)(depth_input)
    
    def conv_block(x, filters, name):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv1")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        return x

    # Encoders
    x1 = conv_block(rgb_scaled, 32, "rgb_enc1")
    p1 = layers.MaxPooling2D(2)(x1)
    x2 = conv_block(p1, 64, "rgb_enc2")
    p2 = layers.MaxPooling2D(2)(x2)
    x3 = conv_block(p2, 128, "rgb_enc3")
    rgb_features = layers.MaxPooling2D(2)(x3)

    d1 = conv_block(depth_scaled, 32, "depth_enc1")
    dp1 = layers.MaxPooling2D(2)(d1)
    d2 = conv_block(dp1, 64, "depth_enc2")
    dp2 = layers.MaxPooling2D(2)(d2)
    d3 = conv_block(dp2, 128, "depth_enc3")
    depth_features = layers.MaxPooling2D(2)(d3)
    
    # Point cloud encoder
    pc_x = layers.Conv1D(64, 1, activation='relu')(pc_input)
    pc_x = layers.BatchNormalization()(pc_x)
    pc_x = layers.Conv1D(128, 1, activation='relu')(pc_x)
    pc_features_vec = layers.GlobalMaxPooling1D()(pc_x)

    pc_features_spatial = layers.Dense(32 * 32 * 64, activation='relu')(pc_features_vec)
    pc_features_spatial = layers.Reshape((32, 32, 64))(pc_features_spatial)
    
    fused = layers.Concatenate()([rgb_features, depth_features, pc_features_spatial])
    fused = conv_block(fused, 256, "fused_bridge")

    # Decoder
    def upsample_block(x, filters, name):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same", name=f"{name}_up")(x)
        x = conv_block(x, filters, name)
        return x

    u1 = upsample_block(fused, 128, "dec1")
    u2 = upsample_block(u1, 64, "dec2")
    u3 = upsample_block(u2, 32, "dec3")
    
    outputs = layers.Conv2D(output_channels, 1, padding="same", name='final_logits')(u3)

    model = tf.keras.Model(inputs=input_layers_dict, outputs=outputs)
    return model