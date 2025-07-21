import os
import sys
import yaml
import logging
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
from data import load_segmentation_data 
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def set_mixed_precision_policy(config, strategy):
    mixed_precision_enabled = config.get('training', {}).get('mixed_precision', False)
    if mixed_precision_enabled:
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled with policy: mixed_bfloat16")
    else:
        logger.info("Mixed precision disabled")

def initialize_strategy():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPUs. Using OneDeviceStrategy.")
        return tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        logger.info("No GPUs found, using CPU strategy.")
        return tf.distribute.get_strategy()

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

class TqdmProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def on_epoch_begin(self, epoch, logs=None):
        logger.info(f"Epoch {epoch + 1}")
        self.progress_bar = tqdm(total=self.params['steps'], desc=f"Epoch {epoch + 1}")

    def on_batch_end(self, batch, logs=None):
        if self.progress_bar:
            self.progress_bar.update(1)
            if logs:
                desc = f"Epoch {self.progress_bar.desc.split()[-1]} - "
                desc += " - ".join([f"{k}: {v:.4f}" for k, v in logs.items() if k != 'batch' and k != 'size'])
                self.progress_bar.set_description(desc)

    def on_epoch_end(self, epoch, logs=None):
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None

class StableIoU(tf.keras.metrics.Metric):
    def __init__(self, from_logits=True, name='stable_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        if self.from_logits:
            y_pred = tf.nn.sigmoid(tf.nn.tanh(y_pred))
        
        y_pred = tf.cast(y_pred > 0.5, self.dtype)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        
        self.intersection.assign_add(intersection)
        self.union.assign_add(union + tf.keras.backend.epsilon())

    def result(self):
        return self.intersection / self.union
    
    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

def build_simple_fused_model(output_channels: int, image_size: tuple, model_config: dict, data_config: dict):
    logger.info("Building simple fused model")
    rgb_input = layers.Input(shape=[*image_size, 3], name='rgb_input')
    depth_input = layers.Input(shape=[*image_size, 3], name='depth_input')
    num_points = data_config.get('modalities_preprocessing', {}).get('point_cloud',{}).get('num_points', 4096)
    pc_input = layers.Input(shape=[num_points, 3], name='pc_input')
    input_layers_dict = {'rgb_input': rgb_input, 'depth_input': depth_input, 'pc_input': pc_input}

    # Internal preprocessing - model handles scaling
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
    
    # Final output - NO TANH ACTIVATION
    outputs = layers.Conv2D(output_channels, 1, padding="same", name='final_logits')(u3)

    model = tf.keras.Model(inputs=input_layers_dict, outputs=outputs)
    return model

def main():
    project_root = _get_project_root()
    strategy = initialize_strategy()
    logger.info(f"Training will use strategy: {strategy.__class__.__name__}")
    
    config_path = project_root / 'models' / 'segmentation' / 'config.yaml'
    config = load_config(str(config_path))
    
    # Disable mixed precision for stability
    logger.info("Mixed precision disabled for stability")
    
    train_ds, val_ds, test_ds, num_train, num_val, num_test, num_classes = load_segmentation_data(config)
    if train_ds is None:
        logger.error("Data loading failed. Exiting.")
        return

    data_cfg = config.get('data', {})
    per_replica_batch_size = data_cfg.get('batch_size', 16)
    steps_per_epoch = num_train // per_replica_batch_size
    validation_steps = num_val // per_replica_batch_size if val_ds else None

    with strategy.scope():
        model = build_simple_fused_model(
            output_channels=num_classes, 
            image_size=tuple(data_cfg.get('image_size')),
            model_config=config.get('model', {}),
            data_config=data_cfg
        )
        
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics_list = [
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", from_logits=True),
            StableIoU(name='stable_iou', from_logits=True)
        ]
        optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
        
        model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_list)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_dir_rel = config.get('paths', {}).get('model_save_dir', 'trained_models/segmentation')
    model_dir_abs = project_root / model_save_dir_rel
    model_dir_abs.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        TqdmProgressCallback(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir_abs / f'simple_model_{timestamp}.h5'),
            monitor='val_stable_iou',
            save_best_only=True,
            verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_stable_iou',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
    ]
    
    # Single training run - no staged training
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=0
    )

    logger.info("Training finished.")
    
    final_model_path = model_dir_abs / f'simple_segmentation_final_{timestamp}.h5'
    model.save(str(final_model_path)) 
    logger.info(f"Final trained model saved to: {final_model_path}")

    if test_ds and num_test > 0:
        logger.info("Evaluating model on the test set...")
        test_results = model.evaluate(test_ds, verbose=1)
        logger.info("Test Set Evaluation Results:")
        if isinstance(test_results, list):
            for metric_name, value in zip(model.metrics_names, test_results):
                logger.info(f"  {metric_name}: {value}")
        else:
            logger.info(f"  loss: {test_results}")

if __name__ == '__main__':
    main()