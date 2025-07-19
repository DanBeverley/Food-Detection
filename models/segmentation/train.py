
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

def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_mixed_precision_policy(config: dict, strategy: tf.distribute.Strategy):
    """Set mixed precision policy based on hardware and config."""
    training_config = config.get('training', {})
    if training_config.get('use_mixed_precision', False):
        policy_name = ''
        if isinstance(strategy, tf.distribute.TPUStrategy):
            policy_name = 'mixed_bfloat16'
            logger.info("TPU detected, using 'mixed_bfloat16' for mixed precision.")
        else:
            gpu_compatible = False
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    details = tf.config.experimental.get_device_details(gpus[0])
                    if details.get('compute_capability', (0,0))[0] >= 7:
                        gpu_compatible = True
            except Exception as e:
                logger.warning(f"Could not get GPU details for mixed precision check: {e}")

            if gpu_compatible:
                policy_name = 'mixed_float16'
                logger.info("Compatible GPU detected, using 'mixed_float16' for mixed precision.")
            else:
                logger.warning("Mixed precision enabled in config, but no compatible GPU (Compute Capability >= 7.0) or TPU found. Mixed precision will not be used effectively for GPU.")
                return

        if policy_name:
            logger.info(f"Setting mixed precision policy to '{policy_name}'.")
            try:
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy(policy_name)
                mixed_precision.set_global_policy(policy)
                logger.info(f"Mixed precision policy set. Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}")
            except Exception as e:
                logger.warning(f"Could not set mixed precision policy: {e}")
    else:
        logger.info("Mixed precision training not enabled in config.")

def initialize_strategy():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPUs. Forcing single-GPU strategy.")
        return tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        logger.info("No GPUs found, using CPU strategy.")
        return tf.distribute.get_strategy()



class TqdmProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback for robust progress tracking and logging during training."""
    
    def on_epoch_begin(self, epoch, logs=None):
        self.pbar = tqdm(total=self.params['steps'], desc=f"Epoch {epoch + 1}/{self.params['epochs']}", unit="step", file=sys.stdout)

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.close()
        log_str = " - ".join([f"{key}: {value:.4f}" for key, value in logs.items()])
        print(f"\nEpoch {epoch + 1} Summary: {log_str}")
        sys.stdout.flush()

    def on_batch_end(self, batch, logs=None):
        self.pbar.update(1)
        if logs:
            self.pbar.set_postfix({key: f"{value:.4f}" for key, value in logs.items()}) 

def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def _test_data_loading(train_dataset: tf.data.Dataset, data_config: dict, num_batches_to_test: int = 2):
    if train_dataset is None:
        logger.error("Training dataset is None. Skipping data loading test.")
        return

    use_depth = data_config.get('use_depth_map', False)
    use_pc = data_config.get('use_point_cloud', False)

    for i, batch_data in enumerate(train_dataset.take(num_batches_to_test)):
        inputs_dict, mask_tensor = batch_data

        if not isinstance(inputs_dict, dict):
            logger.error(f"Expected inputs_dict to be a dict, but got {type(inputs_dict)}. Batch structure might be incorrect.")
            continue

        # RGB Input
        if 'rgb_input' in inputs_dict:
            rgb = inputs_dict['rgb_input']
            try:
                min_val = tf.reduce_min(rgb).numpy()
                max_val = tf.reduce_max(rgb).numpy()
                logger.info(f"  rgb_input: Shape={rgb.shape}, Dtype={rgb.dtype}, Min={min_val}, Max={max_val}")
            except Exception as e:
                logger.warning(f"  rgb_input: Shape={rgb.shape}, Dtype={rgb.dtype}, Stats unavailable: {e}")
        else:
            logger.warning("  rgb_input not found in batch.")

        # Depth Input (if enabled)
        if use_depth:
            if 'depth_input' in inputs_dict:
                depth = inputs_dict['depth_input']
                try:
                    min_val = tf.reduce_min(depth).numpy()
                    max_val = tf.reduce_max(depth).numpy()
                    logger.info(f"  depth_input: Shape={depth.shape}, Dtype={depth.dtype}, Min={min_val}, Max={max_val}")
                except Exception as e:
                    logger.warning(f"  depth_input: Shape={depth.shape}, Dtype={depth.dtype}, Stats unavailable: {e}")
            else:
                logger.warning("  depth_input is expected (use_depth_map=True) but not found in batch.")
        
        # Point Cloud Input (if enabled)
        if use_pc:
            if 'pc_input' in inputs_dict:
                pc = inputs_dict['pc_input']
                try:
                    min_val = tf.reduce_min(pc).numpy()
                    max_val = tf.reduce_max(pc).numpy()
                    logger.info(f"  pc_input: Shape={pc.shape}, Dtype={pc.dtype}, Min={min_val}, Max={max_val}")
                except Exception as e:
                    logger.warning(f"  pc_input: Shape={pc.shape}, Dtype={pc.dtype}, Stats unavailable: {e}")
            else:
                logger.warning("  pc_input is expected (use_point_cloud=True) but not found in batch.")

        # Mask Tensor
        try:
            min_val = tf.reduce_min(mask_tensor).numpy()
            max_val = tf.reduce_max(mask_tensor).numpy()
            logger.info(f"  mask_tensor: Shape={mask_tensor.shape}, Dtype={mask_tensor.dtype}, Min={min_val}, Max={max_val}")
            unique_values, _ = tf.unique(tf.reshape(mask_tensor, [-1]))
            unique_np = unique_values.numpy()
            logger.info(f"  mask_tensor: Unique values={unique_np}")
        except Exception as e:
            logger.warning(f"  mask_tensor: Shape={mask_tensor.shape}, Dtype={mask_tensor.dtype}, Stats unavailable: {e}")
        


def build_simple_fused_model(output_channels: int, image_size: tuple, data_config: dict):
    logger.info("--- Building STABLE simple_fused_model ---")
    rgb_input = layers.Input(shape=[*image_size, 3], name='rgb_input')
    depth_input = layers.Input(shape=[*image_size, 3], name='depth_input')
    num_points = data_config.get('modalities_preprocessing', {}).get('point_cloud',{}).get('num_points', 4096)
    pc_input = layers.Input(shape=[num_points, 3], name='pc_input')
    input_layers_dict = {'rgb_input': rgb_input, 'depth_input': depth_input, 'pc_input': pc_input}

    rgb_scaled = layers.Rescaling(1./127.5, offset=-1)(rgb_input)
    depth_scaled = layers.Rescaling(1./127.5, offset=-1)(depth_input)
    
    def conv_block(x, filters, name):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.ReLU(name=f"{name}_relu1")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.ReLU(name=f"{name}_relu2")(x)
        return x

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
    
    pc_x = layers.Conv1D(64, 1, use_bias=False)(pc_input)
    pc_x = layers.BatchNormalization()(pc_x)
    pc_x = layers.ReLU()(pc_x)
    pc_x = layers.Conv1D(128, 1, use_bias=False)(pc_x)
    pc_x = layers.BatchNormalization()(pc_x)
    pc_x = layers.ReLU()(pc_x)
    pc_features_vec = layers.GlobalMaxPooling1D()(pc_x)

    pc_features_spatial = layers.Dense(32 * 32 * 64, use_bias=False)(pc_features_vec)
    pc_features_spatial = layers.BatchNormalization()(pc_features_spatial)
    pc_features_spatial = layers.ReLU()(pc_features_spatial)
    pc_features_spatial = layers.Reshape((32, 32, 64))(pc_features_spatial)
    
    fused = layers.Concatenate()([rgb_features, depth_features, pc_features_spatial])
    fused = conv_block(fused, 256, "fused_bridge")

    def upsample_block(x, filters, name):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same", name=f"{name}_up")(x)
        x = conv_block(x, filters, name)
        return x

    u1 = upsample_block(fused, 128, "dec1")
    u2 = upsample_block(u1, 64, "dec2")
    u3 = upsample_block(u2, 32, "dec3")
    
    outputs = layers.Conv2D(output_channels, 1, padding="same", name='final_logits')(u3)
    outputs = layers.Activation('tanh', name='constrained_logits')(outputs)

    model = tf.keras.Model(inputs=input_layers_dict, outputs=outputs)
    return model

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

# Custom loss functions
def dice_loss(y_true, y_pred, smooth=1.0):
    """Dice loss for segmentation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for addressing class imbalance."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Clip predictions to prevent log(0)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    
    # Compute cross entropy manually to ensure shape consistency
    ce_loss = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    
    # Compute p_t (predicted probability for the true class)
    p_t = tf.where(y_true == 1, y_pred, 1 - y_pred)
    
    # Compute alpha_t (class weighting)
    alpha_t = tf.where(y_true == 1, alpha, 1 - alpha)
    
    # Compute focal weight
    focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
    
    # Apply focal weight and reduce to scalar
    focal_loss_value = focal_weight * ce_loss
    return tf.reduce_mean(focal_loss_value)

def combined_loss(y_true, y_pred_logits, bce_weight=0.5, dice_weight=0.3, focal_weight=0.2, 
                 label_smoothing=0.1, smooth=1.0, alpha=0.25, gamma=2.0):
    """Combined loss function for better segmentation."""
    
    y_true_float = tf.cast(y_true, tf.float32)
    
    # Use numerically stable binary crossentropy from logits
    bce = tf.keras.losses.binary_crossentropy(
        y_true_float, y_pred_logits, from_logits=True, label_smoothing=label_smoothing
    )
    bce = tf.reduce_mean(bce)
    
    # Convert logits to probabilities for Dice and Focal loss
    y_pred_probs = tf.nn.sigmoid(y_pred_logits)
    
    # Dice loss (already returns scalar)
    dice = dice_loss(y_true_float, y_pred_probs, smooth=smooth)
    
    # Focal loss (already returns scalar)
    focal = focal_loss(y_true_float, y_pred_probs, alpha=alpha, gamma=gamma)
    
    return bce_weight * bce + dice_weight * dice + focal_weight * focal

def main():
    project_root = _get_project_root()
    strategy = initialize_strategy()
    logger.info(f"Training will use strategy: {strategy.__class__.__name__}")
    
    config_path = project_root / 'models' / 'segmentation' / 'config.yaml'
    config = load_config(str(config_path))
    
    logger.info("--- Using STABLE baseline configuration ---")
    
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
            filepath=str(model_dir_abs / f'stable_model_{timestamp}.h5'),
            monitor='val_binary_iou',
            save_best_only=True,
            verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_binary_iou',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
    ]
    
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
    
    final_model_path = model_dir_abs / f'fused_segmentation_final_{timestamp}.h5'
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