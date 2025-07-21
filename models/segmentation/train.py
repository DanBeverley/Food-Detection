
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
        


def build_unet_with_aug_layers(output_channels: int, image_size: tuple, model_config: dict, data_config: dict):
    logger.info("Building U-Net with Augmentation Layers")
    
    # Inputs receive raw [0, 255] data
    rgb_input = layers.Input(shape=[*image_size, 3], name='rgb_input')
    depth_input = layers.Input(shape=[*image_size, 3], name='depth_input')
    num_points = data_config.get('modalities_preprocessing', {}).get('point_cloud', {}).get('num_points', 4096)
    pc_input = layers.Input(shape=[num_points, 3], name='pc_input')
    
    # Augmentation configuration
    aug_cfg = data_config.get('augmentation', {})
    
    # Augmentation layers for images
    aug_rgb = rgb_input
    aug_depth = depth_input
    
    if aug_cfg.get('horizontal_flip', False):
        # Concatenate for synchronized augmentation
        combined = layers.Concatenate()([aug_rgb, aug_depth])
        combined = layers.RandomFlip('horizontal')(combined)
        aug_rgb = combined[..., :3]
        aug_depth = combined[..., 3:6]
    
    if aug_cfg.get('brightness', False):
        aug_rgb = layers.RandomBrightness(0.1)(aug_rgb)
    
    if aug_cfg.get('contrast', False):
        aug_rgb = layers.RandomContrast(0.1)(aug_rgb)
    
    # Preprocessing for pretrained models
    from tensorflow.keras.applications import efficientnet, mobilenet_v3
    rgb_processed = efficientnet.preprocess_input(aug_rgb)
    depth_processed = mobilenet_v3.preprocess_input(aug_depth)
    
    input_layers_dict = {'rgb_input': rgb_input, 'depth_input': depth_input, 'pc_input': pc_input}

    # RGB Encoder with skip connections using preprocessed data
    from tensorflow.keras.applications import EfficientNetB0
    rgb_encoder = EfficientNetB0(
        input_tensor=rgb_processed,
        weights='imagenet',
        include_top=False,
        input_shape=[*image_size, 3]
    )
    rgb_encoder._name = 'rgb_encoder'
    
    rgb_skip_names = [
        'block1a_project_bn',   # 128x128
        'block2b_add',          # 64x64
        'block3b_add',          # 32x32
        'block5c_add',          # 16x16
    ]
    rgb_skips = [rgb_encoder.get_layer(name).output for name in rgb_skip_names]
    rgb_bottleneck = rgb_encoder.output  # 8x8

    # Depth Encoder with skip connections using preprocessed data
    from tensorflow.keras.applications import MobileNetV3Small
    depth_encoder = MobileNetV3Small(
        input_tensor=depth_processed,
        weights=None,
        include_top=False,
        input_shape=[*image_size, 3],
        minimalistic=False
    )
    depth_encoder._name = 'depth_encoder'
    
    depth_skip_names = [
        'multiply',           # 128x128
        'multiply_1',         # 64x64
        'multiply_2',         # 32x32
        'multiply_3',         # 16x16
    ]
    depth_skips = [depth_encoder.get_layer(name).output for name in depth_skip_names]
    depth_bottleneck = depth_encoder.output  # 8x8

    # Point Cloud Encoder
    pc_x = layers.Conv1D(64, 1, activation='relu', name='pc_conv1')(pc_input)
    pc_x = layers.BatchNormalization(name='pc_bn1')(pc_x)
    pc_x = layers.Conv1D(128, 1, activation='relu', name='pc_conv2')(pc_x)
    pc_x = layers.BatchNormalization(name='pc_bn2')(pc_x)
    pc_features_vec = layers.GlobalMaxPooling1D(name='pc_pool')(pc_x)
    
    # Project point cloud to spatial feature map
    pc_bottleneck_size = rgb_bottleneck.shape[1] * rgb_bottleneck.shape[2]
    pc_features_spatial = layers.Dense(pc_bottleneck_size * 64, activation='relu', name='pc_dense')(pc_features_vec)
    pc_features_spatial = layers.Reshape((rgb_bottleneck.shape[1], rgb_bottleneck.shape[2], 64), name='pc_reshape')(pc_features_spatial)

    # Bottleneck fusion
    fused_bottleneck = layers.Concatenate(name='bottleneck_fusion')([rgb_bottleneck, depth_bottleneck, pc_features_spatial])
    
    # Bridge
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='bridge_conv1')(fused_bottleneck)
    x = layers.BatchNormalization(name='bridge_bn1')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='bridge_conv2')(x)
    x = layers.BatchNormalization(name='bridge_bn2')(x)
    
    # Decoder with skip connections
    def upsample_block(x, skip_rgb, skip_depth, filters, name):
        x = layers.UpSampling2D(2, interpolation='bilinear', name=f'{name}_upsample')(x)
        x = layers.Concatenate(name=f'{name}_concat')([x, skip_rgb, skip_depth])
        x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=f'{name}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=f'{name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)
        return x
    
    # Progressive upsampling with skip connections
    x = upsample_block(x, rgb_skips[3], depth_skips[3], 256, 'decode1')  # 16x16
    x = upsample_block(x, rgb_skips[2], depth_skips[2], 128, 'decode2')  # 32x32
    x = upsample_block(x, rgb_skips[1], depth_skips[1], 64, 'decode3')   # 64x64
    x = upsample_block(x, rgb_skips[0], depth_skips[0], 32, 'decode4')   # 128x128
    
    # Final upsampling to original resolution
    x = layers.UpSampling2D(2, interpolation='bilinear', name='final_upsample')(x)  # 256x256
    x = layers.Conv2D(16, 3, padding='same', activation='relu', name='final_conv1')(x)
    x = layers.BatchNormalization(name='final_bn1')(x)
    x = layers.Conv2D(16, 3, padding='same', activation='relu', name='final_conv2')(x)
    x = layers.BatchNormalization(name='final_bn2')(x)
    
    # Output layer
    outputs = layers.Conv2D(output_channels, 1, padding='same', name='final_output')(x)
    outputs = tf.cast(outputs, tf.float32, name='cast_to_float32')
    
    model = tf.keras.Model(inputs=input_layers_dict, outputs=outputs, name='UNet_Fused_Model')
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
        model = build_unet_with_aug_layers(
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
    
    # Stage 1: Pre-train new branches
    logger.info("\n" + "="*60 + "\n=== STAGE 1: Pre-training Depth & Point Cloud Branches ===\n" + "="*60)
    
    rgb_encoder = model.get_layer('rgb_encoder')
    rgb_encoder.trainable = False
    logger.info("RGB encoder frozen for Stage 1")
    
    stage1_callbacks = [
        TqdmProgressCallback(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir_abs / f'stage1_best_{timestamp}.h5'),
            monitor='val_stable_iou',
            save_best_only=True,
            verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_stable_iou',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
    ]
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=stage1_callbacks,
        verbose=0
    )
    
    # Stage 2: Fine-tune entire model
    logger.info("\n" + "="*60 + "\n=== STAGE 2: Fine-tuning all branches together ===\n" + "="*60)
    
    rgb_encoder.trainable = True
    optimizer_stage2 = Adam(learning_rate=1e-5, clipnorm=1.0)
    model.compile(optimizer=optimizer_stage2, loss=loss_function, metrics=metrics_list)
    logger.info("RGB encoder unfrozen for Stage 2")
    
    stage2_callbacks = [
        TqdmProgressCallback(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir_abs / f'stage2_best_{timestamp}.h5'),
            monitor='val_stable_iou',
            save_best_only=True,
            verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_stable_iou',
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_stable_iou',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )
    ]
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        initial_epoch=10,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=stage2_callbacks,
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