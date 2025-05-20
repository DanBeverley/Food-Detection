import os
import sys
import yaml
import logging
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Add _get_project_root function definition
def _get_project_root() -> Path:
    """Assumes this script is in Food-Detection/models/segmentation/"""
    return Path(__file__).resolve().parent.parent.parent

# Assuming data.py is in the same directory or accessible in PYTHONPATH
from data import load_segmentation_data # Use relative import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the specific config file for segmentation training
SEGMENTATION_CONFIG_PATH = os.path.join(_get_project_root(), "models", "segmentation", "config.yaml")

from tensorflow.keras.callbacks import Callback
# Import Keras applications for backbones
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2 # Add more as needed
from tensorflow.keras import layers # Explicitly import layers for clarity

def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config

def _test_data_loading(train_dataset: tf.data.Dataset, data_config: dict, num_batches_to_test: int = 2):
    logger.info(f"--- Starting Data Loading Test for {num_batches_to_test} batches ---")
    if train_dataset is None:
        logger.error("Training dataset is None. Skipping data loading test.")
        return

    use_depth = data_config.get('use_depth_map', False)
    use_pc = data_config.get('use_point_cloud', False)

    for i, batch_data in enumerate(train_dataset.take(num_batches_to_test)):
        logger.info(f"--- Batch {i+1} ---")
        inputs_dict, mask_tensor = batch_data

        if not isinstance(inputs_dict, dict):
            logger.error(f"Expected inputs_dict to be a dict, but got {type(inputs_dict)}. Batch structure might be incorrect.")
            continue

        # RGB Input
        if 'rgb_input' in inputs_dict:
            rgb = inputs_dict['rgb_input']
            logger.info(f"  rgb_input: Shape={rgb.shape}, Dtype={rgb.dtype}, Min={tf.reduce_min(rgb).numpy()}, Max={tf.reduce_max(rgb).numpy()}")
        else:
            logger.warning("  rgb_input not found in batch.")

        # Depth Input (if enabled)
        if use_depth:
            if 'depth_input' in inputs_dict:
                depth = inputs_dict['depth_input']
                logger.info(f"  depth_input: Shape={depth.shape}, Dtype={depth.dtype}, Min={tf.reduce_min(depth).numpy()}, Max={tf.reduce_max(depth).numpy()}")
            else:
                logger.warning("  depth_input is expected (use_depth_map=True) but not found in batch.")
        
        # Point Cloud Input (if enabled)
        if use_pc:
            if 'pc_input' in inputs_dict:
                pc = inputs_dict['pc_input']
                logger.info(f"  pc_input: Shape={pc.shape}, Dtype={pc.dtype}, Min={tf.reduce_min(pc).numpy()}, Max={tf.reduce_max(pc).numpy()}")
            else:
                logger.warning("  pc_input is expected (use_point_cloud=True) but not found in batch.")

        # Mask Tensor
        logger.info(f"  mask_tensor: Shape={mask_tensor.shape}, Dtype={mask_tensor.dtype}, Min={tf.reduce_min(mask_tensor).numpy()}, Max={tf.reduce_max(mask_tensor).numpy()}")
        unique_values, _ = tf.unique(tf.reshape(mask_tensor, [-1]))
        logger.info(f"  mask_tensor: Unique values={unique_values.numpy()}")
        
    logger.info("--- Data Loading Test Finished ---")


def unet_model(output_channels: int, image_size: tuple, model_config: dict, data_config: dict) -> tf.keras.Model:
    """Builds a U-Net model with a dynamic backbone and potentially multi-modal input.
    Args:
        output_channels: Number of output channels (e.g., 1 for binary segmentation).
        image_size: Tuple (height, width) for the input image.
        model_config: Dictionary containing model-specific configurations (e.g., backbone, dropout).
        data_config: Dictionary containing data-specific configurations (for input modalities).
    Returns:
        A Keras U-Net model.
    """
    input_layers_dict = {}
    encoder_inputs_list = []
    backbone_input_channels = 3 # Start with RGB

    # RGB Input (Always present)
    rgb_input = layers.Input(shape=[image_size[0], image_size[1], 3], name='rgb_input')
    input_layers_dict['rgb_input'] = rgb_input
    current_backbone_input = rgb_input
    encoder_inputs_list.append(rgb_input)

    # Depth Input (if enabled)
    if data_config.get('use_depth_map', False):
        depth_input = layers.Input(shape=[image_size[0], image_size[1], 1], name='depth_input')
        input_layers_dict['depth_input'] = depth_input
        current_backbone_input = layers.concatenate([current_backbone_input, depth_input])
        encoder_inputs_list.append(depth_input)
        backbone_input_channels += 1
        logger.info("Depth input enabled and concatenated with RGB for backbone input.")

    # Point Cloud Input (processed separately)
    if data_config.get('use_point_cloud', False):
        pc_cfg = data_config.get('point_cloud', {})
        num_points = pc_cfg.get('num_points', data_config.get('modalities_preprocessing', {}).get('point_cloud',{}).get('num_points', 4096))
        pc_input = layers.Input(shape=[num_points, 3], name='pc_input')
        input_layers_dict['pc_input'] = pc_input
        encoder_inputs_list.append(pc_input)
        logger.info(f"Point cloud input enabled with shape: ({num_points}, 3). Will have a separate processing branch.")

    # Ensure Keras model gets a dictionary of named Input layers if multiple, or single tensor if only RGB
    actual_model_inputs = input_layers_dict if len(encoder_inputs_list) > 1 else encoder_inputs_list[0]
    if len(input_layers_dict) == 0: # Should not happen if rgb_input is always there
        raise ValueError("No input layers were defined for the model.")
    
    # Override input_shape for backbone if it's different from default (e.g. due to depth concatenation)
    # Backbones expect a fixed channel number. If we have RGB+D, we can't directly use a pretrained backbone with 3 channels.
    # Option 1: Modify backbone's first layer (complex, loses some pretraining benefits for initial layer)
    # Option 2: Use a separate Conv2D to map RGB+D to 3 channels, then feed to standard backbone.
    # Option 3: If backbone is 'None' or a custom simple U-Net, build it to handle `backbone_input_channels`.

    backbone_name = model_config.get('backbone', None)
    dropout_rate = model_config.get('dropout', 0.5)
    kernel_initializer = model_config.get('kernel_initializer', 'he_normal')

    skip_connection_layers = []
    encoder_output = None

    if backbone_name and backbone_name.lower() != 'none' and backbone_name.lower() != 'unet_scratch':
        # Create the backbone
        if backbone_input_channels != 3:
            logger.warning(f"Backbone '{backbone_name}' typically expects 3 input channels, but got {backbone_input_channels} (RGB+Depth)."
                           f" Adding a Conv2D layer to project to 3 channels before the backbone.")
            processed_for_backbone = layers.Conv2D(3, (1, 1), padding='same', activation='relu', name='project_to_3_channels')(current_backbone_input)
        else:
            processed_for_backbone = current_backbone_input

        weights = model_config.get('backbone_weights', 'imagenet')
        trainable_backbone = model_config.get('backbone_trainable', True)

        if backbone_name == 'EfficientNetB0':
            base_model = EfficientNetB0(input_tensor=processed_for_backbone, include_top=False, weights=weights)
            # Skip connections from EfficientNetB0 (example layer names, may need adjustment)
            skip_names = ['block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation'] # for B0
            # Use 'top_activation' or similar for final encoder output before bottleneck
            encoder_output_layer_name = 'top_activation' 
        elif backbone_name == 'ResNet50V2':
            base_model = ResNet50V2(input_tensor=processed_for_backbone, include_top=False, weights=weights)
            # Skip connections for ResNet50V2
            skip_names = ['conv1_conv', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out'] # Adjust based on inspection
            encoder_output_layer_name = 'post_relu' # Or the output of the last conv block
        # Add more backbones here as elif clauses
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Please add its configuration.")

        base_model.trainable = trainable_backbone
        encoder_output = base_model.get_layer(encoder_output_layer_name).output
        
        for name in skip_names:
            skip_connection_layers.append(base_model.get_layer(name).output)
        skip_connection_layers.reverse() # Decoder uses them from deep to shallow
        logger.info(f"Using {backbone_name} as encoder. Trainable: {trainable_backbone}. Skip layers: {skip_names}")
    else:
        logger.info("Building U-Net encoder from scratch (no pre-trained backbone or 'unet_scratch' specified).")
        # == U-Net Encoder (Downsampling path) - takes 'current_backbone_input' (RGB or RGB+Depth) ==
        # Block 1
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(current_backbone_input)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        skip_connection_layers.append(conv1)

        # Block 2
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        skip_connection_layers.append(conv2)

        # Block 3
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        skip_connection_layers.append(conv3)

        # Block 4 (Bottleneck for scratch U-Net)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4)
        encoder_output = layers.Dropout(dropout_rate)(conv4)
        # No skip from bottleneck itself for standard U-Net, last skip is conv3 (or conv4 if pool3 is used as bottleneck input for decoder)
        # skip_connection_layers.append(conv4) # This would be the bottleneck itself if used as a skip
        skip_connection_layers.reverse() # From deep (conv3) to shallow (conv1)

    # == Point Cloud Processing Branch (if enabled) ==
    pc_features_for_fusion = None
    if data_config.get('use_point_cloud', False) and 'pc_input' in input_layers_dict:
        pc_input_tensor = input_layers_dict['pc_input']
        # Mini-PointNet style encoder
        pc_conv1 = layers.Conv1D(64, kernel_size=1, padding='same', activation='relu', kernel_initializer=kernel_initializer)(pc_input_tensor)
        pc_bn1 = layers.BatchNormalization()(pc_conv1)
        pc_conv2 = layers.Conv1D(128, kernel_size=1, padding='same', activation='relu', kernel_initializer=kernel_initializer)(pc_bn1)
        pc_bn2 = layers.BatchNormalization()(pc_conv2)
        pc_conv3 = layers.Conv1D(256, kernel_size=1, padding='same', activation='relu', kernel_initializer=kernel_initializer)(pc_bn2)
        pc_bn3 = layers.BatchNormalization()(pc_conv3)
        pc_global_features = layers.GlobalMaxPooling1D()(pc_bn3) # Shape: (batch_size, 256)
        pc_features_for_fusion = pc_global_features
        logger.info(f"Point cloud branch processed. Global features shape: {pc_features_for_fusion.shape}")

    # == Fusion of U-Net Bottleneck (encoder_output) with Point Cloud Features (if enabled) ==
    fused_bottleneck = encoder_output
    if pc_features_for_fusion is not None:
        bottleneck_shape = tf.shape(encoder_output) # Dynamic shape
        target_height = bottleneck_shape[1]
        target_width = bottleneck_shape[2]

        pc_feat_expanded = layers.Reshape((1, 1, pc_features_for_fusion.shape[-1]))(pc_features_for_fusion)
        pc_feat_tiled = layers.Lambda(lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], tf.shape(x[1])[2], 1]), name='tile_pc_features')([pc_feat_expanded, encoder_output])

        fused_bottleneck = layers.concatenate([encoder_output, pc_feat_tiled], axis=-1)
        logger.info(f"U-Net bottleneck fused with tiled point cloud features. New bottleneck shape: {fused_bottleneck.shape}")

    # == U-Net Decoder (Upsampling path) ==
    current_decoder_features = fused_bottleneck
    # Number of upsampling blocks should match number of skip connections from encoder - 1 (if bottleneck is not a skip)
    # Or, iterate through skip_connection_layers
    # For scratch U-Net: conv3, conv2, conv1 as skips. Decoder stages: 3
    # For backbone: len(skip_connection_layers) stages.
    
    num_decoder_blocks = len(skip_connection_layers)
    decoder_filters = [256, 128, 64, 32] # Example, adjust based on backbone or desired complexity
    if backbone_name and backbone_name.lower() != 'none' and backbone_name.lower() != 'unet_scratch':
        # For backbones, skip layers are already reversed (deepest first)
        # Bottleneck is 'encoder_output'. Decoder starts from there.
        pass # Filters might need to match skip connection channels
    else: # Scratch U-Net
        decoder_filters = [256, 128, 64] # For the 3 upsampling stages if bottleneck is 512

    for i in range(num_decoder_blocks):
        skip_feat = skip_connection_layers[i]
        num_filters = decoder_filters[i] if i < len(decoder_filters) else decoder_filters[-1]

        up = layers.Conv2DTranspose(num_filters, (2,2), strides=(2,2), padding='same')(current_decoder_features)
        # Ensure spatial dimensions of skip_feat and up are compatible for concatenation
        # This might require cropping skip_feat if padding='valid' was used in encoder, or ensuring 'same' padding everywhere.
        # Or, resize 'up' to match 'skip_feat' if dynamic shapes cause slight mismatches.
        # up = layers.Resizing(tf.shape(skip_feat)[1], tf.shape(skip_feat)[2])(up) # Example if needed
        
        # Check for channel mismatch before concatenation if skip_feat comes from a varied backbone
        # Ensure 'up' has compatible number of channels or project skip_feat if necessary.
        if up.shape[-1] != skip_feat.shape[-1]:
             logger.warning(f"Channel mismatch for skip connection {i}. Up-sampled: {up.shape[-1]}, Skip: {skip_feat.shape[-1]}. Adjusting skip connection.")
             # This simple projection might not be ideal. Better to ensure decoder filters match skip connection channels.
             # Or, ensure skip layers from backbone are chosen such that their channel counts align with typical U-Net decoder stages.
             projected_skip_feat = layers.Conv2D(up.shape[-1], (1,1), padding='same', activation='relu')(skip_feat)
             merge = layers.concatenate([projected_skip_feat, up], axis=3)
        else:
             merge = layers.concatenate([skip_feat, up], axis=3)

        conv = layers.Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge)
        conv = layers.Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv)
        current_decoder_features = conv

    # Output layer
    final_activation = model_config.get('activation', 'sigmoid') 
    outputs = layers.Conv2D(output_channels, 1, activation=final_activation)(current_decoder_features)

    # Determine the final list of input tensors for the tf.keras.Model
    # 'encoder_inputs_list' contains all Input layers (rgb, depth, pc)
    model_inputs_final = []
    if 'rgb_input' in input_layers_dict: model_inputs_final.append(input_layers_dict['rgb_input'])
    if 'depth_input' in input_layers_dict: model_inputs_final.append(input_layers_dict['depth_input'])
    if 'pc_input' in input_layers_dict: model_inputs_final.append(input_layers_dict['pc_input'])
    
    if not model_inputs_final:
        raise ValueError("No Keras Input layers were collected for the model construction.")
    
    model = tf.keras.Model(inputs=model_inputs_final, outputs=outputs)
    logger.info(f"U-Net model built with final activation: {final_activation}.")
    return model

def main():
    project_root = _get_project_root()
    config_file_path = Path(SEGMENTATION_CONFIG_PATH) 
    config = load_config(config_file_path)

    logger.info("Starting segmentation model training...")

    train_dataset, val_dataset, test_dataset = load_segmentation_data(config) # load_segmentation_data returns three datasets

    if train_dataset is None:
        logger.error("Failed to load training dataset. Exiting.")
        return

    # --- Data Loading Test ---
    data_cfg_for_test = config.get('data', {})
    _test_data_loading(train_dataset, data_cfg_for_test, num_batches_to_test=2) 
    # --- End Data Loading Test ---

    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    optimizer_cfg = config.get('optimizer', {}) # Corrected: Get from root config
    loss_cfg = config.get('loss', {})           # Corrected: Get from root config
    metrics_cfg = training_cfg.get('metrics', ['accuracy'])
    paths_cfg = config.get('paths', {})

    # Define model
    output_channels = model_cfg.get('num_classes', 1) # Use num_classes from model_cfg, affects output layer
    image_h, image_w = data_cfg.get('image_size', [256, 256])
    model = unet_model(output_channels=output_channels, 
                       image_size=(image_h, image_w), 
                       model_config=model_cfg, 
                       data_config=data_cfg) # Pass data_config

    # Compile model
    learning_rate = optimizer_cfg.get('learning_rate', 1e-4)
    optimizer_name = optimizer_cfg.get('name', 'Adam').lower()

    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Add other optimizers if needed, e.g., SGD
    # elif optimizer_name == 'sgd':
    #     momentum = optimizer_cfg.get('momentum', 0.9)
    #     optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    loss_fn_name = loss_cfg.get('name', 'BinaryCrossentropy').lower()
    if loss_fn_name == 'binarycrossentropy':
        # Determine from_logits based on whether the model's final layer has a sigmoid
        # This should align with the 'activation' specified in the model_cfg for the unet_model
        model_final_activation = model_cfg.get('activation', 'sigmoid') # Default to 'sigmoid' if not specified
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=(model_final_activation != 'sigmoid'))
    # Add other loss functions if needed, e.g., Dice Loss
    # elif loss_fn_name == 'dice_loss':
    #     # Placeholder for a custom Dice Loss implementation or from a library
    #     pass 
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn_name}")

    # Metrics
    metrics_list = []
    for m_name in metrics_cfg:
        m_name_lower = m_name.lower()
        if m_name_lower == 'accuracy':
            metrics_list.append('accuracy')
        elif m_name_lower == 'meaniou':
            num_classes_metric = model_cfg.get('num_classes_for_metric', 2) # Typically 2 for binary (bg/fg)
            metrics_list.append(tf.keras.metrics.MeanIoU(num_classes=num_classes_metric, name='mean_iou'))
        # Add other metrics as needed
        else:
            logger.warning(f"Unsupported metric '{m_name}' specified in config. Skipping.")

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=metrics_list)

    model.summary(print_fn=logger.info)

    # Training parameters from config
    epochs = training_cfg.get('epochs', 50)

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_dir_rel = paths_cfg.get('model_save_dir', 'trained_models/segmentation')
    log_dir_rel = paths_cfg.get('log_dir', 'logs/segmentation')
    
    model_dir_abs = project_root / model_save_dir_rel
    model_dir_abs.mkdir(parents=True, exist_ok=True)
    log_dir_abs = project_root / log_dir_rel / timestamp # Add timestamp to log_dir
    log_dir_abs.mkdir(parents=True, exist_ok=True)

    callbacks_list = []
    callbacks_config = training_cfg.get('callbacks', {})

    if callbacks_config.get('model_checkpoint', {}).get('enabled', True):
        ckpt_config = callbacks_config['model_checkpoint']
        checkpoint_filename_template = ckpt_config.get('filename_template', 'unet_epoch-{epoch:02d}_val_iou-{val_mean_iou:.4f}.h5')
        checkpoint_filepath_obj = model_dir_abs / checkpoint_filename_template
        callbacks_list.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_filepath_obj),
            save_weights_only=ckpt_config.get('save_weights_only', False),
            monitor=ckpt_config.get('monitor', 'val_mean_iou'),
            mode=ckpt_config.get('mode', 'max'),
            save_best_only=ckpt_config.get('save_best_only', True),
            verbose=1
        ))
        logger.info(f"ModelCheckpoint enabled. Saving best to: {model_dir_abs}")

    if callbacks_config.get('early_stopping', {}).get('enabled', True):
        es_config = callbacks_config['early_stopping']
        callbacks_list.append(tf.keras.callbacks.EarlyStopping(
            monitor=es_config.get('monitor', 'val_loss'),
            patience=es_config.get('patience', 10),
            mode=es_config.get('mode', 'min'),
            verbose=1,
            restore_best_weights=es_config.get('restore_best_weights', True)
        ))
        logger.info("EarlyStopping enabled.")

    if callbacks_config.get('tensorboard', {}).get('enabled', True):
        tb_config = callbacks_config['tensorboard']
        callbacks_list.append(tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir_abs),
            histogram_freq=tb_config.get('histogram_freq', 1),
            write_graph=tb_config.get('write_graph', True),
            update_freq=tb_config.get('update_freq', 'epoch')
        ))
        logger.info(f"TensorBoard logging enabled to {log_dir_abs}.")

    # ReduceLROnPlateau (optional, add if in config)
    if callbacks_config.get('reduce_lr_on_plateau', {}).get('enabled', False):
        lr_config = callbacks_config['reduce_lr_on_plateau']
        callbacks_list.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor=lr_config.get('monitor', 'val_loss'),
            factor=lr_config.get('factor', 0.1),
            patience=lr_config.get('patience', 5),
            min_lr=lr_config.get('min_lr', 1e-6),
            mode=lr_config.get('mode', 'min'),
            verbose=1
        ))
        logger.info("ReduceLROnPlateau enabled.")

    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Training data examples: {train_dataset.unbatch().take(1) if hasattr(train_dataset, 'unbatch') else 'N/A'}") # Log one example
    logger.info(f"Validation data examples: {val_dataset.unbatch().take(1) if val_dataset and hasattr(val_dataset, 'unbatch') else 'None'}")

    fit_kwargs = {
        'epochs': epochs,
        'callbacks': callbacks_list,
        'verbose': 1,
    }

    if val_dataset:
        fit_kwargs['validation_data'] = val_dataset
    else:
        logger.info("No validation dataset provided or it's empty. Skipping validation during fit.")
        # No need to explicitly pass validation_data=None if not present in kwargs

    history = model.fit(
        train_dataset,
        **fit_kwargs
    )

    logger.info("Training finished.")

    # Save the final model
    final_model_filename = paths_cfg.get('final_model_filename', f'unet_segmentation_final_{timestamp}.h5')
    final_model_path_obj = model_dir_abs / final_model_filename
    model.save(str(final_model_path_obj)) 
    logger.info(f"Final trained model saved to: {str(final_model_path_obj)}")

    # Evaluate on test set if available
    if test_dataset:
        logger.info("Evaluating model on test set...")
        # Load the best model if ModelCheckpoint was used and saved best
        best_model_path = None
        if callbacks_config.get('model_checkpoint', {}).get('enabled', True) and callbacks_config['model_checkpoint'].get('save_best_only', True):
            # Construct the path to the best model if a template was used; this might be tricky if exact filename changed
            # For simplicity, we're assuming the final model after training (with restore_best_weights from EarlyStopping) is good
            # Or, one could list files in model_dir_abs and pick the one with 'best' if naming convention is consistent.
            # Here, we'll evaluate the 'model' object, which should have the best weights if EarlyStopping restored them.
            pass # Using current 'model' which should have best weights if restore_best_weights=True

        test_results = model.evaluate(test_dataset, verbose=1)
        logger.info("Test Set Evaluation Results:")
        for metric_name, value in zip(model.metrics_names, test_results):
            logger.info(f"  {metric_name}: {value:.4f}")
    else:
        logger.info("No test dataset provided for evaluation.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in the segmentation training script: {e}", exc_info=True)