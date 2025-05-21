print("TRAIN.PY TOP LEVEL PRINT STATEMENT EXECUTING NOW") # CASCADE_DIAGNOSTIC_PRINT
import os
import sys
import yaml
import logging
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import traceback

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
        
def build_decoder_block(input_tensor, skip_feature_tensor, num_filters, kernel_size=(3,3), strides=(2,2), upsampling_type='conv_transpose', kernel_initializer='he_normal', block_name=''):
    if upsampling_type == 'conv_transpose':
        up = layers.Conv2DTranspose(num_filters, kernel_size, strides=strides, padding='same', name=f'{block_name}_transpose')(input_tensor)
    elif upsampling_type == 'upsampling_bilinear':
        up = layers.UpSampling2D(size=strides, interpolation='bilinear', name=f'{block_name}_upsample_bilinear')(input_tensor)
        # Adjust channels after upsampling if needed, to match num_filters (target for this block's convs)
        if up.shape[-1] != num_filters:
             up = layers.Conv2D(num_filters, (1,1), padding='same', activation='relu', name=f'{block_name}_upsample_channel_adjust')(up)
    else:
        raise ValueError(f"Unsupported upsampling_type: {upsampling_type}")

    processed_skip_tensor = skip_feature_tensor
    if up.shape[1] != skip_feature_tensor.shape[1] or up.shape[2] != skip_feature_tensor.shape[2]:
        processed_skip_tensor = layers.Resizing(
            height=up.shape[1],
            width=up.shape[2],
            interpolation='bilinear', # Bilinear is often preferred for feature maps
            name=f'{block_name}_skip_resize'
        )(skip_feature_tensor)
    
    # Concatenate the upsampled features with the (potentially resized) skip features
    # The original code also had a channel projection for skip features if channels didn't match 'up'.
    # This might be necessary if 'num_filters' for 'up' differs from 'processed_skip_tensor' channels.
    # For now, let's assume direct concatenation is intended after spatial alignment.
    # If channel mismatch occurs at concat, we might need to re-add skip projection or adjust num_filters.
    try:
        merge = layers.concatenate([up, processed_skip_tensor], axis=3, name=f'{block_name}_concat')
    except ValueError as e:
        logger.error(f"ERROR [{block_name}] concatenating shapes: up={up.shape}, processed_skip_tensor={processed_skip_tensor.shape} with axis=3. Error: {e}")
        # If error is about channels, e.g. up has C1 channels, processed_skip_tensor has C2, then C1+C2 is the new channel count.
        # The error might be if they are expected to be the same for some reason by a later layer, which is not typical for concat.
        # More likely the spatial dimensions were still not fully resolved, or a None dimension issue.
        raise e

    conv = layers.Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer, name=f'{block_name}_conv1')(merge)
    conv = layers.Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer, name=f'{block_name}_conv2')(conv)
    return conv


def unet_model(output_channels: int, image_size: tuple, model_config: dict, data_config: dict):
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
    fused_bottleneck = None # Will hold the features after image and PC fusion

    # --- Image Backbone (Encoder) --- #
    if backbone_name and backbone_name.lower() != 'none' and backbone_name.lower() != 'unet_scratch':
        # Create the backbone
        if backbone_input_channels != 3:
            logger.info(f"Backbone '{backbone_name}' typically expects 3 input channels, but got {backbone_input_channels} (RGB+Depth)."
                           f" Adding a Conv2D layer to project to 3 channels before the backbone.")
            processed_for_backbone = layers.Conv2D(3, (1, 1), padding='same', activation='relu', name='project_to_3_channels')(current_backbone_input)
        else:
            processed_for_backbone = current_backbone_input

        weights = None # Diagnostic: build without imagenet weights
        trainable_backbone = model_config.get('backbone_trainable', True)

        if backbone_name.lower() == 'efficientnetb0':
            base_model = EfficientNetB0(input_tensor=processed_for_backbone, include_top=False, weights=weights)
            # Skip connections from EfficientNetB0 (example layer names, may need adjustment)
            skip_names = model_config.get('skip_connection_layer_names', ['block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation']) # for B0
            # Use 'top_activation' or similar for final encoder output before bottleneck, or fetch from config
            encoder_output_layer_name = model_config.get('encoder_output_layer_name', 'top_activation') 
        elif backbone_name.lower() == 'resnet50v2':
            base_model = ResNet50V2(input_tensor=processed_for_backbone, include_top=False, weights=weights)
            # Skip connections for ResNet50V2
            skip_names = model_config.get('skip_connection_layer_names', ['conv1_conv', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']) # Adjust based on inspection
            encoder_output_layer_name = model_config.get('encoder_output_layer_name', 'post_relu') # Or the output of the last conv block
        # Add more backbones here as elif clauses
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Please add its configuration.")

        base_model.trainable = trainable_backbone
        encoder_output = base_model.get_layer(encoder_output_layer_name).output
        
        # Ensure encoder_output is set correctly from the backbone
        if not base_model.get_layer(encoder_output_layer_name):
            logger.error(f"Critical: Encoder output layer '{encoder_output_layer_name}' not found in backbone {backbone_name}. Available layers: {[l.name for l in base_model.layers]}")
            raise ValueError(f"Encoder output layer '{encoder_output_layer_name}' not found in backbone {backbone_name}.")
        logger.info(f"Image backbone '{backbone_name}' output (encoder_output) shape: {encoder_output.shape}")

        # Collect actual skip layer OUTPUT TENSORS from the base_model, based on names in config
        backbone_skip_outputs = {} # Maps NAME to TENSOR for actual backbone layers
        _configured_skip_names = model_config.get('skip_connection_layer_names', [])
        for name in _configured_skip_names:
            if name == 'MODEL_INPUT': # 'MODEL_INPUT' is not from the backbone model itself
                continue
            try:
                layer_output = base_model.get_layer(name).output
                backbone_skip_outputs[name] = layer_output
                logger.info(f"Successfully fetched skip layer tensor for '{name}' (shape: {layer_output.shape}) from backbone.")
            except ValueError:
                logger.warning(f"Skip connection layer '{name}' (specified in config) not found in backbone '{backbone_name}'. Will be problematic if used in decoder.")
        logger.info(f"Collected {len(backbone_skip_outputs)} skip connection TENSORS from backbone.")

    elif backbone_name and backbone_name.lower() == 'unet_scratch':
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
        # skip_connection_layers.reverse() # From deep (conv3) to shallow (conv1)

    else: # No backbone, implies a very simple direct path (not typical for U-Net)
        logger.warning("No backbone specified, and not 'unet_scratch'. Using 'current_backbone_input' directly as encoder_output. This is unusual for a U-Net.")
        encoder_output = current_backbone_input # This would be RGB or RGB+D

    current_features = encoder_output # This is the bottleneck, e.g., 8x8
    
    # --- Decoder --- #
    # These names from config will drive the decoder loop
    decoder_skip_names_from_config = model_config.get('skip_connection_layer_names', [])
    decoder_filters_from_config = model_config.get('decoder_filters', [])
    upsampling_type = model_config.get('upsampling_type', 'conv_transpose') # Default

    num_decoder_blocks = min(len(decoder_skip_names_from_config), len(decoder_filters_from_config))
    
    logger.info(f"Starting decoder construction with {num_decoder_blocks} blocks.")
    logger.info(f"  Decoder skip names (from config): {decoder_skip_names_from_config[:num_decoder_blocks]}")
    logger.info(f"  Decoder filters (from config): {decoder_filters_from_config[:num_decoder_blocks]}")

    for i, (skip_feat_name, num_out_filters) in enumerate(zip(
            decoder_skip_names_from_config[:num_decoder_blocks],
            decoder_filters_from_config[:num_decoder_blocks]
        )):
        
        skip_feature_tensor = None # This will hold the actual tensor for the skip connection
        if skip_feat_name == 'MODEL_INPUT':
            skip_feature_tensor = current_backbone_input # Use the original model input (e.g., 256x256)
            logger.info(f"Using 'MODEL_INPUT' (shape: {skip_feature_tensor.shape}) as skip connection for decoder block {i+1}.")
        elif skip_feat_name in backbone_skip_outputs:
            skip_feature_tensor = backbone_skip_outputs[skip_feat_name]
            # logger.info(f"Using backbone skip '{skip_feat_name}' (shape: {skip_feature_tensor.shape}) for decoder block {i+1}.")
        else:
            logger.warning(f"Skip connection name '{skip_feat_name}' (from config) not found in collected backbone outputs or is not 'MODEL_INPUT'. Stopping decoder construction.")
            break # Critical: stop decoder if a required skip is missing
            
        logger.info(f"Decoder block {i+1}: current_features_in={current_features.shape}, skip_feature_to_use='{skip_feat_name}' (tensor_shape={skip_feature_tensor.shape}), num_out_filters={num_out_filters}")
        
        current_features = build_decoder_block(
            input_tensor=current_features,
            skip_feature_tensor=skip_feature_tensor,
            num_filters=num_out_filters,
            upsampling_type=upsampling_type,
            kernel_initializer=kernel_initializer,
            block_name=f'decoder_block_{i+1}'
        )
        logger.info(f"Output of decoder_block_{i+1}: {current_features.shape}")

    conv_final_stage = current_features 
    logger.info(f"DEBUG: Shape of tensor BEFORE final output conv layer: {conv_final_stage.shape}")

    # Final output layer
    output_activation = model_config.get('output_activation', 'sigmoid' if output_channels == 1 else 'softmax')
    outputs = layers.Conv2D(output_channels, (1, 1), padding="same", activation=output_activation, name='final_output_conv')(conv_final_stage)
    logger.info(f"DEBUG: Shape of model OUTPUTS after final conv layer: {outputs.shape}")
    
    # Build the Keras model
    model_inputs = [rgb_input]
    if data_config.get('use_depth_map', False):
        model_inputs.append(depth_input)
    if data_config.get('use_point_cloud', False):
        model_inputs.append(pc_input)
    
    if not model_inputs:
        raise ValueError("No Keras Input layers were collected for the model construction.")
    
    model = tf.keras.Model(inputs=model_inputs, outputs=outputs)
    logger.critical(f"CRITICAL_FINAL_MODEL_OUTPUT_SHAPE_CONFIRMATION: {model.output_shape}")

    # Save model summary to a file
    summary_file_path = "model_summary.txt" # Will be saved in the CWD of train.py
    try:
        with open(summary_file_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        logger.info(f"Model summary saved to {summary_file_path}")
    except Exception as e:
        logger.error(f"Could not save model summary to {summary_file_path}: {e}")

    # Log model summary if verbosity is high enough
    if logger.getEffectiveLevel() <= logging.DEBUG:
        model.summary(print_fn=logger.info)

    logger.info(f"U-Net model built with final activation: {output_activation}.")
    return model

def main():
    project_root = _get_project_root()
    config_file_path = Path(SEGMENTATION_CONFIG_PATH) 
    config = load_config(config_file_path)

    logger.info("Starting segmentation model training...")

    train_dataset, val_dataset, test_dataset, num_train_samples = load_segmentation_data(config) # load_segmentation_data returns three datasets

    if train_dataset is None:
        logger.error("Failed to load training dataset. Exiting.")
        return

    # --- Data Loading Test ---
    data_cfg_for_test = config.get('data', {})
    _test_data_loading(train_dataset, data_cfg_for_test, num_batches_to_test=2) 
    # --- End Data Loading Test ---

    logger.info("--- Checking train_dataset structure just before model.compile ---")
    # Take one batch to inspect its structure
    for pre_compile_batch_data in train_dataset.take(1):
        pre_compile_inputs, pre_compile_masks = pre_compile_batch_data
        if isinstance(pre_compile_inputs, dict):
            logger.info("Pre-compile inputs is a dictionary. Keys and shapes:")
            for k, v_tensor in pre_compile_inputs.items():
                if hasattr(v_tensor, 'shape'):
                    logger.info(f"  inputs_dict['{k}']: shape={v_tensor.shape}, dtype={v_tensor.dtype}")
                else:
                    logger.info(f"  inputs_dict['{k}']: (not a tensor, type: {type(v_tensor)}) {v_tensor}")
        elif hasattr(pre_compile_inputs, 'shape'): # If it's a single tensor
            logger.info(f"Pre-compile inputs (single tensor): shape={pre_compile_inputs.shape}, dtype={pre_compile_inputs.dtype}")
        else: # Other unexpected structure
            logger.info(f"Pre-compile inputs (unexpected structure): {type(pre_compile_inputs)}")
        
        if hasattr(pre_compile_masks, 'shape'):
            logger.info(f"Pre-compile masks: shape={pre_compile_masks.shape}, dtype={pre_compile_masks.dtype}")
        else:
            logger.info(f"Pre-compile masks (unexpected structure): {type(pre_compile_masks)}")
    logger.info("--- End of pre-compile dataset check ---")

    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})

    # Ensure weights_path is None if load_weights is False, to prevent Keras from implicitly trying to load.
    if not model_cfg.get('load_weights', False): # Default to False if key is missing
        logger.info("model_cfg['load_weights'] is False. Ensuring 'weights_path' is not used by setting it to None.")
        model_cfg['weights_path'] = None

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
    logger.info("U-Net model built successfully.")

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
    if loss_fn_name == 'binary_crossentropy': 
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
        elif m_name_lower == 'mean_iou': # MODIFIED: Added underscore
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

    model.fit(
        train_dataset,
        epochs=training_cfg.get('epochs', 10),
        validation_data=val_dataset,
        callbacks=callbacks_list,
        steps_per_epoch=training_cfg.get('debug_steps_per_epoch'),
        validation_steps=training_cfg.get('debug_validation_steps')
    )

    logger.info("Training finished.")

    # Save the final model
    final_model_filename = paths_cfg.get('final_model_filename', f'unet_segmentation_final_{timestamp}.h5')
    final_model_path_obj = model_dir_abs / final_model_filename
    model.save(str(final_model_path_obj)) 
    logger.info(f"Final trained model saved to: {str(final_model_path_obj)}")

    # Evaluate on test set if available
    evaluate_on_test_set = config.get('evaluate_on_test_set', True)
    if evaluate_on_test_set and test_dataset:
        logger.info("Evaluating model on the test set after training...") # Indented and slightly changed comment
        # The 'model' object should have the best weights if EarlyStopping with restore_best_weights=True was used.
        test_results = model.evaluate(test_dataset, verbose=1)
        logger.info("Test Set Evaluation Results:")
        if isinstance(test_results, list): # If model has multiple metrics
            for metric_name, value in zip(model.metrics_names, test_results):
                logger.info(f"  {metric_name}: {value}")
        else: # Single loss value
            logger.info(f"  loss: {test_results}")

    if config.get('export_model_to_tflite', False):
        # Export model to TFLite
        logger.info("Exporting model to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_model_path = model_dir_abs / f'unet_segmentation_{timestamp}.tflite'
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        logger.info(f"TFLite model saved to: {tflite_model_path}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("Exception in main execution:", exc_info=True)
        # Ensure the traceback is printed to stderr as well for visibility
        traceback.print_exc() 
        sys.exit(1) # Exit with a non-zero code to indicate failure