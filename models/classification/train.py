import sys
import yaml
import argparse
import logging
from pathlib import Path
import os
import tensorflow as tf

from data import load_classification_data
from model import build_classification_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def initialize_strategy() -> tf.distribute.Strategy:
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        logger.info(f"Found {len(gpus)} GPUs. Using MirroredStrategy.")
        return tf.distribute.MirroredStrategy()
    elif len(gpus) == 1:
        logger.info("Found 1 GPU. Using default strategy.")
        return tf.distribute.get_strategy()
    
    logger.warning("No GPU found. Using CPU strategy.")
    return tf.distribute.get_strategy()


def main(args):
    MODEL_CHECKPOINT_PATH = args.model_checkpoint_path
    
    logger.info(f"Loading configuration from: {args.config}")
    config = yaml.safe_load(Path(args.config).read_text())
    strategy = initialize_strategy()
    
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    optimizer_cfg = config.get('optimizer', {})
    training_cfg = config.get('training', {})
    
    if args.base_data_dir:
        data_cfg['base_data_dir'] = args.base_data_dir
    config['data'] = data_cfg
    
    train_ds, val_ds, num_train, num_val, class_names, num_classes = load_classification_data(config)
    if not train_ds:
        logger.critical("Data loading failed. Cannot proceed.")
        sys.exit(1)
    logger.info(f"Data loaded: {num_train} train, {num_val} val samples.")

    # --- MODEL CREATION AND LOADING (Happens First) ---
    with strategy.scope():
        model = build_classification_model(num_classes, config)
        
        # Load weights IMMEDIATELY after creation if resuming
        if args.resume_training:
            if os.path.exists(MODEL_CHECKPOINT_PATH):
                logger.info("======== RESUMING TRAINING ========")
                logger.info(f"Loading weights from: {MODEL_CHECKPOINT_PATH}")
                # This must be the first operation on the model after building
                model.load_weights(MODEL_CHECKPOINT_PATH)
                logger.info("Weights loaded successfully.")
            else:
                logger.warning(f"Resume flag was set, but no model found at {MODEL_CHECKPOINT_PATH}. Will start fresh.")
                args.resume_training = False # Treat as a new run
        else:
            logger.info("======== STARTING NEW TRAINING ========")


    # If not resuming, run Stage 1
    if not args.resume_training:
        logger.info("--- Stage 1: Training classification head only ---")
        architecture = model_cfg.get('architecture', 'EfficientNetV2B0')
        base_model_name = {
            'EfficientNetV2B0': 'efficientnetv2-b0',
            'MobileNetV2': 'mobilenetv2_1.00_224',
            'MobileNetV3Small': 'mobilenetv3_small'
        }.get(architecture)
        base_model = model.get_layer(name=base_model_name)
        base_model.trainable = False
        
        optimizer_stage1 = tf.keras.optimizers.Adam(
            learning_rate=optimizer_cfg.get('stage1_learning_rate', 1e-3),
            clipnorm=optimizer_cfg.get('clipnorm', 1.0)
        )
        model.compile(
            optimizer=optimizer_stage1,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=config['loss']['params']['label_smoothing']),
            metrics=['accuracy'],
            jit_compile=False
        )
        stage1_epochs = training_cfg.get('stage1_epochs', 5)
        if stage1_epochs > 0:
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=stage1_epochs,
                steps_per_epoch=num_train // data_cfg['batch_size'],
                validation_steps=num_val // data_cfg['batch_size']
            )
    else:
        logger.info("--- Resuming Training: Skipping Stage 1 ---")
    
    # Always run Stage 2
    logger.info("--- Stage 2: Fine-tuning the model ---")
    steps_per_epoch = num_train // data_cfg['batch_size']
    total_decay_steps = steps_per_epoch * training_cfg.get('stage2_epochs', 50)
    cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
        learning_rate=optimizer_cfg.get('stage2_learning_rate', 1e-4),
        decay_steps=total_decay_steps,
        alpha=0.0
    )
    architecture = model_cfg.get('architecture', 'EfficientNetV2B0')
    base_model_name = {
        'EfficientNetV2B0': 'efficientnetv2-b0',
        'MobileNetV2': 'mobilenetv2_1.00_224',
        'MobileNetV3Small': 'mobilenetv3_small'
    }.get(architecture)
    base_model = model.get_layer(name=base_model_name)
    base_model.trainable = True
    num_fine_tune_layers = model_cfg.get('stage2_trainable_layers', 0)
    if num_fine_tune_layers > 0:
        for layer in base_model.layers[:-num_fine_tune_layers]:
            layer.trainable = False
        logger.info(f"Fine-tuning top {num_fine_tune_layers} layers.")

    optimizer_stage2 = tf.keras.optimizers.Adam(
        #learning_rate=optimizer_cfg.get('stage2_learning_rate', 1e-4),
        learning_rate=cosine_schedule,
        clipnorm=optimizer_cfg.get('clipnorm', 1.0)
    )
    model.compile(
        optimizer=optimizer_stage2,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=config['loss']['params']['label_smoothing']),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')],
        jit_compile=False
    )

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_CHECKPOINT_PATH, monitor='val_accuracy', save_best_only=True,
            mode='max', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, mode='max',
            restore_best_weights=True, verbose=1
        ),
        #tf.keras.callbacks.ReduceLROnPlateau(
        #    monitor='val_loss', factor=0.2, patience=3, mode='min',
        #    min_lr=1e-7, verbose=1
        #),
    ]

    initial_epoch_to_start = training_cfg.get('stage1_epochs', 5) if not args.resume_training else 41

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training_cfg.get('stage2_epochs', 50),
        initial_epoch=initial_epoch_to_start,
        steps_per_epoch=num_train // data_cfg['batch_size'],
        validation_steps=num_val // data_cfg['batch_size'],
        callbacks=callbacks_list
    )
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    overfitting_gap = final_train_acc - final_val_acc
    
    logger.info(f"Final training accuracy: {final_train_acc:.4f}")
    logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
    logger.info(f"Overfitting gap: {overfitting_gap:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a food classification model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--base_data_dir", type=str, default=None, help="Absolute path to images.")
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Set this flag to load weights from best_classification_model.keras and continue training."
    )
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default="/kaggle/working/best_classification_model.keras",
        help="Path to the model checkpoint file."
    )
    args = parser.parse_args()
    main(args)